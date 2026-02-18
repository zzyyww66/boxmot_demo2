#!/usr/bin/env python3
"""
Run MOT17-04 baseline vs improved using cached det/emb only.

This script skips detection/embedding generation and only runs:
1) tracker association (run_generate_mot_results)
2) TrackEval evaluation (run_trackeval)
"""

from pathlib import Path
from types import SimpleNamespace

from boxmot.engine.evaluator import run_generate_mot_results, run_trackeval
from run_mot17_04_comparison import (
    BASELINE_CONFIG,
    IMPROVED_CONFIG,
    PROJECT_DIR,
    TRACKER_YAML,
    prepare_mot17_04_source,
    update_tracker_config,
)


def make_args(source: Path, name: str) -> SimpleNamespace:
    return SimpleNamespace(
        source=source,
        project=PROJECT_DIR,
        name=name,
        yolo_model=[Path("yolox_x_MOT17_ablation")],
        reid_model=[Path("lmbn_n_duke")],
        tracking_method="bytetrack",
        fps=30,
        device="",
        split="train",
        benchmark="MOT17-ablation",
        classes=None,
        postprocessing="none",
        ci=False,
    )


def extract_seq_metrics(results: dict, seq: str = "MOT17-04") -> dict:
    per_seq = results.get("per_sequence", {}) if isinstance(results, dict) else {}
    seq_metrics = per_seq.get(seq, {}) if isinstance(per_seq, dict) else {}
    m = seq_metrics if seq_metrics else results
    return {
        "HOTA": float(m.get("HOTA", 0.0)),
        "MOTA": float(m.get("MOTA", 0.0)),
        "IDF1": float(m.get("IDF1", 0.0)),
        "IDSW": float(m.get("IDSW", 0.0)),
        "IDs": float(m.get("IDs", 0.0)),
    }


def run_once(config: dict, run_name: str, source: Path) -> tuple[dict, Path]:
    update_tracker_config(config)
    args = make_args(source, run_name)
    run_generate_mot_results(args)
    results = run_trackeval(args, verbose=True)
    metrics = extract_seq_metrics(results)
    return metrics, args.exp_dir


def main() -> int:
    source = prepare_mot17_04_source()
    original_config = TRACKER_YAML.read_text()

    baseline = {}
    improved = {}
    baseline_dir = None
    improved_dir = None

    try:
        print("=" * 88)
        print("BASELINE: association + evaluation only")
        print("=" * 88)
        baseline, baseline_dir = run_once(BASELINE_CONFIG, "assoc_eval_baseline_mot17_04", source)

        print("\n" + "=" * 88)
        print("IMPROVED: association + evaluation only")
        print("=" * 88)
        improved, improved_dir = run_once(IMPROVED_CONFIG, "assoc_eval_improved_mot17_04", source)
    finally:
        TRACKER_YAML.write_text(original_config)
        print("\nRestored tracker config.")

    print("\n" + "=" * 88)
    print("MOT17-04 COMPARISON (association + evaluation only)")
    print("=" * 88)
    print(f"Baseline dir: {baseline_dir}")
    print(f"Improved dir: {improved_dir}")
    for key in ["HOTA", "MOTA", "IDF1", "IDSW", "IDs"]:
        b = baseline.get(key, 0.0)
        i = improved.get(key, 0.0)
        print(f"{key:>5}: baseline={b:8.3f} improved={i:8.3f} delta={i - b:+8.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
