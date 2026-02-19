#!/usr/bin/env python3
"""
ByteTrack MOT17-04 Comparison Test

Runs baseline vs improved configurations on MOT17-04 only.
"""

import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path


TRACKER_YAML = Path("boxmot/configs/trackers/bytetrack.yaml")
PROJECT_DIR = Path("runs/ablation")
SEQ_NAME = "MOT17-04"
SEQ_SRC = Path("boxmot/engine/trackeval/MOT17-ablation/train/MOT17-04")
SUBSET_ROOT = PROJECT_DIR / "_mot17_04_subset"


BASELINE_CONFIG = {
    "new_track_thresh": 0.6,
    "entry_margin": 0,
    "strict_entry_gate": False,
    "birth_confirm_frames": 1,
    "birth_suppress_iou": 0.0,
    "birth_suppress_center_dist": 0,
    "zombie_max_history": 0,
    "zombie_dist_thresh": 999999,
    "zombie_max_predict_frames": 0,
    "zombie_transition_frames": 30,
    "zombie_match_max_dist": 200,
    "lost_max_history": 0,
    "exit_zone_enabled": False,
    "exit_zone_margin": 50,
    "exit_zone_remove_grace": 30,
    "adaptive_zone_enabled": False,
    "adaptive_zone_update_mode": "warmup_once",
    "adaptive_zone_expand_trigger": "all_high",
    "adaptive_zone_min_box_area": 0,
}


IMPROVED_CONFIG = {
    "new_track_thresh": 0.65,
    "entry_margin": 50,
    "strict_entry_gate": False,
    "birth_confirm_frames": 2,
    "birth_suppress_iou": 0.7,
    "birth_suppress_center_dist": 35,
    "zombie_max_history": 100,
    "zombie_dist_thresh": 150,
    "zombie_max_predict_frames": 5,
    "zombie_transition_frames": 30,
    "zombie_match_max_dist": 200,
    "lost_max_history": 0,
    "exit_zone_enabled": True,
    "exit_zone_margin": 50,
    "exit_zone_remove_grace": 30,
    "adaptive_zone_enabled": True,
    "adaptive_zone_update_mode": "always_expand",
    "adaptive_zone_expand_trigger": "all_high",
    "adaptive_zone_min_box_area": 0,
}


def update_tracker_config(config: dict):
    content = TRACKER_YAML.read_text()
    for key, value in config.items():
        pattern = rf"^({key}:\s*\n\s+type:\s*\w+\s*\n\s+default:)\s*[^\n]*"
        replacement = rf"\1 {value}"
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    TRACKER_YAML.write_text(content)
    print(f"Updated {TRACKER_YAML} with config: {config}")


def prepare_mot17_04_source() -> Path:
    if not SEQ_SRC.exists():
        raise FileNotFoundError(f"Sequence not found: {SEQ_SRC}")

    if SUBSET_ROOT.exists():
        shutil.rmtree(SUBSET_ROOT)
    SUBSET_ROOT.mkdir(parents=True, exist_ok=True)

    dst = SUBSET_ROOT / SEQ_NAME
    try:
        dst.symlink_to(SEQ_SRC.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(SEQ_SRC, dst)
    return SUBSET_ROOT


def _mot_dirs(project: Path) -> set[Path]:
    mot_root = project / "mot"
    if not mot_root.exists():
        return set()
    return {p.resolve() for p in mot_root.iterdir() if p.is_dir()}


def run_evaluation(name: str, source: Path, project: Path) -> dict:
    before = _mot_dirs(project)
    cmd = [
        sys.executable, "-m", "boxmot.engine.cli",
        "eval",
        "yolox_x_MOT17_ablation",
        "lmbn_n_duke",
        "bytetrack",
        "--source", str(source),
        "--save",
        "--name", name,
        "--verbose",
        "--project", str(project),
    ]
    print(f"\n{'=' * 80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    after = _mot_dirs(project)
    new_dirs = sorted((after - before), key=lambda p: p.stat().st_mtime)
    exp_dir = new_dirs[-1] if new_dirs else None

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "exp_dir": exp_dir,
    }


def extract_seq_metrics(exp_dir: Path, seq_name: str = SEQ_NAME) -> dict:
    if exp_dir is None:
        return {}
    detailed = exp_dir / "pedestrian_detailed.csv"
    if not detailed.exists():
        return {}

    with detailed.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("seq") == seq_name:
                return {
                    "HOTA": float(row.get("HOTA___AUC", 0.0)),
                    "MOTA": float(row.get("MOTA", 0.0)),
                    "IDF1": float(row.get("IDF1", 0.0)),
                    "IDSW": float(row.get("IDSW", 0.0)),
                    "IDs": float(row.get("IDs", 0.0)),
                }
    return {}


def print_comparison_table(baseline_metrics: dict, improved_metrics: dict):
    print("\n" + "=" * 80)
    print(f"METRICS COMPARISON ({SEQ_NAME})")
    print("=" * 80)
    print(f"{'Metric':<15} {'Baseline':>15} {'Improved':>15} {'Delta':>15} {'Change %':>15}")
    print("-" * 80)

    metrics_to_compare = ["HOTA", "MOTA", "IDF1", "IDSW", "IDs"]
    for metric in metrics_to_compare:
        baseline_val = baseline_metrics.get(metric, 0)
        improved_val = improved_metrics.get(metric, 0)
        delta = improved_val - baseline_val

        if baseline_val != 0:
            change_pct = (delta / baseline_val) * 100
        else:
            change_pct = 0

        if metric in {"IDSW", "IDs"}:
            direction = "better" if delta < 0 else "worse"
        else:
            direction = "better" if delta > 0 else "worse"

        print(f"{metric:<15} {baseline_val:>15.2f} {improved_val:>15.2f} {delta:>+15.2f} {change_pct:>+14.1f}% ({direction})")

    print("=" * 80)


def main():
    original_config = TRACKER_YAML.read_text()
    source = prepare_mot17_04_source()

    results = {}
    baseline_metrics = {}
    improved_metrics = {}

    try:
        print("\n" + "=" * 80)
        print("STEP 1: BASELINE TEST (new features disabled)")
        print("=" * 80)
        update_tracker_config(BASELINE_CONFIG)
        results["baseline"] = run_evaluation("bytetrack_baseline_mot17_04", source, PROJECT_DIR)
        baseline_metrics = extract_seq_metrics(results["baseline"]["exp_dir"])

        print("\n" + "=" * 80)
        print("STEP 2: IMPROVED TEST (new features enabled)")
        print("=" * 80)
        update_tracker_config(IMPROVED_CONFIG)
        results["improved"] = run_evaluation("bytetrack_improved_mot17_04", source, PROJECT_DIR)
        improved_metrics = extract_seq_metrics(results["improved"]["exp_dir"])
    finally:
        TRACKER_YAML.write_text(original_config)
        print("\n" + "=" * 80)
        print("Restored original tracker config")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    for test_name, result in results.items():
        status = "SUCCESS" if result["returncode"] == 0 else "FAILED"
        print(f"\n{test_name.upper()}: {status}")
        if result["exp_dir"] is not None:
            print(f"  Result dir: {result['exp_dir']}")
        if result["returncode"] != 0:
            print(f"  Error: {result['stderr'][:300]}")

    if baseline_metrics and improved_metrics:
        print_comparison_table(baseline_metrics, improved_metrics)
    else:
        print("\nFailed to extract per-sequence metrics from pedestrian_detailed.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
