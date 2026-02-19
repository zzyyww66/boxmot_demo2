#!/usr/bin/env python3
"""
MOT17-04 parameter sweep for improved ByteTrack configuration.

Search strategy:
1) Phase A: 16 single-factor trials around current improved config
2) Phase B: 12 interaction trials from top-4 sensitive parameters
3) Re-run top-3 candidates twice for stability

Guardrail:
- HOTA/MOTA/IDF1 each can drop by at most 0.10 against reference improved config.
Primary objective:
- Minimize IDSW under guardrail.
"""

from __future__ import annotations

import itertools
import json
import math
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace

from boxmot.engine.evaluator import run_generate_mot_results, run_trackeval
from run_mot17_04_comparison import (
    IMPROVED_CONFIG,
    PROJECT_DIR,
    TRACKER_YAML,
    prepare_mot17_04_source,
    update_tracker_config,
)

SEQ_NAME = "MOT17-04"
GUARDRAIL_DROP = 0.10
REPORT_DIR = PROJECT_DIR / "param_sweep"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


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


def extract_seq_metrics(results: dict, seq: str = SEQ_NAME) -> dict:
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


def run_once(config: dict, run_name: str, source: Path) -> dict:
    update_tracker_config(config)
    args = make_args(source, run_name)
    run_generate_mot_results(args)
    results = run_trackeval(args, verbose=True)
    return extract_seq_metrics(results)


def pass_guardrail(metrics: dict, reference: dict, max_drop: float = GUARDRAIL_DROP) -> bool:
    return (
        metrics["HOTA"] >= reference["HOTA"] - max_drop
        and metrics["MOTA"] >= reference["MOTA"] - max_drop
        and metrics["IDF1"] >= reference["IDF1"] - max_drop
    )


def sort_key(row: dict) -> tuple:
    m = row["metrics"]
    return (m["IDSW"], m["IDs"], -m["IDF1"], -m["MOTA"], -m["HOTA"])


def metric_delta(metrics: dict, reference: dict) -> dict:
    return {k: metrics[k] - reference[k] for k in ("HOTA", "MOTA", "IDF1", "IDSW", "IDs")}


def sanitize(v) -> str:
    return str(v).replace(".", "p")


def phase_a_cases(base: dict) -> list[dict]:
    points = [
        ("entry_margin", [40, 60]),
        ("exit_zone_margin", [35, 65]),
        ("exit_zone_remove_grace", [24, 36, 42]),
        ("zombie_dist_thresh", [130, 170]),
        ("zombie_max_predict_frames", [4, 6]),
        ("adaptive_zone_expand_trigger", ["outside_high"]),
        ("adaptive_zone_margin", [40, 60]),
        ("adaptive_zone_min_box_area", [200, 400]),
    ]
    cases = []
    idx = 1
    for param, values in points:
        for value in values:
            cfg = deepcopy(base)
            cfg[param] = value
            cases.append(
                {
                    "phase": "A",
                    "index": idx,
                    "param": param,
                    "value": value,
                    "name": f"sweep_A{idx:02d}_{param}_{sanitize(value)}",
                    "config": cfg,
                }
            )
            idx += 1
    assert len(cases) == 16
    return cases


def pick_top4_params(phase_a_results: list[dict], reference: dict) -> list[str]:
    by_param = {}
    for row in phase_a_results:
        by_param.setdefault(row["param"], []).append(row)

    scored = []
    for param, rows in by_param.items():
        valid = [r for r in rows if r["pass_guardrail"]]
        target = valid if valid else rows
        best = min(target, key=sort_key)
        score = reference["IDSW"] - best["metrics"]["IDSW"]
        scored.append((score, -best["metrics"]["IDs"], best["metrics"]["IDF1"], param))

    scored.sort(reverse=True)
    top = [t[3] for t in scored[:4]]

    if len(top) < 4:
        fallback = [
            "exit_zone_remove_grace",
            "exit_zone_margin",
            "entry_margin",
            "zombie_dist_thresh",
            "adaptive_zone_margin",
        ]
        for p in fallback:
            if p not in top:
                top.append(p)
            if len(top) == 4:
                break
    return top


def pick_alt_value(param: str, base_value, phase_a_results: list[dict]):
    rows = [r for r in phase_a_results if r["param"] == param and r["value"] != base_value]
    valid = [r for r in rows if r["pass_guardrail"]]
    source = valid if valid else rows
    if not source:
        return base_value
    best = min(source, key=sort_key)
    return best["value"]


def phase_b_cases(base: dict, phase_a_results: list[dict], reference: dict) -> list[dict]:
    top4 = pick_top4_params(phase_a_results, reference)
    candidates = {}
    for p in top4:
        base_value = base[p]
        alt_value = pick_alt_value(p, base_value, phase_a_results)
        if alt_value == base_value:
            rows = [r for r in phase_a_results if r["param"] == p and r["value"] != base_value]
            if rows:
                alt_value = rows[0]["value"]
        candidates[p] = [base_value, alt_value]

    combos = []
    seen = set()
    bit_patterns = list(itertools.product([0, 1], repeat=4))
    bit_patterns = [b for b in bit_patterns if b != (0, 0, 0, 0)]
    bit_patterns.sort(key=lambda b: (-sum(b), b))

    idx = 1
    for bits in bit_patterns:
        cfg = deepcopy(base)
        changed = []
        for i, p in enumerate(top4):
            value = candidates[p][bits[i]]
            cfg[p] = value
            if bits[i] == 1:
                changed.append((p, value))

        signature = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
        if signature in seen:
            continue
        seen.add(signature)
        combos.append(
            {
                "phase": "B",
                "index": idx,
                "name": f"sweep_B{idx:02d}_" + "_".join(f"{k}_{sanitize(v)}" for k, v in changed),
                "changed": changed,
                "config": cfg,
                "top4": top4,
            }
        )
        idx += 1
        if len(combos) == 12:
            break

    return combos


def rerun_case(base_name: str, cfg: dict, source: Path, rep: int) -> dict:
    name = f"{base_name}_rerun{rep}"
    metrics = run_once(cfg, name, source)
    return {"name": name, "metrics": metrics}


def print_row(row: dict, reference: dict):
    m = row["metrics"]
    d = metric_delta(m, reference)
    status = "PASS" if row["pass_guardrail"] else "FAIL"
    print(
        f"{row['name']:<50} {status:<4} "
        f"HOTA {m['HOTA']:.3f} ({d['HOTA']:+.3f})  "
        f"MOTA {m['MOTA']:.3f} ({d['MOTA']:+.3f})  "
        f"IDF1 {m['IDF1']:.3f} ({d['IDF1']:+.3f})  "
        f"IDSW {m['IDSW']:.0f} ({d['IDSW']:+.0f})  "
        f"IDs {m['IDs']:.0f} ({d['IDs']:+.0f})"
    )


def main() -> int:
    source = prepare_mot17_04_source()
    original_yaml = TRACKER_YAML.read_text()
    reference_cfg = deepcopy(IMPROVED_CONFIG)

    report = {
        "dataset": SEQ_NAME,
        "guardrail_drop": GUARDRAIL_DROP,
        "reference_config": reference_cfg,
        "reference_metrics": None,
        "phase_a": [],
        "phase_b": [],
        "top3_reruns": [],
        "best_candidate": None,
    }

    try:
        print("=" * 96)
        print("REFERENCE RUN (current improved config)")
        print("=" * 96)
        reference_metrics = run_once(reference_cfg, "sweep_reference_improved", source)
        report["reference_metrics"] = reference_metrics
        print(
            "Reference:"
            f" HOTA={reference_metrics['HOTA']:.3f}"
            f" MOTA={reference_metrics['MOTA']:.3f}"
            f" IDF1={reference_metrics['IDF1']:.3f}"
            f" IDSW={reference_metrics['IDSW']:.0f}"
            f" IDs={reference_metrics['IDs']:.0f}"
        )

        phase_a = phase_a_cases(reference_cfg)
        print("\n" + "=" * 96)
        print("PHASE A (18 single-factor trials)")
        print("=" * 96)
        for case in phase_a:
            metrics = run_once(case["config"], case["name"], source)
            row = {
                "phase": case["phase"],
                "index": case["index"],
                "name": case["name"],
                "param": case["param"],
                "value": case["value"],
                "config": case["config"],
                "metrics": metrics,
                "pass_guardrail": pass_guardrail(metrics, reference_metrics),
            }
            report["phase_a"].append(row)
            print_row(row, reference_metrics)

        phase_b = phase_b_cases(reference_cfg, report["phase_a"], reference_metrics)
        print("\n" + "=" * 96)
        print("PHASE B (12 interaction trials)")
        print("=" * 96)
        print(f"Top-4 sensitive parameters: {phase_b[0]['top4'] if phase_b else []}")
        for case in phase_b:
            metrics = run_once(case["config"], case["name"], source)
            row = {
                "phase": case["phase"],
                "index": case["index"],
                "name": case["name"],
                "changed": case["changed"],
                "config": case["config"],
                "metrics": metrics,
                "pass_guardrail": pass_guardrail(metrics, reference_metrics),
            }
            report["phase_b"].append(row)
            print_row(row, reference_metrics)

        combined = report["phase_a"] + report["phase_b"]
        valid = [r for r in combined if r["pass_guardrail"]]
        valid.sort(key=sort_key)

        if not valid:
            print("\nNo candidate passed guardrail.")
            return 1

        top3 = valid[:3]
        print("\n" + "=" * 96)
        print("TOP-3 STABILITY RE-RUNS (2 repeats each)")
        print("=" * 96)
        for idx, row in enumerate(top3, start=1):
            reruns = []
            for rep in (1, 2):
                r = rerun_case(row["name"], row["config"], source, rep)
                reruns.append(r)
                print(
                    f"Top{idx} {r['name']:<60}"
                    f"IDSW={r['metrics']['IDSW']:.0f} IDs={r['metrics']['IDs']:.0f} "
                    f"IDF1={r['metrics']['IDF1']:.3f} MOTA={r['metrics']['MOTA']:.3f} HOTA={r['metrics']['HOTA']:.3f}"
                )
            report["top3_reruns"].append({"base": row, "reruns": reruns})

        stability_rows = []
        for item in report["top3_reruns"]:
            base = item["base"]
            metrics_all = [base["metrics"]] + [r["metrics"] for r in item["reruns"]]
            agg = {}
            for k in ("HOTA", "MOTA", "IDF1", "IDSW", "IDs"):
                values = [m[k] for m in metrics_all]
                agg[k] = {
                    "mean": mean(values),
                    "std": pstdev(values) if len(values) > 1 else 0.0,
                }
            stability_rows.append({"base": base, "agg": agg})

        stability_rows.sort(
            key=lambda x: (
                x["agg"]["IDSW"]["mean"],
                x["agg"]["IDs"]["mean"],
                -x["agg"]["IDF1"]["mean"],
                -x["agg"]["MOTA"]["mean"],
                -x["agg"]["HOTA"]["mean"],
                x["agg"]["IDSW"]["std"] + x["agg"]["IDs"]["std"],
            )
        )
        best = stability_rows[0]
        report["best_candidate"] = best

        print("\n" + "=" * 96)
        print("FINAL BEST CANDIDATE")
        print("=" * 96)
        b = best["base"]
        a = best["agg"]
        print(f"Name: {b['name']}")
        print(f"Config overrides from improved:")
        for k, v in b["config"].items():
            if reference_cfg.get(k) != v:
                print(f"  - {k}: {reference_cfg.get(k)} -> {v}")
        print("Averaged over 3 runs (base + 2 reruns):")
        print(
            f"  HOTA={a['HOTA']['mean']:.3f}±{a['HOTA']['std']:.3f}, "
            f"MOTA={a['MOTA']['mean']:.3f}±{a['MOTA']['std']:.3f}, "
            f"IDF1={a['IDF1']['mean']:.3f}±{a['IDF1']['std']:.3f}, "
            f"IDSW={a['IDSW']['mean']:.3f}±{a['IDSW']['std']:.3f}, "
            f"IDs={a['IDs']['mean']:.3f}±{a['IDs']['std']:.3f}"
        )
        d = metric_delta({k: a[k]["mean"] for k in ("HOTA", "MOTA", "IDF1", "IDSW", "IDs")}, reference_metrics)
        print(
            f"  Delta vs reference: HOTA {d['HOTA']:+.3f}, MOTA {d['MOTA']:+.3f}, "
            f"IDF1 {d['IDF1']:+.3f}, IDSW {d['IDSW']:+.3f}, IDs {d['IDs']:+.3f}"
        )

        out_json = REPORT_DIR / "mot17_04_param_sweep_results.json"
        serializable = json.loads(json.dumps(report, default=str))
        out_json.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))
        print(f"\nSaved report: {out_json}")
        return 0
    finally:
        TRACKER_YAML.write_text(original_yaml)
        print("\nRestored tracker yaml defaults.")


if __name__ == "__main__":
    raise SystemExit(main())
