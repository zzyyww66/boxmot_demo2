#!/usr/bin/env python3
"""Systematically tune the improved ByteTrack config on SOMPT22 with cached dets/embs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from boxmot.engine.evaluator import (
    eval_init,
    run_generate_dets_embs,
    run_generate_mot_results,
    run_trackeval,
)
from boxmot.utils import ROOT, logger as LOGGER


DEFAULT_SOURCE = ROOT / "train"
DEFAULT_CACHE_ROOT = (
    ROOT / "runs_sompt22_person_cache_yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17"
)
DEFAULT_OUTPUT_ROOT = ROOT / "runs_tune_sompt22_bytetrack"
DEFAULT_TRACKER_CONFIG = ROOT / "boxmot/configs/trackers/bytetrack_improved.yaml"
DEFAULT_BEST_CONFIG = ROOT / "boxmot/configs/trackers/bytetrack_sompt22_tuned.yaml"
DEFAULT_YOLO = ROOT / "yolov8m_pretrain_crowdhuman.pt"
DEFAULT_REID = ROOT / "osnet_x0_25_msmt17.pt"
DEFAULT_COARSE_SEQUENCES = ["SOMPT22-07", "SOMPT22-08", "SOMPT22-10", "SOMPT22-13"]


SEARCH_CANDIDATES: dict[str, list[Any]] = {
    "track_thresh": [0.45, 0.50, 0.55],
    "new_track_thresh": [0.60, 0.65, 0.70],
    "match_thresh": [0.76, 0.80, 0.84],
    "birth_confirm_frames": [1, 2, 3],
    "birth_suppress_iou": [0.50, 0.70, 0.80],
    "birth_suppress_center_dist": [25, 35, 45],
    "entry_margin": [40, 50, 60],
    "strict_entry_gate": [False, True],
    "exit_zone_remove_grace": [20, 30],
    "lost_reid_enabled": [False, True],
    "lost_match_max_dist": [100, 120, 140],
    "lost_reid_max_frames": [10, 15, 20],
    "lost_reid_thresh": [0.20, 0.25, 0.30],
    "lost_match_cost_thresh": [0.28, 0.35, 0.42],
    "zombie_reid_enabled": [False, True],
    "zombie_transition_frames": [25, 30, 35],
    "zombie_dist_thresh": [130, 150, 170],
    "zombie_reid_thresh": [0.30, 0.35, 0.40],
    "zombie_match_cost_thresh": [0.38, 0.45, 0.52],
    "adaptive_zone_update_mode": ["warmup_once", "always_expand"],
    "adaptive_zone_expand_trigger": ["outside_high", "unmatched_high", "all_high"],
    "adaptive_zone_entry_mode": ["outside_only", "margin_inside"],
    "adaptive_zone_margin": [40, 50, 60],
    "spatial_prior_enabled": [False, True],
    "spatial_prior_region_enabled": [False, True],
    "spatial_prior_entry_mode": ["bias_only", "strict_region"],
    "spatial_prior_region_birth": [0.80, 0.85, 0.90],
    "spatial_prior_region_birth_grow": [0.50, 0.60, 0.70],
    "spatial_prior_entry_support_threshold": [80, 100, 120],
    "spatial_prior_entry_birth_threshold": [6, 8, 10],
}


@dataclass
class TrialResult:
    trial_id: int
    phase: str
    name: str
    scope: str
    sequence_filter: list[str] | None
    config: dict[str, Any]
    metrics: dict[str, Any]
    elapsed_sec: float
    exp_dir: str
    config_path: str

    @property
    def rank_tuple(self) -> tuple[float, float, float, float, float]:
        return metric_rank_tuple(self.metrics)

    @property
    def score(self) -> float:
        metrics = self.metrics
        return (
            float(metrics.get("HOTA", 0.0))
            + 0.10 * float(metrics.get("MOTA", 0.0))
            + 0.10 * float(metrics.get("IDF1", 0.0))
            - 0.002 * float(metrics.get("IDSW", 0.0))
            - 0.0005 * float(metrics.get("IDs", 0.0))
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "phase": self.phase,
            "name": self.name,
            "scope": self.scope,
            "sequence_filter": self.sequence_filter,
            "config": self.config,
            "metrics": self.metrics,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "exp_dir": self.exp_dir,
            "config_path": self.config_path,
            "rank_tuple": list(self.rank_tuple),
            "score": round(self.score, 6),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "TrialResult":
        return cls(
            trial_id=int(record["trial_id"]),
            phase=str(record["phase"]),
            name=str(record["name"]),
            scope=str(record["scope"]),
            sequence_filter=record.get("sequence_filter"),
            config=dict(record["config"]),
            metrics=dict(record["metrics"]),
            elapsed_sec=float(record["elapsed_sec"]),
            exp_dir=str(record["exp_dir"]),
            config_path=str(record["config_path"]),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Systematically tune the improved ByteTrack variant for SOMPT22 while "
            "reusing cached detections and embeddings."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tracker-config", type=Path, default=DEFAULT_TRACKER_CONFIG)
    parser.add_argument("--write-best-config", type=Path, default=DEFAULT_BEST_CONFIG)
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--reid-model", type=Path, default=DEFAULT_REID)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--classes", type=int, nargs="*", default=[0])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--imgsz", type=int, nargs=2, default=[1088, 1920])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--read-threads", type=int, default=None)
    parser.add_argument(
        "--coarse-sequences",
        nargs="*",
        default=DEFAULT_COARSE_SEQUENCES,
        help="Subset used for the broad but faster coarse search.",
    )
    parser.add_argument(
        "--full-sequences",
        nargs="*",
        default=None,
        help="Optional explicit full-eval subset; defaults to all SOMPT22 train sequences.",
    )
    parser.add_argument(
        "--max-full-candidates",
        type=int,
        default=6,
        help="How many top coarse configs to re-evaluate on the full split.",
    )
    parser.add_argument(
        "--skip-coarse",
        action="store_true",
        help="Skip the coarse pass and only evaluate the baseline on the full split.",
    )
    parser.add_argument(
        "--keep-existing-output",
        action="store_true",
        help="Reuse the existing output root if it already exists.",
    )
    return parser.parse_args()


def load_tracker_config(path: Path) -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Tracker config {path} is not a YAML mapping")
    return config


def tracker_defaults(raw_cfg: dict[str, dict[str, Any]]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for key, details in raw_cfg.items():
        if not isinstance(details, dict) or "default" not in details:
            raise ValueError(f"Tracker config entry {key!r} must contain a default value")
        defaults[key] = details["default"]
    return defaults


def ensure_output_root(output_root: Path, cache_root: Path, keep_existing: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()) and not keep_existing:
        raise FileExistsError(
            f"Output root already exists and is not empty: {output_root}. "
            "Pass --keep-existing-output to reuse it."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    src = cache_root / "dets_n_embs"
    if not src.exists():
        raise FileNotFoundError(f"Cached dets/embs not found: {src}")

    dst = output_root / "dets_n_embs"
    if dst.is_symlink():
        target = dst.resolve()
        if target != src.resolve():
            raise RuntimeError(f"{dst} already points to {target}, expected {src}")
        return
    if dst.exists():
        raise RuntimeError(f"{dst} already exists and is not a symlink")
    dst.symlink_to(src.resolve(), target_is_directory=True)


def build_eval_args(
    args: argparse.Namespace,
    project_root: Path,
    sequence_filter: list[str] | None,
) -> SimpleNamespace:
    bench = "SOMPT22"
    opt = SimpleNamespace(
        source=args.source,
        project=project_root,
        name="",
        yolo_model=[args.yolo_model],
        reid_model=[args.reid_model],
        tracking_method="bytetrack",
        tracker_config=args.tracker_config,
        classes=list(args.classes),
        device=args.device,
        fps=args.fps,
        imgsz=list(args.imgsz),
        benchmark=bench,
        split="train",
        dataset_config_name="SOMPT22-full",
        sequence_filter=list(sequence_filter) if sequence_filter else None,
        batch_size=args.batch_size,
        read_threads=args.read_threads,
        auto_batch=True,
        resume=True,
        half=False,
        ci=False,
        postprocessing="none",
        verbose=False,
    )
    eval_init(opt)
    return opt


def metric_rank_tuple(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(metrics.get("HOTA", 0.0)),
        float(metrics.get("MOTA", 0.0)),
        float(metrics.get("IDF1", 0.0)),
        -float(metrics.get("IDSW", 0.0)),
        -float(metrics.get("IDs", 0.0)),
    )


def better_metrics(lhs: dict[str, Any], rhs: dict[str, Any]) -> bool:
    return metric_rank_tuple(lhs) > metric_rank_tuple(rhs)


def parse_summary_metrics(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        raise FileNotFoundError(f"TrackEval summary file not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if len(lines) < 2:
        raise ValueError(f"Unexpected summary format in {summary_path}")

    headers = lines[0].split()
    values = lines[1].split()
    if len(headers) != len(values):
        raise ValueError(
            f"Summary header/value length mismatch in {summary_path}: "
            f"{len(headers)} vs {len(values)}"
        )

    metrics: dict[str, Any] = {}
    for key, raw_value in zip(headers, values):
        try:
            if raw_value.isdigit():
                metrics[key] = int(raw_value)
            else:
                numeric = float(raw_value)
                metrics[key] = int(numeric) if numeric.is_integer() else numeric
        except ValueError:
            metrics[key] = raw_value
    return metrics


def config_key(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_config_from_defaults(
    raw_template: dict[str, dict[str, Any]],
    defaults: dict[str, Any],
    output_path: Path,
) -> None:
    rendered = copy.deepcopy(raw_template)
    for key, value in defaults.items():
        if key in rendered and isinstance(rendered[key], dict):
            rendered[key]["default"] = value
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(rendered, handle, sort_keys=False)


def log_headline(prefix: str, metrics: dict[str, Any], elapsed_sec: float) -> None:
    LOGGER.info(
        (
            f"{prefix} | HOTA={metrics.get('HOTA', 0.0):.3f} "
            f"MOTA={metrics.get('MOTA', 0.0):.3f} "
            f"IDF1={metrics.get('IDF1', 0.0):.3f} "
            f"IDSW={int(metrics.get('IDSW', 0))} "
            f"IDs={int(metrics.get('IDs', 0))} "
            f"time={elapsed_sec / 60.0:.1f}m"
        )
    )


def persist_results(output_root: Path, results: list[TrialResult]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_root / "trial_results.jsonl"
    csv_path = output_root / "leaderboard.csv"
    md_path = output_root / "leaderboard.md"

    ordered_by_trial = sorted(results, key=lambda item: item.trial_id)
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for result in ordered_by_trial:
            handle.write(json.dumps(result.to_record(), ensure_ascii=True) + "\n")

    ranked = sorted(results, key=lambda item: item.rank_tuple, reverse=True)

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "trial_id",
                "phase",
                "name",
                "scope",
                "HOTA",
                "MOTA",
                "IDF1",
                "IDSW",
                "IDs",
                "elapsed_sec",
                "exp_dir",
                "config_path",
            ]
        )
        for item in ranked:
            writer.writerow(
                [
                    item.trial_id,
                    item.phase,
                    item.name,
                    item.scope,
                    item.metrics.get("HOTA", 0.0),
                    item.metrics.get("MOTA", 0.0),
                    item.metrics.get("IDF1", 0.0),
                    item.metrics.get("IDSW", 0.0),
                    item.metrics.get("IDs", 0.0),
                    round(item.elapsed_sec, 3),
                    item.exp_dir,
                    item.config_path,
                ]
            )

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("| rank | phase | name | scope | HOTA | MOTA | IDF1 | IDSW | IDs |\n")
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for rank, item in enumerate(ranked, start=1):
            handle.write(
                "| "
                f"{rank} | {item.phase} | {item.name} | {item.scope} | "
                f"{item.metrics.get('HOTA', 0.0):.3f} | "
                f"{item.metrics.get('MOTA', 0.0):.3f} | "
                f"{item.metrics.get('IDF1', 0.0):.3f} | "
                f"{int(item.metrics.get('IDSW', 0))} | "
                f"{int(item.metrics.get('IDs', 0))} |\n"
            )


def load_existing_results(output_root: Path) -> dict[int, TrialResult]:
    jsonl_path = output_root / "trial_results.jsonl"
    if not jsonl_path.exists():
        return {}

    loaded: dict[int, TrialResult] = {}
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            result = TrialResult.from_record(json.loads(line))
            loaded[result.trial_id] = result
    return loaded


def upsert_result(results_by_id: dict[int, TrialResult], result: TrialResult) -> None:
    results_by_id[result.trial_id] = result


def append_unique_result(results: list[TrialResult], result: TrialResult) -> None:
    for idx, existing in enumerate(results):
        if existing.trial_id == result.trial_id:
            results[idx] = result
            return
    results.append(result)


def prepare_cached_dets_embs(args: argparse.Namespace, project_root: Path) -> None:
    LOGGER.info("Preparing cache link and verifying cached dets/embs...")
    prep_args = build_eval_args(args, project_root=project_root, sequence_filter=None)
    run_generate_dets_embs(prep_args)


def evaluate_trial(
    args: argparse.Namespace,
    raw_tracker_cfg: dict[str, dict[str, Any]],
    trial_root: Path,
    trial_id: int,
    phase: str,
    name: str,
    scope: str,
    sequence_filter: list[str] | None,
    config: dict[str, Any],
    existing_results: dict[int, TrialResult] | None = None,
) -> TrialResult:
    existing = (existing_results or {}).get(trial_id)
    normalized_sequence_filter = list(sequence_filter) if sequence_filter else None
    if existing is not None:
        if (
            existing.phase == phase
            and existing.name == name
            and existing.scope == scope
            and existing.sequence_filter == normalized_sequence_filter
            and config_key(existing.config) == config_key(config)
        ):
            LOGGER.info(
                f"[trial {trial_id:03d}] resume hit for {phase}/{name}/{scope}; skipping rerun"
            )
            return existing
        raise RuntimeError(
            f"Existing trial {trial_id} does not match the expected search plan. "
            "Use a fresh --output-root for a different tuning run."
        )

    config_dir = trial_root / "trial_configs"
    config_path = config_dir / f"trial_{trial_id:03d}_{name}.yaml"
    write_config_from_defaults(raw_tracker_cfg, config, config_path)

    opt = build_eval_args(args, project_root=trial_root, sequence_filter=sequence_filter)

    start = time.perf_counter()
    run_generate_mot_results(opt, evolve_config=config)
    run_trackeval(opt, verbose=False)
    metrics = parse_summary_metrics(opt.exp_dir / "pedestrian_summary.txt")
    elapsed = time.perf_counter() - start

    result = TrialResult(
        trial_id=trial_id,
        phase=phase,
        name=name,
        scope=scope,
        sequence_filter=normalized_sequence_filter,
        config=copy.deepcopy(config),
        metrics=metrics,
        elapsed_sec=elapsed,
        exp_dir=str(opt.exp_dir),
        config_path=str(config_path),
    )
    log_headline(f"[trial {trial_id:03d}] {phase}/{name}/{scope}", metrics, elapsed)
    return result


def unique_top_configs(
    results: list[TrialResult],
    limit: int,
) -> list[TrialResult]:
    ranked = sorted(results, key=lambda item: item.rank_tuple, reverse=True)
    seen: set[str] = set()
    top: list[TrialResult] = []
    for item in ranked:
        key = config_key(item.config)
        if key in seen:
            continue
        seen.add(key)
        top.append(item)
        if len(top) >= limit:
            break
    return top


def main() -> None:
    args = parse_args()

    args.source = args.source.resolve()
    args.cache_root = args.cache_root.resolve()
    args.output_root = args.output_root.resolve()
    args.tracker_config = args.tracker_config.resolve()
    args.write_best_config = args.write_best_config.resolve()
    args.yolo_model = args.yolo_model.resolve()
    args.reid_model = args.reid_model.resolve()

    if not args.source.exists():
        raise FileNotFoundError(f"SOMPT22 source not found: {args.source}")
    if not args.tracker_config.exists():
        raise FileNotFoundError(f"Tracker config not found: {args.tracker_config}")
    if not args.yolo_model.exists():
        raise FileNotFoundError(f"YOLO weights not found: {args.yolo_model}")
    if not args.reid_model.exists():
        raise FileNotFoundError(f"ReID weights not found: {args.reid_model}")

    ensure_output_root(args.output_root, args.cache_root, keep_existing=args.keep_existing_output)

    raw_tracker_cfg = load_tracker_config(args.tracker_config)
    base_config = tracker_defaults(raw_tracker_cfg)
    existing_results = load_existing_results(args.output_root)
    if existing_results:
        LOGGER.info(
            f"Found {len(existing_results)} completed trials under {args.output_root}; "
            "the run will resume from the next missing trial."
        )

    prepare_cached_dets_embs(args, args.output_root)

    all_results_by_id: dict[int, TrialResult] = dict(existing_results)
    all_results: list[TrialResult] = sorted(all_results_by_id.values(), key=lambda item: item.trial_id)
    coarse_results: list[TrialResult] = []
    trial_id = 1

    baseline_full = evaluate_trial(
        args=args,
        raw_tracker_cfg=raw_tracker_cfg,
        trial_root=args.output_root,
        trial_id=trial_id,
        phase="baseline",
        name="baseline_full",
        scope="full",
        sequence_filter=args.full_sequences,
        config=base_config,
        existing_results=existing_results,
    )
    upsert_result(all_results_by_id, baseline_full)
    append_unique_result(all_results, baseline_full)
    persist_results(args.output_root, list(all_results_by_id.values()))
    trial_id += 1

    if not args.skip_coarse:
        current_best_config = copy.deepcopy(base_config)
        current_best_metrics = baseline_full.metrics
        changed_params: list[str] = []

        coarse_baseline = evaluate_trial(
            args=args,
            raw_tracker_cfg=raw_tracker_cfg,
            trial_root=args.output_root,
            trial_id=trial_id,
            phase="coarse",
            name="baseline_subset",
            scope="coarse",
            sequence_filter=args.coarse_sequences,
            config=current_best_config,
            existing_results=existing_results,
        )
        upsert_result(all_results_by_id, coarse_baseline)
        append_unique_result(all_results, coarse_baseline)
        append_unique_result(coarse_results, coarse_baseline)
        persist_results(args.output_root, list(all_results_by_id.values()))
        trial_id += 1
        current_best_metrics = coarse_baseline.metrics

        LOGGER.info(
            f"Starting coarse coordinate search over {len(SEARCH_CANDIDATES)} parameters "
            f"on {args.coarse_sequences}..."
        )

        for param_name, candidates in SEARCH_CANDIDATES.items():
            original_value = current_best_config.get(param_name)
            local_best_value = original_value
            local_best_metrics = current_best_metrics

            for candidate in candidates:
                if candidate == original_value:
                    continue
                trial_config = copy.deepcopy(current_best_config)
                trial_config[param_name] = candidate
                result = evaluate_trial(
                    args=args,
                    raw_tracker_cfg=raw_tracker_cfg,
                    trial_root=args.output_root,
                    trial_id=trial_id,
                    phase="coarse",
                    name=f"{param_name}_{str(candidate).replace(' ', '_')}",
                    scope="coarse",
                    sequence_filter=args.coarse_sequences,
                    config=trial_config,
                    existing_results=existing_results,
                )
                upsert_result(all_results_by_id, result)
                append_unique_result(all_results, result)
                append_unique_result(coarse_results, result)
                persist_results(args.output_root, list(all_results_by_id.values()))
                trial_id += 1

                if better_metrics(result.metrics, local_best_metrics):
                    local_best_metrics = result.metrics
                    local_best_value = candidate

            if local_best_value != original_value:
                current_best_config[param_name] = local_best_value
                current_best_metrics = local_best_metrics
                changed_params.append(param_name)
                LOGGER.info(
                    f"Accepted coarse improvement: {param_name} {original_value!r} -> {local_best_value!r}"
                )

        if changed_params:
            LOGGER.info(
                f"Starting coarse refinement pass over changed parameters: {changed_params}"
            )
            for param_name in changed_params:
                original_value = current_best_config.get(param_name)
                local_best_value = original_value
                local_best_metrics = current_best_metrics
                for candidate in SEARCH_CANDIDATES[param_name]:
                    if candidate == original_value:
                        continue
                    trial_config = copy.deepcopy(current_best_config)
                    trial_config[param_name] = candidate
                    result = evaluate_trial(
                        args=args,
                        raw_tracker_cfg=raw_tracker_cfg,
                        trial_root=args.output_root,
                        trial_id=trial_id,
                        phase="refine",
                        name=f"{param_name}_{str(candidate).replace(' ', '_')}",
                        scope="coarse",
                        sequence_filter=args.coarse_sequences,
                        config=trial_config,
                        existing_results=existing_results,
                    )
                    upsert_result(all_results_by_id, result)
                    append_unique_result(all_results, result)
                    append_unique_result(coarse_results, result)
                    persist_results(args.output_root, list(all_results_by_id.values()))
                    trial_id += 1

                    if better_metrics(result.metrics, local_best_metrics):
                        local_best_metrics = result.metrics
                        local_best_value = candidate

                if local_best_value != original_value:
                    current_best_config[param_name] = local_best_value
                    current_best_metrics = local_best_metrics
                    LOGGER.info(
                        f"Accepted refinement improvement: {param_name} {original_value!r} -> {local_best_value!r}"
                    )

        top_candidates = unique_top_configs(coarse_results + [baseline_full], args.max_full_candidates)
    else:
        top_candidates = [baseline_full]

    LOGGER.info(
        "Running full-split rechecks for the top coarse candidates "
        f"(count={len(top_candidates)})..."
    )
    full_rechecks: list[TrialResult] = [baseline_full]
    seen_full: set[str] = {config_key(baseline_full.config)}

    for candidate in top_candidates:
        candidate_key = config_key(candidate.config)
        if candidate_key in seen_full:
            continue
        result = evaluate_trial(
            args=args,
            raw_tracker_cfg=raw_tracker_cfg,
            trial_root=args.output_root,
            trial_id=trial_id,
            phase="full",
            name=f"full_from_trial_{candidate.trial_id:03d}",
            scope="full",
            sequence_filter=args.full_sequences,
            config=candidate.config,
            existing_results=existing_results,
        )
        upsert_result(all_results_by_id, result)
        append_unique_result(all_results, result)
        full_rechecks.append(result)
        seen_full.add(candidate_key)
        persist_results(args.output_root, list(all_results_by_id.values()))
        trial_id += 1

    best_full = max(full_rechecks, key=lambda item: item.rank_tuple)
    write_config_from_defaults(raw_tracker_cfg, best_full.config, args.write_best_config)

    summary = {
        "best_full_trial_id": best_full.trial_id,
        "best_full_name": best_full.name,
        "best_full_phase": best_full.phase,
        "best_full_metrics": best_full.metrics,
        "best_full_exp_dir": best_full.exp_dir,
        "best_full_config_path": best_full.config_path,
        "emitted_best_config": str(args.write_best_config),
        "baseline_full_trial_id": baseline_full.trial_id,
        "baseline_full_metrics": baseline_full.metrics,
        "coarse_sequences": args.coarse_sequences,
        "full_sequences": args.full_sequences,
        "search_parameters": list(SEARCH_CANDIDATES.keys()),
        "total_trials": len(all_results),
    }
    with open(args.output_root / "best_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    persist_results(args.output_root, list(all_results_by_id.values()))

    LOGGER.info("")
    LOGGER.info("Best full-split config:")
    log_headline("[best_full]", best_full.metrics, best_full.elapsed_sec)
    LOGGER.info(f"Best full trial dir: {best_full.exp_dir}")
    LOGGER.info(f"Best config written to: {args.write_best_config}")


if __name__ == "__main__":
    main()
