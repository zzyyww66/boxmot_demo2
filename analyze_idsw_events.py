#!/usr/bin/env python3
"""Analyze per-frame ID switch (IDSW) events for MOTChallenge-style results.

This script reproduces TrackEval CLEAR matching logic for a single sequence and
exports all IDSW events with frame-level details to CSV.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from boxmot.engine.trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox


@dataclass
class Event:
    """Single ID switch event detail."""

    frame: int
    gt_id: int
    prev_tracker_id: int
    new_tracker_id: int
    iou: float
    tracker_conf: float
    last_match_frame: int
    gap_frames: int
    gt_tlwh: tuple[float, float, float, float]
    trk_tlwh: tuple[float, float, float, float]


@dataclass
class FrameData:
    """Preprocessed frame data used by CLEAR matching."""

    gt_ids: np.ndarray
    gt_dets: np.ndarray
    tracker_ids: np.ndarray
    tracker_dets: np.ndarray
    tracker_confidences: np.ndarray
    similarity_scores: np.ndarray


def _build_dataset(run_dir: Path, gt_root: Path, seq: str) -> MotChallenge2DBox:
    """Create a TrackEval MOT dataset wrapper for one run/sequence."""
    config = {
        "GT_FOLDER": str(gt_root),
        "TRACKERS_FOLDER": str(run_dir.parent),
        "TRACKERS_TO_EVAL": [run_dir.name],
        "TRACKER_SUB_FOLDER": "",
        "OUTPUT_FOLDER": None,
        "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": "MOT17",
        "SPLIT_TO_EVAL": "train",
        "SEQ_INFO": {seq: None},
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
        "SKIP_SPLIT_FOL": True,
        "PRINT_CONFIG": False,
    }
    return MotChallenge2DBox(config)


def _distractor_classes(dataset: MotChallenge2DBox) -> list[int]:
    """Return distractor class IDs using TrackEval's MOT policy."""
    distractor_names = ["person_on_vehicle", "static_person", "distractor", "reflection"]
    if dataset.benchmark == "MOT20":
        distractor_names.append("non_mot_vehicle")
    return [dataset.class_name_to_class_id[name] for name in distractor_names]


def _preprocess_frames_mot(raw_data: dict, dataset: MotChallenge2DBox) -> list[FrameData]:
    """Apply MOT preprocessing equivalent to TrackEval without ID remapping."""
    distractor_classes = _distractor_classes(dataset)
    cls_id = dataset.class_name_to_class_id["pedestrian"]

    frames: list[FrameData] = []
    for t in range(raw_data["num_timesteps"]):
        gt_ids = raw_data["gt_ids"][t]
        gt_dets = raw_data["gt_dets"][t]
        gt_classes = raw_data["gt_classes"][t]
        gt_zero_marked = raw_data["gt_extras"][t]["zero_marked"]

        tracker_ids = raw_data["tracker_ids"][t]
        tracker_dets = raw_data["tracker_dets"][t]
        tracker_classes = raw_data["tracker_classes"][t]
        tracker_confidences = raw_data["tracker_confidences"][t]
        similarity_scores = raw_data["similarity_scores"][t]

        # TrackEval constraint for MOT pedestrian evaluation.
        if len(tracker_classes) > 0 and np.max(tracker_classes) > 1:
            raise ValueError(
                f"Found non-pedestrian tracker classes in frame {t + 1}: "
                f"max class id={int(np.max(tracker_classes))}"
            )

        # Remove tracker dets matched to distractor GT classes.
        to_remove_tracker = np.array([], dtype=int)
        if (
            dataset.do_preproc
            and dataset.benchmark != "MOT15"
            and gt_ids.shape[0] > 0
            and tracker_ids.shape[0] > 0
        ):
            matching_scores = similarity_scores.copy()
            matching_scores[matching_scores < 0.5 - np.finfo("float").eps] = 0
            match_rows, match_cols = linear_sum_assignment(-matching_scores)
            matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo("float").eps
            match_rows = match_rows[matched_mask]
            match_cols = match_cols[matched_mask]
            is_distractor = np.isin(gt_classes[match_rows], distractor_classes)
            to_remove_tracker = match_cols[is_distractor]

        tracker_ids = np.delete(tracker_ids, to_remove_tracker, axis=0)
        tracker_dets = np.delete(tracker_dets, to_remove_tracker, axis=0)
        tracker_confidences = np.delete(tracker_confidences, to_remove_tracker, axis=0)
        similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

        if dataset.do_preproc and dataset.benchmark != "MOT15":
            gt_keep = (gt_zero_marked != 0) & (gt_classes == cls_id)
        else:
            gt_keep = gt_zero_marked != 0

        gt_ids = gt_ids[gt_keep]
        gt_dets = gt_dets[gt_keep, :]
        similarity_scores = similarity_scores[gt_keep]

        frames.append(
            FrameData(
                gt_ids=gt_ids,
                gt_dets=gt_dets,
                tracker_ids=tracker_ids,
                tracker_dets=tracker_dets,
                tracker_confidences=tracker_confidences,
                similarity_scores=similarity_scores,
            )
        )
    return frames


def _collect_idsw_events(frames: list[FrameData], threshold: float) -> list[Event]:
    """Reproduce CLEAR matching and collect all IDSW events."""
    all_gt_ids = sorted({int(gt_id) for f in frames for gt_id in f.gt_ids.tolist()})
    if not all_gt_ids:
        return []

    gt_to_idx = {gt_id: i for i, gt_id in enumerate(all_gt_ids)}

    prev_tracker_id = np.nan * np.zeros(len(all_gt_ids))
    prev_timestep_tracker_id = np.nan * np.zeros(len(all_gt_ids))
    last_match_frame = np.nan * np.zeros(len(all_gt_ids))

    events: list[Event] = []

    for frame_idx, frame in enumerate(frames, start=1):
        gt_ids_t = frame.gt_ids
        tracker_ids_t = frame.tracker_ids

        if len(gt_ids_t) == 0:
            continue
        if len(tracker_ids_t) == 0:
            continue

        gt_idxs_t = np.asarray([gt_to_idx[int(gt_id)] for gt_id in gt_ids_t], dtype=int)
        similarity = frame.similarity_scores

        score_mat = tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_idxs_t[:, np.newaxis]]
        score_mat = 1000 * score_mat + similarity
        score_mat[similarity < threshold - np.finfo("float").eps] = 0

        match_rows, match_cols = linear_sum_assignment(-score_mat)
        matched_mask = score_mat[match_rows, match_cols] > 0 + np.finfo("float").eps
        match_rows = match_rows[matched_mask]
        match_cols = match_cols[matched_mask]

        matched_gt_idxs = gt_idxs_t[match_rows]
        matched_gt_ids = gt_ids_t[match_rows]
        matched_tracker_ids = tracker_ids_t[match_cols]

        prev_matched_tracker_ids = prev_tracker_id[matched_gt_idxs]
        is_idsw = (~np.isnan(prev_matched_tracker_ids)) & (matched_tracker_ids != prev_matched_tracker_ids)

        switched_local_idx = np.where(is_idsw)[0]
        for local_i in switched_local_idx:
            row_i = int(match_rows[local_i])
            col_i = int(match_cols[local_i])
            gt_idx = int(matched_gt_idxs[local_i])

            prev_tid = int(prev_matched_tracker_ids[local_i])
            new_tid = int(matched_tracker_ids[local_i])
            gt_id = int(matched_gt_ids[local_i])
            iou = float(similarity[row_i, col_i])
            conf = float(frame.tracker_confidences[col_i]) if len(frame.tracker_confidences) > col_i else np.nan

            last_f = int(last_match_frame[gt_idx]) if not np.isnan(last_match_frame[gt_idx]) else -1
            gap = frame_idx - last_f if last_f >= 0 else -1

            gt_box = tuple(float(v) for v in frame.gt_dets[row_i].tolist())
            trk_box = tuple(float(v) for v in frame.tracker_dets[col_i].tolist())
            events.append(
                Event(
                    frame=frame_idx,
                    gt_id=gt_id,
                    prev_tracker_id=prev_tid,
                    new_tracker_id=new_tid,
                    iou=iou,
                    tracker_conf=conf,
                    last_match_frame=last_f,
                    gap_frames=gap,
                    gt_tlwh=gt_box,
                    trk_tlwh=trk_box,
                )
            )

        prev_tracker_id[matched_gt_idxs] = matched_tracker_ids
        prev_timestep_tracker_id[:] = np.nan
        prev_timestep_tracker_id[matched_gt_idxs] = matched_tracker_ids
        last_match_frame[matched_gt_idxs] = frame_idx

    return events


def _write_events_csv(events: Iterable[Event], output_csv: Path) -> None:
    """Write event list to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "gt_id",
                "prev_tracker_id",
                "new_tracker_id",
                "iou",
                "tracker_conf",
                "last_match_frame",
                "gap_frames",
                "gt_x",
                "gt_y",
                "gt_w",
                "gt_h",
                "trk_x",
                "trk_y",
                "trk_w",
                "trk_h",
            ]
        )
        for e in events:
            writer.writerow(
                [
                    e.frame,
                    e.gt_id,
                    e.prev_tracker_id,
                    e.new_tracker_id,
                    f"{e.iou:.6f}",
                    f"{e.tracker_conf:.6f}",
                    e.last_match_frame,
                    e.gap_frames,
                    *(f"{v:.3f}" for v in e.gt_tlwh),
                    *(f"{v:.3f}" for v in e.trk_tlwh),
                ]
            )


def _print_summary(events: list[Event]) -> None:
    """Print concise switch summary for quick diagnosis."""
    print(f"IDSW events: {len(events)}")
    if not events:
        return

    by_gt = Counter(e.gt_id for e in events)
    by_pair = Counter((e.prev_tracker_id, e.new_tracker_id) for e in events)
    by_frame = Counter(e.frame for e in events)

    print("\nTop GT IDs by switch count:")
    for gt_id, cnt in by_gt.most_common(10):
        print(f"  GT {gt_id}: {cnt}")

    print("\nTop tracker-id transitions:")
    for (old, new), cnt in by_pair.most_common(10):
        print(f"  {old} -> {new}: {cnt}")

    print("\nFrames with switches:")
    for frame, cnt in sorted(by_frame.items()):
        print(f"  frame {frame}: {cnt}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Export MOT17 IDSW events for one run.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory that contains <seq>.txt and pedestrian_detailed.csv",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="MOT17-04",
        help="Sequence name (default: MOT17-04)",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("boxmot/engine/trackeval/MOT17-ablation/train"),
        help="GT root containing <seq>/gt/gt.txt",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="CLEAR match threshold (default: 0.5)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <run-dir>/<seq>_idsw_events.csv)",
    )
    return parser.parse_args()


def main() -> int:
    """Run analysis and write event CSV."""
    args = parse_args()
    run_dir: Path = args.run_dir
    seq: str = args.seq
    gt_root: Path = args.gt_root
    output_csv = args.output_csv or (run_dir / f"{seq}_idsw_events.csv")

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not (run_dir / f"{seq}.txt").exists():
        raise FileNotFoundError(f"Tracker result not found: {run_dir / f'{seq}.txt'}")
    if not (gt_root / seq / "gt" / "gt.txt").exists():
        raise FileNotFoundError(f"GT file not found: {gt_root / seq / 'gt' / 'gt.txt'}")

    dataset = _build_dataset(run_dir=run_dir, gt_root=gt_root, seq=seq)
    raw_data = dataset.get_raw_seq_data(run_dir.name, seq)
    frames = _preprocess_frames_mot(raw_data=raw_data, dataset=dataset)
    events = _collect_idsw_events(frames=frames, threshold=args.threshold)

    _write_events_csv(events, output_csv)
    _print_summary(events)
    print(f"\nEvent CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
