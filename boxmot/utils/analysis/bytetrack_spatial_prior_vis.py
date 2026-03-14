from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from boxmot.trackers.tracker_zoo import create_tracker
from boxmot.utils import DATASET_CONFIGS, WEIGHTS, logger as LOGGER
from boxmot.utils.dataloaders.MOT17 import MOT17DetEmbDataset


def load_dataset_cfg(name: str) -> dict:
    path = DATASET_CONFIGS / f"{name}.yaml"
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def resolve_source(source: str) -> Path:
    cfg_path = DATASET_CONFIGS / f"{source}.yaml"
    if cfg_path.exists():
        cfg = load_dataset_cfg(source)
        benchmark = cfg.get("benchmark", {})
        return (Path(benchmark["source"]) / benchmark["split"]).resolve()
    return Path(source).resolve()


def parse_sequence_filter(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    items = {part.strip() for part in raw.replace(",", " ").split() if part.strip()}
    return items or None


def colorize_map(values: np.ndarray, size: tuple[int, int], cmap: int) -> np.ndarray:
    resized = cv2.resize(values.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    finite = resized[np.isfinite(resized)]
    positive = finite[finite > 0]
    if positive.size == 0:
        scaled = np.zeros_like(resized, dtype=np.uint8)
    else:
        hi = max(float(np.percentile(positive, 99.0)), 1e-6)
        scaled = np.clip((resized / hi) * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.applyColorMap(scaled, cmap)


def resize_mask(mask: np.ndarray | None, size: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    return cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)


def overlay_regions(
    image: np.ndarray,
    entry_mask: np.ndarray | None,
    core_mask: np.ndarray | None,
) -> np.ndarray:
    h, w = image.shape[:2]
    entry = resize_mask(entry_mask, (w, h)).astype(bool)
    core = resize_mask(core_mask, (w, h)).astype(bool)
    overlay = image.copy()
    overlay[core] = (overlay[core] * 0.65 + np.array([255, 180, 0]) * 0.35).astype(np.uint8)
    overlay[entry] = (overlay[entry] * 0.45 + np.array([0, 220, 80]) * 0.55).astype(np.uint8)
    return overlay


def make_region_map(
    entry_mask: np.ndarray | None,
    core_mask: np.ndarray | None,
    size: tuple[int, int],
) -> np.ndarray:
    w, h = size
    entry = resize_mask(entry_mask, size).astype(bool)
    core = resize_mask(core_mask, size).astype(bool)
    canvas = np.full((h, w, 3), 24, dtype=np.uint8)
    canvas[core] = (255, 180, 0)
    canvas[entry] = (0, 220, 80)
    return canvas


def annotate(image: np.ndarray, lines: list[str]) -> np.ndarray:
    out = image.copy()
    y = 30
    for line in lines:
        cv2.putText(
            out,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30
    return out


def label_panel(image: np.ndarray, title: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 48), (18, 18, 18), -1)
    cv2.putText(
        out,
        title,
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def build_panel(seq_name: str, tracker, representative_image: np.ndarray) -> tuple[np.ndarray, dict]:
    prob_maps = tracker.spatial_prior.get_probability_maps()
    metric_maps = tracker.spatial_prior.get_metric_maps()
    h, w = representative_image.shape[:2]
    size = (w, h)

    overlay = overlay_regions(representative_image, tracker.spatial_entry_mask, tracker.spatial_core_mask)
    region_map = make_region_map(tracker.spatial_entry_mask, tracker.spatial_core_mask, size)
    walkable = colorize_map(prob_maps["walkable"], size, cv2.COLORMAP_TURBO)
    birth_density = colorize_map(metric_maps["birth_density"], size, cv2.COLORMAP_VIRIDIS)

    stats = tracker.spatial_prior.stats()
    info_lines = [
        f"{seq_name}",
        f"stage={tracker.spatial_prior_stage}",
        f"support_samples={tracker.spatial_prior_support_samples}",
        f"birth_events={tracker.spatial_prior_birth_events}",
        f"support_total={stats.support_total:.2f}",
        f"birth_total={stats.birth_total:.2f}",
    ]
    overlay = annotate(overlay, info_lines)

    top = np.hstack([label_panel(overlay, "Frame Overlay"), label_panel(walkable, "Walkable Support")])
    bottom = np.hstack([label_panel(birth_density, "Birth Density"), label_panel(region_map, "Entry/Core Regions")])
    panel = np.vstack([top, bottom])

    summary = {
        "sequence": seq_name,
        "stage": tracker.spatial_prior_stage,
        "support_samples": int(tracker.spatial_prior_support_samples),
        "birth_events": int(tracker.spatial_prior_birth_events),
        "support_total": float(stats.support_total),
        "birth_total": float(stats.birth_total),
        "birth_density_max": float(metric_maps["birth_density"].max()),
        "entry_cells": int(np.count_nonzero(tracker.spatial_entry_mask)) if tracker.spatial_entry_mask is not None else 0,
        "core_cells": int(np.count_nonzero(tracker.spatial_core_mask)) if tracker.spatial_core_mask is not None else 0,
        "grid_shape": list(prob_maps["walkable"].shape),
    }
    return panel, summary


def run_sequence(
    seq_name: str,
    dataset: MOT17DetEmbDataset,
    tracker_config: Path,
    device: str,
    output_dir: Path,
) -> dict:
    tracker = create_tracker(
        tracker_type="bytetrack",
        tracker_config=tracker_config,
        reid_weights=WEIGHTS / "lmbn_n_duke.pt",
        device=device,
        half=False,
        per_class=False,
    )
    sequence = dataset.get_sequence(seq_name)
    for frame in sequence:
        tracker.update(frame["dets"], frame["img"], None)

    if not getattr(tracker, "spatial_prior", None):
        raise RuntimeError("Tracker config does not enable spatial_prior.")

    tracker._update_spatial_prior_stage()
    tracker._refresh_spatial_region_masks()

    frame_paths = dataset.seqs[seq_name]["frame_paths"]
    representative_path = frame_paths[len(frame_paths) // 2]
    representative_image = cv2.imread(str(representative_path), cv2.IMREAD_COLOR)
    if representative_image is None:
        raise RuntimeError(f"Failed to read representative frame: {representative_path}")

    panel, summary = build_panel(seq_name, tracker, representative_image)

    seq_dir = output_dir / seq_name
    seq_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(seq_dir / "entry_prior_panel.png"), panel)
    cv2.imwrite(
        str(seq_dir / "entry_prior_overlay.png"),
        overlay_regions(representative_image, tracker.spatial_entry_mask, tracker.spatial_core_mask),
    )
    with open(seq_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ByteTrack spatial-prior entry regions from cached detections.")
    parser.add_argument("--source", default="SOMPT22-full", help="Dataset config name or dataset path.")
    parser.add_argument("--project", type=Path, default=Path("runs/sompt22_compare_v8m_full_final"))
    parser.add_argument("--model-name", default="yolov8m_pretrain_crowdhuman")
    parser.add_argument("--reid-name", default="lmbn_n_duke")
    parser.add_argument("--tracker-config", type=Path, default=Path("boxmot/configs/trackers/bytetrack_improved.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sequences", default=None, help="Comma or space separated sequence names to process.")
    args = parser.parse_args()

    source_root = resolve_source(args.source)
    output_dir = args.output_dir or args.project / "spatial_prior_vis_entry_only"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MOT17DetEmbDataset(
        mot_root=str(source_root),
        det_emb_root=str(args.project / "dets_n_embs"),
        model_name=args.model_name,
        reid_name=args.reid_name,
        target_fps=None,
    )
    selected = parse_sequence_filter(args.sequences)
    summaries = []
    for seq_name in dataset.sequence_names():
        if selected is not None and seq_name not in selected:
            continue
        LOGGER.info(f"Visualizing spatial prior for {seq_name}")
        summaries.append(
            run_sequence(
                seq_name=seq_name,
                dataset=dataset,
                tracker_config=args.tracker_config,
                device=args.device,
                output_dir=output_dir,
            )
        )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    LOGGER.info(f"Saved spatial-prior visualizations to {output_dir}")


if __name__ == "__main__":
    main()
