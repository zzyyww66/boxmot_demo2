#!/usr/bin/env python3
"""Launch live MOT17-04 tracking with switchable ByteTrack configs."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from shutil import which


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the MOT17-04 live runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Run real-time detection + ByteTrack association on MOT17-04 "
            "with a display window and person IDs."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("original", "improved"),
        default="improved",
        help="ByteTrack config preset.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Input source (MOT17-04 image folder/video path).",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=Path("yolox_x_MOT17_ablation.pt"),
        help="Detector weights path.",
    )
    parser.add_argument(
        "--reid-model",
        type=Path,
        default=Path("lmbn_n_duke.pt"),
        help="ReID weights path.",
    )
    parser.add_argument(
        "--imgsz",
        type=str,
        default="1088,1920",
        help="Inference image size passed to BoxMOT CLI.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.01,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Compute device, e.g. '0', '0,1', or 'cpu'.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/live_mot17_04"),
        help="Output project directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name under project. Defaults to mode value.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save tracked video.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Enable FP16 inference (GPU only).",
    )
    parser.add_argument(
        "--vid-stride",
        type=int,
        default=1,
        help="Process every Nth frame (higher is faster, less smooth).",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save tracking TXT results.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=None,
        help="Optional bounding-box line width.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without executing it.",
    )
    return parser


def resolve_path(root: Path, path_value: Path) -> Path:
    """Resolve path against repository root if relative."""
    if path_value.is_absolute():
        # Common typo: user passes '/.venv/...' instead of '<repo>/.venv/...'
        if not path_value.exists() and str(path_value).startswith("/.venv/"):
            candidate = root / str(path_value).lstrip("/")
            if candidate.exists():
                return candidate
        return path_value
    return root / path_value


def default_source_path(repo_root: Path) -> Path:
    """Prefer full MOT17-04 if present, otherwise fallback to MOT17-mini."""
    full = repo_root / ".venv/lib/python3.11/site-packages/trackeval/MOT17/train/MOT17-04/img1"
    mini = repo_root / "assets/MOT17-mini/train/MOT17-04-FRCNN/img1"
    return full if full.exists() else mini


def normalize_source_path(source: Path) -> Path:
    """If a MOT sequence root is provided, use its img1 folder automatically."""
    if source.is_dir():
        img1 = source / "img1"
        if img1.exists() and img1.is_dir():
            return img1
    return source


def count_frames(source: Path) -> int:
    """Best-effort frame count for image-directory sources."""
    if not source.is_dir():
        return -1
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in source.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def ensure_path_exists(label: str, path_value: Path) -> None:
    """Exit with a clear message if a required path is missing."""
    if not path_value.exists():
        raise FileNotFoundError(f"{label} not found: {path_value}")


def build_track_command(args: argparse.Namespace, repo_root: Path) -> list[str]:
    """Build underlying BoxMOT tracking command."""
    tracker_cfg_map = {
        "original": Path("boxmot/configs/trackers/bytetrack_original.yaml"),
        "improved": Path("boxmot/configs/trackers/bytetrack_improved.yaml"),
    }

    source_arg = args.source if args.source is not None else default_source_path(repo_root)
    source = resolve_path(repo_root, source_arg)
    source = normalize_source_path(source)
    yolo_model = resolve_path(repo_root, args.yolo_model)
    reid_model = resolve_path(repo_root, args.reid_model)
    tracker_cfg = resolve_path(repo_root, tracker_cfg_map[args.mode])
    project = resolve_path(repo_root, args.project)
    run_name = args.name or args.mode

    ensure_path_exists("Source", source)
    ensure_path_exists("YOLO model", yolo_model)
    ensure_path_exists("ReID model", reid_model)
    ensure_path_exists("Tracker config", tracker_cfg)

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "boxmot.engine.cli",
        "track",
        "--source",
        str(source),
        "--yolo-model",
        str(yolo_model),
        "--reid-model",
        str(reid_model),
        "--tracking-method",
        "bytetrack",
        "--tracker-config",
        str(tracker_cfg),
        "--classes",
        "0",
        "--imgsz",
        args.imgsz,
        "--conf",
        str(args.conf),
        "--fps",
        "30",
        "--project",
        str(project),
        "--name",
        run_name,
        "--exist-ok",
        "--show",
        "--show-labels",
        "--hide-conf",
        "--hide-class",
    ]

    cmd.extend(["--device", args.device])
    if args.save:
        cmd.append("--save")
    if args.save_txt:
        cmd.append("--save-txt")
    if args.half:
        cmd.append("--half")
    if args.vid_stride > 1:
        cmd.extend(["--vid-stride", str(args.vid_stride)])
    if args.line_width is not None:
        cmd.extend(["--line-width", str(args.line_width)])
    return cmd


def main() -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if which("uv") is None:
        print("Error: 'uv' is not installed or not found in PATH.", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parent.parent
    cmd = build_track_command(args, repo_root)
    source_arg = args.source if args.source is not None else default_source_path(repo_root)
    source_resolved = normalize_source_path(resolve_path(repo_root, source_arg))
    n_frames = count_frames(source_resolved)
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    print(f"[run_mot17_04_live] Mode: {args.mode}")
    print(f"[run_mot17_04_live] Source: {source_resolved}")
    print(f"[run_mot17_04_live] Device: {args.device}")
    if n_frames > 0:
        print(f"[run_mot17_04_live] Frames: {n_frames}")
        if n_frames <= 20:
            print(
                "[run_mot17_04_live] Note: source has very few frames, so the window will close quickly."
            )
    print(f"[run_mot17_04_live] Command: {cmd_str}")

    if args.dry_run:
        return 0

    try:
        return subprocess.run(cmd, cwd=repo_root).returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
