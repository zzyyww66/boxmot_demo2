#!/usr/bin/env python3
"""Run only the improved ByteTrack test."""

import subprocess
import sys
from pathlib import Path
import re
import shutil

TRACKER_YAML = Path("boxmot/configs/trackers/bytetrack.yaml")
PROJECT_DIR = Path("runs/ablation")
SEQ_SRC = Path("boxmot/engine/trackeval/MOT17-ablation/train/MOT17-04")
SUBSET_ROOT = PROJECT_DIR / "_mot17_04_subset"

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
    """Update the tracker config file with new parameters."""
    content = TRACKER_YAML.read_text()

    for key, value in config.items():
        pattern = rf"^({key}:\s*\n\s+type:\s*\w+\s*\n\s+default:)\s*\S+"
        replacement = rf"\1 {value}"
        content_new = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        if content_new == content:
            pattern = rf"^({key}:\s*\n\s+type:\s*\w+\s*\n\s+default:)\s*[^\n]*"
            replacement = rf"\1 {value}"
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        else:
            content = content_new

    TRACKER_YAML.write_text(content)
    print(f"Updated {TRACKER_YAML} with config: {config}")

def prepare_mot17_04_source() -> Path:
    if not SEQ_SRC.exists():
        raise FileNotFoundError(f"Sequence not found: {SEQ_SRC}")
    if SUBSET_ROOT.exists():
        shutil.rmtree(SUBSET_ROOT)
    SUBSET_ROOT.mkdir(parents=True, exist_ok=True)
    dst = SUBSET_ROOT / "MOT17-04"
    try:
        dst.symlink_to(SEQ_SRC.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(SEQ_SRC, dst)
    return SUBSET_ROOT

def main():
    original_config = TRACKER_YAML.read_text()
    source = prepare_mot17_04_source()

    try:
        print("="*80)
        print("STEP 2: IMPROVED TEST (all new features enabled)")
        print("="*80)
        update_tracker_config(IMPROVED_CONFIG)

        cmd = [
            sys.executable, "-m", "boxmot.engine.cli",
            "eval",
            "yolox_x_MOT17_ablation",
            "lmbn_n_duke",
            "bytetrack",
            "--source", str(source),
            "--save",
            "--name", "bytetrack_improved_mot17_04",
            "--verbose",
            "--project", str(PROJECT_DIR),
        ]

        print(f"\nRunning: {' '.join(cmd)}\n")
        result = subprocess.run(cmd)
        return result.returncode

    finally:
        TRACKER_YAML.write_text(original_config)
        print("\nRestored original tracker config")

if __name__ == "__main__":
    sys.exit(main())
