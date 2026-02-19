#!/usr/bin/env python3
"""Run improved ByteTrack test with conservative config."""

import subprocess
import sys
from pathlib import Path
import re

TRACKER_YAML = Path("boxmot/configs/trackers/bytetrack.yaml")

# Conservative config - disable entry zone, larger thresholds
IMPROVED_CONFIG_V2 = {
    "entry_margin": 0,              # Disable entry zone restriction
    "zombie_max_history": 100,
    "zombie_dist_thresh": 250,      # Larger distance threshold
    "zombie_max_predict_frames": 5,
    "zombie_transition_frames": 15, # Shorter transition
    "zombie_match_max_dist": 300,   # Larger max dist for zombie matching
    "exit_zone_enabled": True,
    "exit_zone_margin": 50,
    "exit_zone_remove_grace": 30,
    "adaptive_zone_enabled": False, # Disable adaptive zone
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

def main():
    original_config = TRACKER_YAML.read_text()

    try:
        print("="*80)
        print("STEP 2: IMPROVED TEST V2 (conservative config)")
        print("="*80)
        update_tracker_config(IMPROVED_CONFIG_V2)

        cmd = [
            sys.executable, "-m", "boxmot.engine.cli",
            "eval",
            "yolox_x_MOT17_ablation",
            "lmbn_n_duke",
            "bytetrack",
            "--source", "MOT17-ablation",
            "--save",
            "--name", "bytetrack_improved_v2_mot17_04",
            "--verbose",
            "--project", "runs/ablation",
        ]

        print(f"\nRunning: {' '.join(cmd)}\n")
        result = subprocess.run(cmd)
        return result.returncode

    finally:
        TRACKER_YAML.write_text(original_config)
        print("\nRestored original tracker config")

if __name__ == "__main__":
    sys.exit(main())
