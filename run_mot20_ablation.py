#!/usr/bin/env python3
"""
ByteTrack MOT20 Ablation Test Script

Runs both baseline and improved configurations on MOT20 dataset.
This script automates the ablation study to compare:
- Baseline: entry_margin=0, zombie features disabled
- Improved: entry_margin=50, zombie features enabled
"""

import subprocess
import sys
import shutil
import time
from pathlib import Path

# Configuration
DETECTOR = "yolox_x_MOT20_ablation"
REID = "lmbn_n_duke"
TRACKER = "bytetrack"
SOURCE = "boxmot/engine/trackeval/MOT20-ablation/train"
PROJECT = "runs/ablation"

# Tracker config paths
BASELINE_CONFIG = "boxmot/configs/trackers/bytetrack_baseline_mot20.yaml"
IMPROVED_CONFIG = "boxmot/configs/trackers/bytetrack_improved_mot20.yaml"
ORIGINAL_CONFIG = "boxmot/configs/trackers/bytetrack.yaml"

def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.1f}s")
    return result.returncode == 0

def run_baseline_test():
    """Run baseline test with disabled features."""
    # Copy baseline config to bytetrack.yaml
    shutil.copy(BASELINE_CONFIG, ORIGINAL_CONFIG)

    cmd = [
        sys.executable, "-m", "boxmot.engine.cli", "eval",
        DETECTOR, REID, TRACKER,
        "--source", SOURCE,
        "--name", "bytetrack_baseline_mot20",
        "--save",
        "--verbose",
        "--project", PROJECT
    ]

    return run_command(cmd, "Baseline Test (features disabled)")

def run_improved_test():
    """Run improved test with enabled features."""
    # Copy improved config to bytetrack.yaml
    shutil.copy(IMPROVED_CONFIG, ORIGINAL_CONFIG)

    cmd = [
        sys.executable, "-m", "boxmot.engine.cli", "eval",
        DETECTOR, REID, TRACKER,
        "--source", SOURCE,
        "--name", "bytetrack_improved_mot20",
        "--save",
        "--verbose",
        "--project", PROJECT
    ]

    return run_command(cmd, "Improved Test (features enabled)")

def restore_config():
    """Restore original config from backup."""
    backup = "/tmp/bytetrack.yaml.bak"
    if Path(backup).exists():
        shutil.copy(backup, ORIGINAL_CONFIG)
        print(f"Restored original config from {backup}")

def main():
    """Main test runner."""
    print("ByteTrack MOT20 Ablation Test")
    print("=" * 60)
    print(f"Detector: {DETECTOR}")
    print(f"ReID: {REID}")
    print(f"Tracker: {TRACKER}")
    print(f"Dataset: MOT20-ablation")
    print("=" * 60)

    # Ensure original config is backed up
    if not Path("/tmp/bytetrack.yaml.bak").exists():
        shutil.copy(ORIGINAL_CONFIG, "/tmp/bytetrack.yaml.bak")

    try:
        # Run baseline test only (improved will be run manually after user's modifications)
        baseline_success = run_baseline_test()
        if not baseline_success:
            print("WARNING: Baseline test failed!")
        else:
            print("\n" + "=" * 60)
            print("BASELINE TEST COMPLETE!")
            print("=" * 60)
            print("\nNow you can make modifications before running improved test.")
            print("To run improved test later, execute: python run_mot20_improved.py")

    finally:
        # Restore config
        restore_config()

    print("\n" + "=" * 60)
    print("Baseline Test Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {PROJECT}/")
    print("\nExpected result locations:")
    print(f"  Baseline: {PROJECT}/mot/{DETECTOR}_{REID}_{TRACKER}_baseline_mot20*/")

if __name__ == "__main__":
    main()
