#!/usr/bin/env python3
"""
ByteTrack MOT17-04 Comparison Test

Tests Baseline vs Improved ByteTrack configurations on MOT17-04 sequence.
"""

import subprocess
import sys
from pathlib import Path
import re

# Tracker configuration file
TRACKER_YAML = Path("boxmot/configs/trackers/bytetrack.yaml")

# Baseline config (disable all new features)
BASELINE_CONFIG = {
    "entry_margin": 0,
    "zombie_iou_thresh": 0.3,
    "zombie_max_history": 0,
    "zombie_dist_thresh": 999999,
    "zombie_max_predict_frames": 0,
}

# Improved config (all new features enabled)
IMPROVED_CONFIG = {
    "entry_margin": 50,
    "zombie_iou_thresh": 0.3,
    "zombie_max_history": 100,
    "zombie_dist_thresh": 150,
    "zombie_max_predict_frames": 5,
}


def update_tracker_config(config: dict):
    """Update the tracker config file with new parameters."""
    content = TRACKER_YAML.read_text()

    for key, value in config.items():
        # Find and replace the default value for each parameter
        pattern = rf"({key}:\s*\n\s+type:.*\n\s+default:)\s*\S+"
        replacement = rf"\1 {value}"
        content = re.sub(pattern, replacement, content)

    TRACKER_YAML.write_text(content)
    print(f"Updated {TRACKER_YAML} with config: {config}")


def run_evaluation(name: str, source: str, project: str = "runs/ablation") -> dict:
    """Run the evaluation and return results."""
    cmd = [
        sys.executable, "-m", "boxmot.engine.cli",
        "eval",
        "yolox_x_MOT17_ablation",  # detector
        "lmbn_n_duke",             # ReID model
        "bytetrack",               # tracker
        "--source", source,
        "--save",
        "--name", name,
        "--verbose",
        "--project", project,
    ]

    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}


def extract_metrics(stdout: str) -> dict:
    """Extract HOTA, MOTA, IDF1, IDSW from stdout."""
    metrics = {}

    # Look for TrackEval output patterns
    lines = stdout.split('\n')
    for line in lines:
        # Try to find metrics in various formats
        if 'HOTA' in line and ':' in line:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    metrics['HOTA'] = float(parts[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if 'MOTA' in line and ':' in line:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    metrics['MOTA'] = float(parts[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if 'IDF1' in line and ':' in line:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    metrics['IDF1'] = float(parts[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if 'IDSW' in line and ':' in line:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    metrics['IDSW'] = float(parts[1].strip().split()[0])
            except (ValueError, IndexError):
                pass

    return metrics


def main():
    # Save original config
    original_config = TRACKER_YAML.read_text()

    # Test data path
    source = "/tmp/MOT17-04-only/train"
    project = "runs/ablation"

    results = {}

    try:
        # Step 1: Baseline test
        print("\n" + "="*80)
        print("STEP 1: BASELINE TEST (entry_margin=0, zombie features disabled)")
        print("="*80)
        update_tracker_config(BASELINE_CONFIG)
        results["baseline"] = run_evaluation(
            "bytetrack_baseline_mot17_04",
            source,
            project
        )

        # Step 2: Improved test
        print("\n" + "="*80)
        print("STEP 2: IMPROVED TEST (entry_margin=50, all features enabled)")
        print("="*80)
        update_tracker_config(IMPROVED_CONFIG)
        results["improved"] = run_evaluation(
            "bytetrack_improved_mot17_04",
            source,
            project
        )

    finally:
        # Restore original config
        TRACKER_YAML.write_text(original_config)
        print("\n" + "="*80)
        print("Restored original tracker config")
        print("="*80)

    # Summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    for test_name, result in results.items():
        status = "SUCCESS" if result["returncode"] == 0 else "FAILED"
        print(f"\n{test_name.upper()}: {status}")
        if result["returncode"] != 0:
            print(f"  Error: {result['stderr'][:200]}")

    print("\n" + "="*80)
    print("RESULTS LOCATION")
    print("="*80)
    print(f"Baseline results: {project}/bytetrack_baseline_mot17_04/")
    print(f"Improved results: {project}/bytetrack_improved_mot17_04/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
