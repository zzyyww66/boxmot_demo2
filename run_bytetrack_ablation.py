#!/usr/bin/env python3
"""
ByteTrack Lifecycle Gating Ablation Test

This script runs ByteTrack with two configurations:
1. Baseline: entry_margin=0 (original ByteTrack behavior)
2. Lifecycle Gating: entry_margin=50 (with birth control and zombie rescue)

Results are compared for IDSW, IDF1, MOTA metrics.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Configuration for the two test variants
BASELINE_CONFIG = {
    "entry_margin": 0,
}

GATING_CONFIG = {
    "entry_margin": 50,
}


def update_tracker_config(tracker_yaml: Path, config: dict):
    """Update the tracker config file with new parameters."""
    content = tracker_yaml.read_text()

    # Update entry_margin default
    for key, value in config.items():
        # Find and replace the default value
        import re
        pattern = rf"({key}:\s*\n\s+type:.*\n\s+default:)\s*\S+"
        replacement = rf"\1 {value}"
        content = re.sub(pattern, replacement, content)

    tracker_yaml.write_text(content)
    print(f"Updated {tracker_yaml} with {config}")


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


def main():
    parser = argparse.ArgumentParser(description="ByteTrack Lifecycle Gating Ablation Test")
    parser.add_argument("--source", default="MOT17-mini",
                       help="Dataset source to use (default: MOT17-mini)")
    parser.add_argument("--project", default="runs/ablation",
                       help="Output project directory")
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline test (run gating only)")
    parser.add_argument("--skip-gating", action="store_true",
                       help="Skip gating test (run baseline only)")
    args = parser.parse_args()

    tracker_yaml = Path("boxmot/configs/trackers/bytetrack.yaml")

    # Save original config
    original_config = tracker_yaml.read_text()

    results = {}

    try:
        # Test 1: Baseline (entry_margin=0)
        if not args.skip_baseline:
            print("\n" + "="*80)
            print("TEST 1: Baseline (entry_margin=0)")
            print("="*80)
            update_tracker_config(tracker_yaml, BASELINE_CONFIG)
            results["baseline"] = run_evaluation(
                "bytetrack_baseline",
                args.source,
                args.project
            )

        # Test 2: Lifecycle Gating (entry_margin=50)
        if not args.skip_gating:
            print("\n" + "="*80)
            print("TEST 2: Lifecycle Gating (entry_margin=50)")
            print("="*80)
            update_tracker_config(tracker_yaml, GATING_CONFIG)
            results["gating"] = run_evaluation(
                "bytetrack_gating",
                args.source,
                args.project
            )

    finally:
        # Restore original config
        tracker_yaml.write_text(original_config)
        print("\n" + "="*80)
        print("Restored original tracker config")
        print("="*80)

    # Summary
    print("\n" + "="*80)
    print("ABLATION TEST SUMMARY")
    print("="*80)
    for test_name, result in results.items():
        status = "SUCCESS" if result["returncode"] == 0 else "FAILED"
        print(f"\n{test_name.upper()}: {status}")
        if result["returncode"] != 0:
            print(f"  Error: {result['stderr'][:200]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
