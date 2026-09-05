#!/usr/bin/env python3
"""
Orchestrator: wait for the feature cache to be complete, then run training.

This lets us launch one background task that handles the whole pipeline
(extraction -> training) and reports when done, instead of polling.

Usage:
    python train/run_pipeline.py
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CACHE_FILE = Path("data/features_cache.npz")
EXPECTED_IMAGES = 13996  # COCO 5000 + phone 153 + AI 8843
POLL_INTERVAL = 30  # seconds


def cache_count():
    if not CACHE_FILE.exists():
        return 0
    try:
        data = np.load(CACHE_FILE, allow_pickle=True, mmap_mode="r")
        return len(data["paths"])
    except Exception:
        # Cache may be mid-write; treat as incomplete
        return 0


def main():
    print("Waiting for feature cache to complete...")
    while True:
        n = cache_count()
        print(f"  Cache: {n}/{EXPECTED_IMAGES}", flush=True)
        if n >= EXPECTED_IMAGES:
            print("Cache complete. Starting training...")
            break
        time.sleep(POLL_INTERVAL)

    # Run the training script
    result = subprocess.run(
        [sys.executable, "train/train_attention_pooling.py"],
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    print(f"\nTraining exited with code {result.returncode}")


if __name__ == "__main__":
    main()
