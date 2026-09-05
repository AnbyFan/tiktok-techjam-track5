#!/usr/bin/env python3
"""
Download the final.jpg (HDR+ processed output) from the Google HDR+ burst
dataset subset, renaming each to a unique <burst_id>.jpg so they don't collide.

Uses gsutil via subprocess, parallelized with concurrent.futures.
"""

import subprocess
import concurrent.futures
from pathlib import Path

GSUTIL = r"C:\Users\wongq\AppData\Roaming\Python\Python314\Scripts\gsutil.exe"
SRC_PREFIX = "gs://hdrplusdata/20171106_subset/results_20171023"
DEST_DIR = Path(__file__).resolve().parent.parent / "data" / "hdrplus_real"
MAX_WORKERS = 8


def list_final_jpgs():
    """Return list of (burst_id, gsutil_url) for every final.jpg in the subset."""
    out = subprocess.check_output(
        [GSUTIL, "ls", f"{SRC_PREFIX}/*/final.jpg"], text=True
    )
    results = []
    for line in out.splitlines():
        line = line.strip()
        if not line.endswith("final.jpg"):
            continue
        # e.g. gs://.../results_20171023/<burst_id>/final.jpg
        burst_id = line.split("/")[-2]
        results.append((burst_id, line))
    return results


def download_one(item):
    burst_id, url = item
    dest = DEST_DIR / f"{burst_id}.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        return (burst_id, "skipped")
    try:
        subprocess.run(
            [GSUTIL, "cp", url, str(dest)],
            check=True, capture_output=True, text=True,
        )
        return (burst_id, "ok")
    except subprocess.CalledProcessError as e:
        return (burst_id, f"error: {e.stderr.strip()[:120]}")


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print("Listing final.jpg files...")
    items = list_final_jpgs()
    print(f"Found {len(items)} files. Downloading...")

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for burst_id, status in ex.map(download_one, items):
            done += 1
            if status != "ok" or done % 20 == 0:
                print(f"[{done}/{len(items)}] {burst_id}: {status}")

    print(f"\nDone. {len(list(DEST_DIR.glob('*.jpg')))} files in {DEST_DIR}")


if __name__ == "__main__":
    main()
