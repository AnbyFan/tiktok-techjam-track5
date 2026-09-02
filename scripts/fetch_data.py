#!/usr/bin/env python3
"""
Fetch and organize image folders for eval_robustness.py.

Produces:
    data/val/real/        COCO val2017 images (no auth needed, ~1 GB)
    data/cifake/real/     CIFAKE test REAL     (needs Kaggle creds, --cifake)
    data/cifake/ai/       CIFAKE test FAKE     (needs Kaggle creds, --cifake)

For the AI side with zero downloads, just copy a few hundred of your own
ComfyUI generations into data/val/ai/ manually.

Usage:
    python fetch_data.py                 # COCO val2017 -> data/val/real
    python fetch_data.py --cifake        # also fetch CIFAKE (kaggle.json)
    python fetch_data.py --max-images 500
"""

import argparse
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--max-images", type=int, default=None,
                   help="cap images copied per folder (default: all)")
    p.add_argument("--cifake", action="store_true",
                   help="also fetch CIFAKE via kagglehub (needs ~/.kaggle/kaggle.json)")
    return p.parse_args()


def progress(count, block, total):
    pct = count * block * 100 // max(total, 1)
    print(f"\r    {pct:3d}%", end="", flush=True)


def copy_cap(src_dir, dst_dir, exts, limit):
    dst_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(src_dir).rglob("*") if p.suffix.lower() in exts)
    if limit:
        paths = paths[:limit]
    for p in paths:
        shutil.copy2(p, dst_dir / p.name)
    return len(paths)


def fetch_coco(root, limit):
    zip_path = root / "val2017.zip"
    extract_dir = root / "coco"
    if not (extract_dir / "val2017").exists():
        if not zip_path.exists():
            print(f"[coco] downloading {COCO_URL} (~1 GB)")
            urlretrieve(COCO_URL, zip_path, progress)
            print()
        print("[coco] extracting ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
    n = copy_cap(extract_dir / "val2017", root / "val" / "real", {".jpg"}, limit)
    print(f"[coco] {n} real images -> {root/'val'/'real'}")


def fetch_cifake(root, limit):
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("pip install kagglehub  (and set up ~/.kaggle/kaggle.json)")
    print("[cifake] downloading via kagglehub ...")
    base = Path(kagglehub.dataset_download(
        "birdy654/cifake-real-and-ai-generated-synthetic-images"))
    # CIFAKE layout: {train,test}/{REAL,FAKE}
    test = base / "test"
    if not test.exists():
        candidates = [d for d in base.rglob("*") if d.is_dir()
                      and (d / "REAL").exists() and (d / "FAKE").exists()]
        if not candidates:
            raise SystemExit(f"could not find REAL/FAKE folders under {base}")
        test = candidates[0]
        print(f"[cifake] using {test}")
    n_r = copy_cap(test / "REAL", root / "cifake" / "real", {".png", ".jpg"}, limit)
    n_f = copy_cap(test / "FAKE", root / "cifake" / "ai", {".png", ".jpg"}, limit)
    print(f"[cifake] real={n_r} -> {root/'cifake'/'real'}")
    print(f"[cifake] ai={n_f} -> {root/'cifake'/'ai'}")


def main():
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    fetch_coco(root, args.max_images)
    if args.cifake:
        fetch_cifake(root, args.max_images)

    ai_dir = root / "val" / "ai"
    if not ai_dir.exists() or not any(ai_dir.iterdir()):
        ai_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[action needed] put AI-generated images into {ai_dir}/")
        print("  fastest: copy a folder of your own ComfyUI/Flux outputs there")
        if args.cifake:
            print(f"  or use CIFAKE: --ai-dir {root/'cifake'/'ai'}")

    print("\n[next] full sequence:")
    print("  python train_probe.py --features features/sid_set --out probe_v1")
    print(f"  python eval_robustness.py --real-dir {root/'val'/'real'} "
          f"--ai-dir {root/'val'/'ai'} --probe probe_v1 --max-per-class 100")


if __name__ == "__main__":
    main()
