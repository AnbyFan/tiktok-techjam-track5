#!/usr/bin/env python3
"""
Extract CLIP features from the local Midjourney folder (all AI=1) and write
the same shard format as extract_features_folder.py, so train_probe_cw.py can
mix it into the "all generators" set.

AI-only variant of extract_features_folder.py (which requires both real and AI
dirs). Applies the same Track 5 augmentation (--augment --aug-copies N) so the
Midjourney contribution is robust to the eval transforms, matching sid_set_aug.

Usage:
    python extract_midjourney.py --ai-dir data/midjourney/midjourney \
        --out features/midjourney --augment --aug-copies 3
"""

import argparse
import csv
import json
import zlib
from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

from eval_robustness import apply_transform, build_specs

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ai-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dataset-name", default="midjourney")
    p.add_argument("--model", default="ViT-L-14")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--shard-size", type=int, default=2048)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--aug-copies", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_aug_pool():
    families = {}
    for name, kind, params in build_specs():
        if kind == "clean":
            continue
        families.setdefault(kind, []).append((name, params))
    return families


def sample_transform(rng, families, kinds):
    kind = kinds[rng.integers(len(kinds))]
    name, params = families[kind][rng.integers(len(families[kind]))]
    return name, kind, params


def list_images(d, limit):
    paths = sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in EXTS)
    return paths[:limit] if limit else paths


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    families = build_aug_pool()
    kinds = sorted(families)

    items = [(p, 1) for p in list_images(args.ai_dir, args.max_images)]
    rows_per_img = 1 + (args.aug_copies if args.augment else 0)
    print(f"[data] ai={len(items)} x{rows_per_img} rows/img "
          f"= {len(items) * rows_per_img} features")
    if not items:
        raise SystemExit(f"No images found in {args.ai_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device)
    model.eval()
    torch.backends.cudnn.benchmark = True
    print(f"[init] {args.model} ({args.pretrained}) on {device}")

    def encode(imgs):
        with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=(device == "cuda")):
            t = torch.stack(imgs).to(device, non_blocking=True)
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()

    manifest = {"counts": {"1": 0}, "shards": [],
                "dataset": args.dataset_name, "model": args.model,
                "pretrained": args.pretrained, "split": "local",
                "augment": args.augment, "aug_copies": args.aug_copies,
                "seed": args.seed}
    feats_buf, meta_buf, shard_idx = [], [], 0

    def flush(force=False):
        nonlocal feats_buf, meta_buf, shard_idx
        if not feats_buf or (not force and len(feats_buf) < args.shard_size):
            return
        np.save(out / f"features_{shard_idx:05d}.npy",
                np.stack(feats_buf).astype(np.float32))
        with (out / f"meta_{shard_idx:05d}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["img_id", "label", "dataset", "transform"])
            w.writerows(meta_buf)
        manifest["shards"].append(f"features_{shard_idx:05d}.npy")
        shard_idx += 1
        feats_buf, meta_buf = [], []
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    batches = [items[i:i + args.batch_size]
               for i in range(0, len(items), args.batch_size)]
    skipped = 0
    for batch in tqdm(batches, desc="extracting", unit="batch"):
        tensors, meta = [], []
        for path, label in batch:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                skipped += 1
                continue
            rng = np.random.default_rng(
                (args.seed + zlib.crc32(str(path).encode())) % 2**32)
            variants = [("clean", img)]
            if args.augment:
                for _ in range(args.aug_copies):
                    name, kind, params = sample_transform(rng, families, kinds)
                    variants.append((name, apply_transform(img, kind, params, rng)))
            for tname, vimg in variants:
                try:
                    tensors.append(preprocess(vimg))
                    suffix = "" if tname == "clean" else f"_{tname}"
                    meta.append((path.stem + suffix, label,
                                 args.dataset_name, tname))
                except Exception:
                    skipped += 1
        if not tensors:
            continue
        feats = encode(tensors)
        for (_, label, _, _), fv in zip(meta, feats):
            manifest["counts"][str(label)] += 1
        feats_buf.extend(feats)
        meta_buf.extend(meta)
        flush()
    flush(force=True)

    print(f"\n[done] {manifest['counts']}  skipped={skipped}")
    print(f"[done] {len(manifest['shards'])} shard(s) in {out}/")


if __name__ == "__main__":
    main()
