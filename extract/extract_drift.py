#!/usr/bin/env python3
"""
Extract CLIP features + feature-drift stats (770-d) for training a drift probe.

For each image it computes:
    f_clean    = normalize(CLIP(x))                    # 768-d
    drift_mean, drift_std                              # 2-d (K noisy re-encodes)
    feature    = [f_clean, drift_mean, drift_std]      # 770-d

This MUST match ensemble_core.py's drift config (drift_k, drift_sigma) so the
training features line up with what the scorer computes at inference.

Usage:
    python extract_drift.py \
        --real-dir data/cifake/real --ai-dir data/cifake/ai \
        --out features/cifake_drift --drift-k 4 --drift-sigma 0.2
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

from drift import add_drift_features

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real-dir", required=True)
    p.add_argument("--ai-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dataset-name", default="drift")
    p.add_argument("--model", default="ViT-L-14")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--shard-size", type=int, default=2048)
    p.add_argument("--drift-k", type=int, default=4)
    p.add_argument("--drift-sigma", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def list_images(d, limit):
    paths = sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in EXTS)
    return paths[:limit] if limit else paths


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = ([(p, 0) for p in list_images(args.real_dir, args.max_per_class)] +
             [(p, 1) for p in list_images(args.ai_dir, args.max_per_class)])
    print(f"[data] real={sum(1 for _, l in items if l == 0)} "
          f"ai={sum(1 for _, l in items if l == 1)}  "
          f"drift_k={args.drift_k} drift_sigma={args.drift_sigma}")
    if not items:
        raise SystemExit("No images found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device)
    model.eval()
    torch.backends.cudnn.benchmark = True
    print(f"[init] {args.model} ({args.pretrained}) on {device}")

    manifest = {"counts": {"0": 0, "1": 0}, "shards": [],
                "dataset": args.dataset_name, "model": args.model,
                "pretrained": args.pretrained, "split": "local",
                "drift": True, "drift_k": args.drift_k,
                "drift_sigma": args.drift_sigma, "seed": args.seed}
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
                tensors.append(preprocess(img))
                meta.append((path.stem, label))
            except Exception:
                skipped += 1
        if not tensors:
            continue
        t = torch.stack(tensors)
        # Clean features.
        with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=(device == "cuda")):
            f = model.encode_image(t.to(device, non_blocking=True))
            f = f / f.norm(dim=-1, keepdim=True)
        f_clean = f.float().cpu().numpy()
        # Drift features (reproducible per-image rng, matching scorer behavior).
        feats = add_drift_features(
            model, f_clean, t, args.drift_k, args.drift_sigma, device,
            rng=np.random.default_rng(args.seed))
        for (stem, label), fv in zip(meta, feats):
            manifest["counts"][str(label)] += 1
            feats_buf.append(fv)
            meta_buf.append((stem, label, args.dataset_name, "clean"))
        flush()
    flush(force=True)

    print(f"\n[done] {manifest['counts']}  skipped={skipped}")
    print(f"[done] {len(manifest['shards'])} shard(s) in {out}/")


if __name__ == "__main__":
    main()
