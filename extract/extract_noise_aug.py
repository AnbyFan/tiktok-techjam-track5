#!/usr/bin/env python3
"""
Extract CLIP features with NOISE-FOCUSED augmentation.

Unlike extract_features_folder.py / extract_midjourney.py (which sample one of
six transform families uniformly, so noise is only ~1/6 of the augmentation),
this script emits, for every image:

    clean + noise_s0.02 + noise_s0.05 + noise_s0.10

i.e. the exact Gaussian-noise levels used by the Track 5 eval harness. The
resulting feature dir is meant to be ADDED to an existing probe's feature_dirs
so the probe sees many more noise examples at the eval distribution.

Usage (AI-only):
    python extract_noise_aug.py --ai-dir data/midjourney/midjourney \
        --out features/midjourney_noise --dataset-name midjourney_noise

Usage (mixed real + AI):
    python extract_noise_aug.py --real-dir data/x/real --ai-dir data/x/ai \
        --out features/x_noise
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

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ai-dir", default=None)
    p.add_argument("--real-dir", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--dataset-name", default=None,
                   help="recorded in meta csv; default: out dir name")
    p.add_argument("--model", default="ViT-L-14")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--batch-size", type=int, default=128,
                   help="SOURCE images per batch; noise copies multiply tensors")
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--shard-size", type=int, default=2048)
    p.add_argument("--noise-sigmas", type=float, nargs="+",
                   default=[0.02, 0.05, 0.10])
    p.add_argument("--include-clean", action="store_true",
                   help="also emit a clean row per image (default: noise-only)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def list_images(d, limit):
    paths = sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in EXTS)
    return paths[:limit] if limit else paths


def main():
    args = parse_args()
    if not args.ai_dir and not args.real_dir:
        raise SystemExit("Provide --ai-dir and/or --real-dir.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset_name or out.name

    items = []
    if args.real_dir:
        items += [(p, 0) for p in list_images(args.real_dir, args.max_per_class)]
    if args.ai_dir:
        items += [(p, 1) for p in list_images(args.ai_dir, args.max_per_class)]
    if not items:
        raise SystemExit("No images found.")

    sigmas = args.noise_sigmas
    rows_per_img = len(sigmas) + (1 if args.include_clean else 0)
    n_real = sum(1 for _, l in items if l == 0)
    print(f"[data] real={n_real} ai={len(items) - n_real} x{rows_per_img} "
          f"rows/img = {len(items) * rows_per_img} features")
    print(f"[data] noise sigmas: {sigmas} include_clean={args.include_clean}")

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

    counts = {}
    for _, l in items:
        counts[str(l)] = counts.get(str(l), 0)
    manifest = {"counts": {}, "shards": [],
                "dataset": dataset_name, "model": args.model,
                "pretrained": args.pretrained, "split": "local",
                "noise_augmented": True, "noise_sigmas": sigmas,
                "include_clean": args.include_clean, "seed": args.seed}
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

    def add_noise(img_tensor, sigma, rng):
        noise = rng.normal(0.0, sigma, img_tensor.shape).astype(np.float32)
        noisy = img_tensor + torch.from_numpy(noise)
        return torch.clamp(noisy, 0.0, 1.0)

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
            clean_t = preprocess(img)
            rng = np.random.default_rng(
                (args.seed + zlib.crc32(str(path).encode())) % 2**32)
            if args.include_clean:
                tensors.append(clean_t)
                meta.append((path.stem, label, dataset_name, "clean"))
            for sigma in sigmas:
                nt = add_noise(clean_t, sigma, rng)
                tensors.append(nt)
                meta.append((f"{path.stem}_noise_s{sigma}", label,
                             dataset_name, f"noise_s{sigma}"))
        if not tensors:
            continue
        feats = encode(tensors)
        for (_, label, _, _), fv in zip(meta, feats):
            manifest["counts"][str(label)] = \
                manifest["counts"].get(str(label), 0) + 1
            counts[str(label)] += 1
        feats_buf.extend(feats)
        meta_buf.extend(meta)
        flush()
    flush(force=True)

    print(f"\n[done] {manifest['counts']}  skipped={skipped}")
    print(f"[done] {len(manifest['shards'])} shard(s) in {out}/")


if __name__ == "__main__":
    main()
