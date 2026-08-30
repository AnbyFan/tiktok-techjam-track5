#!/usr/bin/env python3
"""
Extract CLIP features from LOCAL image folders (CIFAKE, ComfyUI outputs,
downloaded subsets, ...). Writes the same shard format as
extract_features.py so train_probe.py can mix caches freely.

Augmented mode (--augment): each source image emits its CLEAN version plus
--aug-copies transformed versions, with transforms sampled from the exact
Track 5 spec table (same code as eval_robustness.py, so the training and
evaluation distributions match by construction). Family-first sampling:
pick one of the 6 transform families uniformly, then a severity uniformly
within it. Labels never change -- a blurred fake is still a fake.

Usage:
    # plain extraction (unchanged behavior)
    python extract_features_folder.py --real-dir data/cifake/real --ai-dir data/cifake/ai --out features/cifake

    # augmented extraction: 1 clean + 2 transformed rows per image
    python extract_features_folder.py --real-dir data/cifake/real --ai-dir data/cifake/ai --out features/cifake_aug --augment --aug-copies 2

Labels written: real-dir -> 0, ai-dir -> 1 (matches the SID_Set mapping).
Requires eval_robustness.py in the same folder (imports its transforms).
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
    p.add_argument("--real-dir", required=True)
    p.add_argument("--ai-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dataset-name", default=None,
                   help="recorded in meta csv; default: out dir name")
    p.add_argument("--model", default="ViT-L-14")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--batch-size", type=int, default=128,
                   help="SOURCE images per batch; --aug-copies multiplies the "
                        "actual tensor count internally")
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--shard-size", type=int, default=2048)
    p.add_argument("--augment", action="store_true",
                   help="emit clean + --aug-copies transformed rows per image")
    p.add_argument("--aug-copies", type=int, default=1,
                   help="1 -> 50%% clean features, 2 -> 33%%, 3 -> 25%%")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_aug_pool():
    """kind -> [(spec_name, params), ...] for every non-clean spec."""
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
    dataset_name = args.dataset_name or out.name
    families = build_aug_pool()
    kinds = sorted(families)

    items = [(p, 0) for p in list_images(args.real_dir, args.max_per_class)] + \
            [(p, 1) for p in list_images(args.ai_dir, args.max_per_class)]
    n_real = sum(1 for _, l in items if l == 0)
    rows_per_img = 1 + (args.aug_copies if args.augment else 0)
    print(f"[data] real={n_real} ai={len(items) - n_real} "
          f"x{rows_per_img} rows/img = {len(items) * rows_per_img} features")
    if n_real == 0 or n_real == len(items):
        raise SystemExit("Need images in BOTH --real-dir and --ai-dir.")

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

    manifest = {"counts": {"0": 0, "1": 0}, "shards": [],
                "dataset": dataset_name, "model": args.model,
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
                    meta.append((path.stem + suffix, label, dataset_name, tname))
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
    print(f"[next] python train_probe.py --features features/sid_set {out} "
          f"--out probe_v3")


if __name__ == "__main__":
    main()
