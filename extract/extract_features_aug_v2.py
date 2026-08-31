#!/usr/bin/env python3
"""
Enhanced feature extraction with:
1. 6 augmentation copies per AI image (instead of 3)
2. Combined transforms (e.g., JPEG + noise together)

Usage:
    python extract_features_aug_v2.py --out features/sid_set_aug_v2 \
        --per-class 4000 --aug-copies 6 --batch-size 48
"""

import argparse
import csv
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import open_clip
from datasets import load_dataset
from tqdm import tqdm

from eval_robustness import apply_transform, build_specs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="saberzl/SID_Set")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="features/sid_set_aug_v2")
    p.add_argument("--image-col", default="image")
    p.add_argument("--label-col", default="label")
    p.add_argument("--id-col", default="img_id")
    p.add_argument("--real-label", type=int, default=0)
    p.add_argument("--ai-label", type=int, default=1)
    p.add_argument("--per-class", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--model", default="ViT-L-14")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--shard-size", type=int, default=2048)
    p.add_argument("--aug-copies", type=int, default=6)
    p.add_argument("--use-combined", action="store_true",
                   help="Use combined transforms (JPEG+noise, blur+noise, etc.)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--token", default=None)
    return p.parse_args()


def build_aug_pool_with_combined():
    """Build augmentation pool including combined transforms."""
    families = {}
    for name, kind, params in build_specs():
        if kind == "clean":
            continue
        families.setdefault(kind, []).append((name, params))

    # Add combined transforms
    combined = [
        ("jpeg_q50_noise_s0.05", "combined_jpeg_noise", {"quality": 50, "sigma": 0.05}),
        ("jpeg_q70_noise_s0.05", "combined_jpeg_noise", {"quality": 70, "sigma": 0.05}),
        ("blur_s1.0_noise_s0.05", "combined_blur_noise", {"sigma_blur": 1.0, "sigma_noise": 0.05}),
        ("blur_s0.5_noise_s0.02", "combined_blur_noise", {"sigma_blur": 0.5, "sigma_noise": 0.02}),
        ("resize_0.5x_jpeg_q50", "combined_resize_jpeg", {"scale": 0.5, "quality": 50}),
    ]
    families["combined"] = combined

    return families


def apply_combined_transform(img, kind, params, rng):
    """Apply combined transforms."""
    from PIL import ImageFilter, ImageEnhance
    import io

    if kind == "combined_jpeg_noise":
        # Apply JPEG compression then noise
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=params["quality"])
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = arr + rng.normal(0.0, params["sigma"], arr.shape).astype(np.float32)
        return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8))

    elif kind == "combined_blur_noise":
        # Apply blur then noise
        img = img.filter(ImageFilter.GaussianBlur(radius=params["sigma_blur"]))
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = arr + rng.normal(0.0, params["sigma_noise"], arr.shape).astype(np.float32)
        return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8))

    elif kind == "combined_resize_jpeg":
        # Apply resize then JPEG compression
        w, h = img.size
        small = img.resize((max(1, int(w * params["scale"])), max(1, int(h * params["scale"]))), Image.BICUBIC)
        small = small.resize((w, h), Image.BICUBIC)
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=params["quality"])
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    else:
        # Fallback to single transform
        return apply_transform(img, kind, params, rng)


def stream_rows(args):
    while True:
        try:
            ds = load_dataset(args.dataset, args.config, split=args.split,
                              streaming=True, token=args.token)
            for row in ds:
                yield row
            return
        except Exception as e:
            print(f"[stream] error: {e!r} -- retrying in 30 s", flush=True)
            time.sleep(30)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Build augmentation pool
    if args.use_combined:
        families = build_aug_pool_with_combined()
        print("[aug] Using combined transforms")
    else:
        families = build_aug_pool_with_combined()  # Still use combined pool
        # Filter to only single transforms
        families = {k: v for k, v in families.items() if k != "combined"}

    kinds = sorted(families)
    print(f"[aug] Transform families: {kinds}")

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

    # Resume state
    manifest = {"counts": {}, "shards": []}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {"counts": {}, "shards": []}

    valid_shards = []
    for name in manifest.get("shards", []):
        try:
            np.load(out / name, mmap_mode="r")
            valid_shards.append(name)
        except Exception:
            print(f"[resume] dropping corrupt shard {name}", flush=True)
    manifest["shards"] = valid_shards

    done_rids = set()
    counts = {args.real_label: 0, args.ai_label: 0}
    for name in valid_shards:
        meta = name.replace("features_", "meta_").replace(".npy", ".csv")
        try:
            with (out / meta).open() as f:
                for r in csv.DictReader(f):
                    if r.get("transform") == "clean":
                        done_rids.add(r["img_id"])
                        lab = int(r["label"])
                        if lab in counts:
                            counts[lab] += 1
        except Exception:
            pass
    shard_idx = len(valid_shards)
    if done_rids:
        print(f"[resume] resuming: {len(done_rids)} base images on disk", flush=True)

    feats_buf, meta_buf = [], []

    def flush_shard(force=False):
        nonlocal shard_idx, feats_buf, meta_buf
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
        manifest["counts"] = {str(k): v for k, v in counts.items()}
        manifest.update(dataset=args.dataset, split=args.split,
                        model=args.model, pretrained=args.pretrained,
                        aug_copies=args.aug_copies, use_combined=args.use_combined,
                        seed=args.seed)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def quotas_met():
        return all(counts[l] >= args.per_class for l in counts)

    pend_tensors, pend_meta = [], []

    def process_pending():
        if not pend_tensors:
            return
        feats = encode(pend_tensors)
        for (rid, label, tname), fv in zip(pend_meta, feats):
            feats_buf.append(fv)
            meta_buf.append((rid, label, args.dataset, tname))
        flush_shard()
        pend_tensors.clear()
        pend_meta.clear()

    target = args.per_class * 2
    with tqdm(desc="extracting", unit="img", total=target) as pbar:
        for row in stream_rows(args):
            if quotas_met():
                break
            label = row.get(args.label_col)
            if label not in counts or counts[label] >= args.per_class:
                continue
            rid = str(row.get(args.id_col) or f"noid_{pbar.n}")
            if rid in done_rids:
                continue
            img = row.get(args.image_col)
            try:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                clean_t = preprocess(img)
            except Exception:
                continue
            counts[label] += 1
            done_rids.add(rid)
            pend_tensors.append(clean_t)
            pend_meta.append((rid, label, "clean"))

            if label == args.ai_label and args.aug_copies > 0:
                rng = np.random.default_rng(
                    (args.seed + zlib.crc32(rid.encode())) % 2**32)
                for _ in range(args.aug_copies):
                    # Randomly choose a transform family
                    kind = kinds[rng.integers(len(kinds))]
                    name, params = families[kind][rng.integers(len(families[kind]))]
                    try:
                        if kind == "combined":
                            vimg = apply_combined_transform(img, kind, params, rng)
                        else:
                            vimg = apply_transform(img, kind, params, rng)
                        pend_tensors.append(preprocess(vimg))
                        pend_meta.append((f"{rid}_{name}", label, name))
                    except Exception:
                        pass

            if len(pend_tensors) >= args.batch_size * 4:
                process_pending()
            pbar.update(1)
            pbar.set_postfix(real=counts[args.real_label], ai=counts[args.ai_label])
        process_pending()

    flush_shard(force=True)
    print(f"\n[done] real={counts[args.real_label]} ai={counts[args.ai_label]}")
    print(f"[done] {len(manifest['shards'])} shard(s) in {out}/")


if __name__ == "__main__":
    main()
