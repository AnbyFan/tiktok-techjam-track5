#!/usr/bin/env python3
"""
Stream sid_set, extract CLIP features with NOISE AUGMENTATION.

Each image gets multiple versions:
- 1 clean version
- 1 version with noise sigma=0.02
- 1 version with noise sigma=0.05
- 1 version with noise sigma=0.10

This teaches probes to be robust to noise transforms.

Usage:
    python extract_sid_set_noisy.py --out features/sid_set_noisy \
        --per-class 4000
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import open_clip
from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="saberzl/SID_Set")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="features/sid_set_noisy")
    p.add_argument("--image-col", default="image")
    p.add_argument("--label-col", default="label")
    p.add_argument("--id-col", default="img_id")
    p.add_argument("--real-label", type=int, default=0)
    p.add_argument("--ai-label", type=int, default=1)
    p.add_argument("--per-class", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--model", default="ViT-L-14")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--shard-size", type=int, default=2048)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--noise-sigmas", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--token", default=None)
    return p.parse_args()


def stream_rows(args):
    while True:
        try:
            ds = load_dataset(args.dataset, None, split=args.split,
                              streaming=True, token=args.token)
            for row in ds:
                yield row
            return
        except Exception as e:
            print(f"[stream] error: {e!r} -- retrying in 30 s", flush=True)
            time.sleep(30)


def add_noise_to_tensor(tensor, sigma, rng):
    """Add Gaussian noise to a preprocessed tensor."""
    noise = rng.normal(0.0, sigma, tensor.shape).astype(np.float32)
    noisy = tensor + torch.from_numpy(noise)
    return torch.clamp(noisy, 0.0, 1.0)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device)
    model.eval()
    torch.backends.cudnn.benchmark = True
    print(f"[init] {args.model} ({args.pretrained}) on {device}")
    print(f"[init] Noise sigmas: {args.noise_sigmas}")
    print(f"[init] Each image -> {1 + len(args.noise_sigmas)} versions (clean + noisy)")

    def encode(tensors):
        with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=(device == "cuda")):
            t = tensors.to(device, non_blocking=True)
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()

    seen_path = out / "seen_ids.txt"
    seen = set(seen_path.read_text().split()) if seen_path.exists() else set()
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"counts": {}, "shards": []}
    keep = (args.real_label, args.ai_label)
    counts = {l: manifest["counts"].get(str(l), 0) for l in keep}
    shard_idx = len(manifest["shards"])

    def save_manifest():
        manifest["counts"] = {str(k): v for k, v in counts.items()}
        manifest.update(dataset=args.dataset, split=args.split, model=args.model,
                        pretrained=args.pretrained, noise_augmented=True,
                        noise_sigmas=args.noise_sigmas, seed=args.seed)
        manifest_path.write_text(json.dumps(manifest, indent=2))

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
        save_manifest()

    def quotas_met():
        return all(counts[l] >= args.per_class for l in keep)

    seen_f = seen_path.open("a")
    batch_tensors, batch_meta = [], []
    total_done = sum(counts.values())

    def process_batch():
        nonlocal batch_tensors, batch_meta, total_done
        if not batch_tensors:
            return
        t = torch.stack(batch_tensors)
        feats = encode(t)
        for (rid, label, transform), fv in zip(batch_meta, feats):
            feats_buf.append(fv)
            meta_buf.append((rid, label, args.dataset, transform))
            counts[label] += 1
            total_done += 1
        seen_f.flush()
        pbar.update(len(batch_tensors))
        pbar.set_postfix(real=counts[args.real_label], ai=counts[args.ai_label])
        flush_shard()
        batch_tensors, batch_meta = [], []

    target = args.max_samples or (args.per_class * len(keep))
    with tqdm(desc="extracting", unit="img", total=target) as pbar:
        for i, row in enumerate(stream_rows(args)):
            if quotas_met() or (args.max_samples and total_done >= args.max_samples):
                break
            label = row.get(args.label_col)
            if label not in counts or counts[label] >= args.per_class:
                continue
            rid = str(row.get(args.id_col) or f"noid_{i}")
            if rid in seen:
                continue
            img = row.get(args.image_col)
            try:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                clean_tensor = preprocess(img)
            except Exception:
                continue

            # Add clean version
            batch_tensors.append(clean_tensor)
            batch_meta.append((rid, label, "clean"))

            # Add noisy versions
            rng = np.random.default_rng(args.seed + i)
            for sigma in args.noise_sigmas:
                noisy_tensor = add_noise_to_tensor(clean_tensor, sigma, rng)
                batch_tensors.append(noisy_tensor)
                batch_meta.append((rid, label, f"noise_s{sigma}"))

            seen.add(rid)
            seen_f.write(rid + "\n")

            if len(batch_tensors) == args.batch_size:
                process_batch()
        process_batch()

    flush_shard(force=True)
    save_manifest()
    seen_f.close()

    print(f"\n[done] real={counts[args.real_label]} ai={counts[args.ai_label]}")
    print(f"[done] {len(manifest['shards'])} shard(s) in {out}/")


if __name__ == "__main__":
    main()
