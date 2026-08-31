#!/usr/bin/env python3
"""
Stream a Hugging Face image dataset (default: saberzl/SID_Set), extract
CLIP features + feature-drift stats (770-d), and cache to disk.

Same streaming logic as extract_features.py, but each image gets:
    feature = [f_clean(768-d), drift_mean(1-d), drift_std(1-d)]

The drift config (drift_k, drift_sigma) MUST match the ensemble config used
at inference so training and eval features line up.

Usage:
    python extract_sid_set_drift.py --out features/sid_set_drift \
        --per-class 4000 --drift-k 4 --drift-sigma 0.05
"""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import open_clip
from datasets import load_dataset
from tqdm import tqdm

from drift import add_drift_features


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="saberzl/SID_Set")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="features/sid_set_drift")
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
    p.add_argument("--drift-k", type=int, default=4)
    p.add_argument("--drift-sigma", type=float, default=0.05)
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


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device)
    model.eval()
    torch.backends.cudnn.benchmark = True
    print(f"[init] {args.model} ({args.pretrained}) on {device}  "
          f"drift_k={args.drift_k} drift_sigma={args.drift_sigma}")

    def encode(imgs):
        with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=(device == "cuda")):
            t = torch.stack(imgs).to(device, non_blocking=True)
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
                        pretrained=args.pretrained, drift=True,
                        drift_k=args.drift_k, drift_sigma=args.drift_sigma,
                        seed=args.seed)
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
        f_clean = encode(batch_tensors)
        feats = add_drift_features(model, f_clean, t, args.drift_k,
                                   args.drift_sigma, device,
                                   rng=np.random.default_rng(args.seed))
        for (rid, label), fv in zip(batch_meta, feats):
            feats_buf.append(fv)
            meta_buf.append((rid, label, args.dataset, "clean"))
            counts[label] += 1
            total_done += 1
            seen.add(rid)
            seen_f.write(rid + "\n")
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
                batch_tensors.append(preprocess(img))
            except Exception:
                continue
            batch_meta.append((rid, label))
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
