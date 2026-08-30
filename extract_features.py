#!/usr/bin/env python3
"""
Stream a Hugging Face image dataset, extract CLIP features on GPU, and cache
features + metadata to disk. Raw images are never stored.

Designed for an overnight run on a single 8 GB GPU (RTX 3070 Ti).

Usage:
    # 1. Verify label mapping first (no GPU work, ~1 min)
    python extract_features.py --probe-labels --probe-n 1000

    # 2. Smoke test end-to-end (~2 min)
    python extract_features.py --out features/smoke --max-samples 256

    # 3. Overnight run (10k real + 10k AI)
    python extract_features.py --out features/sid_set --per-class 10000

Requires: pip install open_clip_torch datasets pyarrow tqdm numpy pillow
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="saberzl/SID_Set")
    p.add_argument("--config", default=None, help="HF dataset config name, if any")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="features/sid_set")
    p.add_argument("--image-col", default="image")
    p.add_argument("--label-col", default="label")
    p.add_argument("--id-col", default="img_id")
    p.add_argument("--real-label", type=int, default=0,
                   help="confirm with --probe-labels before trusting")
    p.add_argument("--ai-label", type=int, default=1,
                   help="rows with any other label are skipped")
    p.add_argument("--per-class", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128,
                   help="drop to 64 if you hit OOM on 8 GB")
    p.add_argument("--model", default="ViT-L-14")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--shard-size", type=int, default=2048,
                   help="images per .npy shard flush")
    p.add_argument("--max-samples", type=int, default=None,
                   help="global cap, for smoke tests")
    p.add_argument("--probe-labels", action="store_true",
                   help="only print the label distribution, then exit")
    p.add_argument("--probe-n", type=int, default=1000)
    p.add_argument("--token", default=None, help="HF token if the dataset needs one")
    return p.parse_args()


def stream_rows(args):
    """Yield rows forever, recreating the stream on network errors."""
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


def probe(args):
    counts = Counter()
    for row in tqdm(stream_rows(args), total=args.probe_n, desc="probing"):
        counts[row.get(args.label_col)] += 1
        if sum(counts.values()) >= args.probe_n:
            break
    print("\nLabel distribution over sampled rows:", dict(counts))
    print("Update --real-label / --ai-label to match the dataset's mapping.")


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.probe_labels:
        probe(args)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device)
    model.eval()
    torch.backends.cudnn.benchmark = True
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[init] {args.model} ({args.pretrained}) on {device} -- {n_params:.0f}M params")

    def encode(imgs):
        with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=(device == "cuda")):
            t = torch.stack(imgs).to(device, non_blocking=True)
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()

    # --- resume state ---
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
        manifest.update(dataset=args.dataset, config=args.config,
                        split=args.split, model=args.model,
                        pretrained=args.pretrained)
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
            w.writerow(["img_id", "label", "dataset"])
            w.writerows(meta_buf)
        manifest["shards"].append(f"features_{shard_idx:05d}.npy")
        shard_idx += 1
        feats_buf, meta_buf = [], []
        save_manifest()

    def quotas_met():
        return all(counts[l] >= args.per_class for l in keep)

    seen_f = seen_path.open("a")
    batch_imgs, batch_meta = [], []
    total_done = sum(counts.values())

    def process_batch():
        nonlocal batch_imgs, batch_meta, total_done
        if not batch_imgs:
            return
        feats = encode(batch_imgs)
        for (rid, label), fv in zip(batch_meta, feats):
            feats_buf.append(fv)
            meta_buf.append((rid, label, args.dataset))
            counts[label] += 1
            total_done += 1
            seen.add(rid)
            seen_f.write(rid + "\n")
        seen_f.flush()
        pbar.update(len(batch_imgs))
        pbar.set_postfix(real=counts[args.real_label], ai=counts[args.ai_label])
        flush_shard()
        batch_imgs, batch_meta = [], []

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
                batch_imgs.append(preprocess(img))
            except Exception:
                continue  # corrupt/undecodable image
            batch_meta.append((rid, label))
            if len(batch_imgs) == args.batch_size:
                process_batch()
        process_batch()  # tail batch

    flush_shard(force=True)
    save_manifest()
    seen_f.close()

    print(f"\n[done] real={counts[args.real_label]} ai={counts[args.ai_label]}")
    print(f"[done] {len(manifest['shards'])} shard(s) in {out}/")
    print("[next] train_probe.py can np.load each features_*.npy and join "
          "meta_*.csv for labels.")


if __name__ == "__main__":
    main()
