#!/usr/bin/env python3
"""
Extract CLIP features from Flux_AIGC_Dataset and save in the same format
as other feature caches.

Usage:
    python extract_flux_features.py --output-dir features/flux_aug
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import open_clip
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="ViT-L-14")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-images", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device)
    model.eval()
    print(f"[init] {args.model} on {device}")

    # Load Flux dataset
    print("[data] Loading Flux_AIGC_Dataset...")
    dataset = load_dataset('Bluemaki/Flux_AIGC_Dataset', split='train', streaming=True)

    # Check if we've already processed some images
    seen_file = out_dir / "seen_ids.txt"
    seen_ids = set()
    if seen_file.exists():
        seen_ids = set(seen_file.read_text().split())
        print(f"[resume] Found {len(seen_ids)} already processed")

    # Process images
    all_features = []
    all_meta = []
    processed = 0

    for i, sample in enumerate(tqdm(dataset, desc="Processing", unit="img")):
        # Skip if we've hit the max
        if args.max_images and processed >= args.max_images:
            break

        # Get image ID (use index or hash)
        img_id = f"flux_{i}"

        # Skip if already processed
        if img_id in seen_ids:
            continue

        try:
            # Get the image
            img = sample['image']

            # Preprocess
            tensor = preprocess(img).unsqueeze(0).to(device)

            # Extract features
            with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.float16, enabled=(device == "cuda")):
                feat = model.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)

            feat = feat.float().cpu().numpy()[0]
            all_features.append(feat)
            all_meta.append({
                "id": img_id,
                "label": 1,  # All Flux images are AI-generated
                "source": "flux"
            })
            processed += 1

            # Save in chunks
            if len(all_features) >= 10000:
                shard_id = len(all_features) // 10000
                np.save(out_dir / f"features_{shard_id:05d}.npy",
                       np.array(all_features))
                with (out_dir / f"meta_{shard_id:05d}.csv").open("w") as f:
                    for m in all_meta:
                        f.write(f"{m['id']},{m['label']},{m['source']}\n")
                all_features = []
                all_meta = []
                print(f"[saved] Shard {shard_id}")

        except Exception as e:
            print(f"[error] {img_id}: {e}")
            continue

    # Save remaining
    if all_features:
        shard_id = (processed - 1) // 10000
        np.save(out_dir / f"features_{shard_id:05d}.npy",
               np.array(all_features))
        with (out_dir / f"meta_{shard_id:05d}.csv").open("w") as f:
            for m in all_meta:
                f.write(f"{m['id']},{m['label']},{m['source']}\n")

    # Save manifest
    manifest = {
        "total_images": processed,
        "feature_dim": 768,
        "model": args.model,
        "pretrained": args.pretrained
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Save seen IDs
    seen_file.write_text("\n".join([f"flux_{i}" for i in range(processed)]))

    print(f"\n[done] Processed {processed} images")
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
