#!/usr/bin/env python3
"""
Extract and cache CLIP patch features for all training images.

This is slow (one CLIP forward pass per 4x4 patch per image). Run it ONCE;
the attention pooling trainer then loads from this cache and iterates fast.

Incremental: images already in the cache are skipped, so it's resumable.

Usage:
    python train/extract_features_cache.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from PIL import Image
import open_clip

REAL_DIR = Path("data/val/real")
PHONE_DIR = Path("data/hdrplus_real")
AI_DIR = Path("data/wildfake/Images/Diffusion_based/DALLE/DALLE/Advanced/DALLE3")
CACHE_FILE = Path("data/features_cache.npz")
NUM_PATCHES = 4


def extract_patch_features(model, preprocess, image, device, num_patches=NUM_PATCHES):
    w, h = image.size
    patch_w = w // num_patches
    patch_h = h // num_patches
    features = []
    for i in range(num_patches):
        for j in range(num_patches):
            box = (j * patch_w, i * patch_h, (j + 1) * patch_w, (i + 1) * patch_h)
            patch = image.crop(box)
            tensor = preprocess(patch).unsqueeze(0).to(device)
            with torch.inference_mode():
                feat = model.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat.float().cpu().numpy()[0])
    return np.array(features)  # (n_patches, dim)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()

    # Build the full image list with labels (0=real, 1=AI)
    real_paths = sorted(REAL_DIR.glob("*.jpg"))
    phone_paths = sorted(PHONE_DIR.glob("*.jpg")) if PHONE_DIR.exists() else []
    ai_paths = sorted(AI_DIR.rglob("*.jpg"))

    # (path, label) — phone photos are real (label 0)
    all_items = [(p, 0) for p in real_paths] + \
                [(p, 0) for p in phone_paths] + \
                [(p, 1) for p in ai_paths]
    print(f"Total images: {len(all_items)} "
          f"(COCO {len(real_paths)}, phone {len(phone_paths)}, AI {len(ai_paths)})")

    # Load existing cache if present (keyed by relative path string)
    cache_paths, cache_feats = [], []
    if CACHE_FILE.exists():
        data = np.load(CACHE_FILE, allow_pickle=True)
        cache_paths = list(data["paths"])
        cache_feats = data["features"]
        print(f"Loaded existing cache: {len(cache_paths)} images")
    cached_set = set(cache_paths)

    # Extract missing features
    new_feats = []
    new_paths = []
    dim = None
    for i, (path, label) in enumerate(all_items):
        key = str(path)
        if key in cached_set:
            continue
        img = Image.open(path).convert("RGB")
        feats = extract_patch_features(model, preprocess, img, device)
        if dim is None:
            dim = feats.shape[1]
        new_feats.append(feats)
        new_paths.append(key)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_items)} (cached {len(cached_set)}, new {len(new_paths)})")
        # Save periodically in case of interruption
        if len(new_paths) % 500 == 0:
            save_cache(cache_paths + new_paths,
                       np.array(cache_feats + new_feats) if (cache_feats or new_feats) else None)

    # Save final cache
    if new_feats:
        print(f"\nExtracted {len(new_feats)} new features. Saving cache...")
        all_feats = np.concatenate([np.array(cache_feats), np.array(new_feats)], axis=0) \
            if cache_feats else np.array(new_feats)
        save_cache(cache_paths + new_paths, all_feats)
    else:
        print("\nAll features already cached. Nothing to do.")

    print(f"\nDone. Cache: {CACHE_FILE} ({len(cache_paths) + len(new_paths)} images, dim={dim})")


def save_cache(paths, features):
    if features is None:
        return
    np.savez_compressed(CACHE_FILE, paths=np.array(paths, dtype=object), features=features)


if __name__ == "__main__":
    main()
