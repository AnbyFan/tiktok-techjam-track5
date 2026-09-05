#!/usr/bin/env python3
"""
Measure the false-positive rate of the attention pooling model on real
phone photos (Google HDR+ dataset). These are all REAL, so any image
flagged as AI (prob >= 0.5) is a false positive.

Usage:
    python eval/eval_phone_fpr.py [model_path]

Defaults to models/model_attention_pooling.joblib
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import joblib
import numpy as np
import open_clip
from PIL import Image

PHONE_DIR = Path("data/hdrplus_real")


class AttentionPooling(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        attn_out, _ = self.attn(query, x, x)
        attn_out = self.norm(attn_out + query)
        attn_out = self.proj(attn_out)
        return attn_out.squeeze(1)


def extract_patch_features(model, preprocess, image, device, num_patches=4):
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
    return np.array(features)


def main():
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/model_attention_pooling.joblib")
    print(f"Model: {model_path}")
    print(f"Phone dir: {PHONE_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    clip_model.eval()

    model_state = joblib.load(model_path)
    dim = model_state['dim']
    n_heads = model_state['n_heads']
    attn_pool = AttentionPooling(dim, n_heads=n_heads).to(device)
    attn_pool.load_state_dict(model_state['attn_pool'])
    attn_pool.eval()
    classifier = nn.Linear(dim, 1).to(device)
    classifier.load_state_dict(model_state['classifier'])
    classifier.eval()

    phone_paths = sorted(PHONE_DIR.glob("*.jpg"))
    print(f"\nEvaluating {len(phone_paths)} phone photos...\n")

    probs = []
    for i, path in enumerate(phone_paths):
        img = Image.open(path).convert("RGB")
        feats = extract_patch_features(clip_model, preprocess, img, device)
        feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.inference_mode():
            pooled = attn_pool(feats_t)
            logits = classifier(pooled)
            prob = torch.sigmoid(logits).item()
        probs.append((path.name, prob))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(phone_paths)}")

    probs.sort(key=lambda x: -x[1])
    n_fp = sum(1 for _, p in probs if p >= 0.5)
    print(f"\n{'='*60}")
    print(f"PHONE PHOTO FALSE-POSITIVE REPORT")
    print(f"{'='*60}")
    print(f"Total phone photos: {len(probs)}")
    print(f"Flagged as AI (FP): {n_fp} ({n_fp/len(probs)*100:.1f}%)")
    print(f"Mean AI probability: {np.mean([p for _,p in probs]):.4f}")
    print(f"Median AI probability: {np.median([p for _,p in probs]):.4f}")

    print(f"\nTop 15 most AI-like phone photos:")
    for name, p in probs[:15]:
        flag = "FP" if p >= 0.5 else "ok"
        print(f"  [{flag}] {p:.4f}  {name}")

    # Save full results
    out_file = Path("outputs") / "phone_fpr_results.txt"
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"FP rate: {n_fp}/{len(probs)}\n\n")
        for name, p in probs:
            f.write(f"{p:.4f}  {name}\n")
    print(f"\nFull results saved to {out_file}")


if __name__ == "__main__":
    main()
