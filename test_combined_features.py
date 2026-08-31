#!/usr/bin/env python3
"""
Test combined CLIP + frequency features on eval set.
Quick test to see if frequency features help when concatenated with CLIP.
"""

import numpy as np
import torch
import open_clip
from PIL import Image
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from extract_freq_features import extract_features_for_image

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    real_dir = "data/val/real"
    ai_dir = "data/wildfake/Images/Diffusion_based/DALLE/DALLE/Advanced/DALLE3"
    max_per_class = 200  # Quick test
    
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()
    print(f"[init] CLIP on {device}")
    
    def encode_clip(imgs):
        with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=(device == "cuda")):
            t = torch.stack(imgs).to(device, non_blocking=True)
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()
    
    def get_features(directory, max_images):
        paths = sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in EXTS)
        paths = paths[:max_images]
        
        clip_feats = []
        freq_feats = []
        
        batch_imgs = []
        for path in tqdm(paths, desc=directory.split("/")[-1], unit="img"):
            try:
                img = Image.open(path).convert("RGB")
                clip_feats.append(preprocess(img))
                freq_feats.append(extract_features_for_image(img))
            except:
                continue
        
        clip_arr = encode_clip(clip_feats)
        freq_arr = np.stack(freq_feats).astype(np.float32)
        
        return clip_arr, freq_arr
    
    print("[extract] Real images...")
    real_clip, real_freq = get_features(real_dir, max_per_class)
    print(f"  CLIP: {real_clip.shape}, Freq: {real_freq.shape}")
    
    print("[extract] AI images...")
    ai_clip, ai_freq = get_features(ai_dir, max_per_class)
    print(f"  CLIP: {ai_clip.shape}, Freq: {ai_freq.shape}")
    
    # Test 1: CLIP only
    X_clip = np.concatenate([real_clip, ai_clip])
    y = np.concatenate([np.zeros(len(real_clip)), np.ones(len(ai_clip))])
    
    clf = LogisticRegression(max_iter=3000, class_weight={0: 1.0, 1: 1.5})
    clf.fit(X_clip, y)
    probs = clf.predict_proba(X_clip)[:, 1]
    preds = (probs >= 0.5).astype(int)
    print(f"\n[CLIP only] Acc: {accuracy_score(y, preds):.4f} AUROC: {roc_auc_score(y, probs):.4f}")
    
    # Test 2: CLIP + Frequency
    X_combined = np.concatenate([
        np.concatenate([real_clip, real_freq], axis=1),
        np.concatenate([ai_clip, ai_freq], axis=1)
    ])
    
    clf2 = LogisticRegression(max_iter=3000, class_weight={0: 1.0, 1: 1.5})
    clf2.fit(X_combined, y)
    probs2 = clf2.predict_proba(X_combined)[:, 1]
    preds2 = (probs2 >= 0.5).astype(int)
    print(f"[CLIP+Freq] Acc: {accuracy_score(y, preds2):.4f} AUROC: {roc_auc_score(y, probs2):.4f}")
    
    # Test 3: Frequency only (for reference)
    X_freq = np.concatenate([real_freq, ai_freq])
    clf3 = LogisticRegression(max_iter=3000)
    clf3.fit(X_freq, y)
    probs3 = clf3.predict_proba(X_freq)[:, 1]
    preds3 = (probs3 >= 0.5).astype(int)
    print(f"[Freq only] Acc: {accuracy_score(y, preds3):.4f} AUROC: {roc_auc_score(y, probs3):.4f}")


if __name__ == "__main__":
    main()
