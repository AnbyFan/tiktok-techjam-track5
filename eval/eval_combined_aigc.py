#!/usr/bin/env python3
"""
Evaluate combined AIGC probe on our validation set.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from PIL import Image
import open_clip
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score

MODEL_PATH = "probe_combined_aigc.joblib"
REAL_DIR = Path("data/val/real")
AI_DIR = Path("data/wildfake/Images/Diffusion_based/DALLE/DALLE/Advanced/DALLE3")


def main():
    print("=" * 60)
    print("EVALUATION: Combined AIGC Probe on Validation Set")
    print("=" * 60)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()
    print(f"\n[1] Loaded CLIP on {device}")

    # Load probe
    clf = joblib.load(MODEL_PATH)
    print(f"[2] Loaded probe from {MODEL_PATH}")

    # Get validation images
    real_paths = sorted(REAL_DIR.glob("*.jpg"))[:500]
    ai_paths = sorted(AI_DIR.rglob("*.jpg"))[:500]
    print(f"    Real: {len(real_paths)}")
    print(f"    AI: {len(ai_paths)}")

    # Extract features
    print("[3] Extracting features...")
    features = []
    labels = []

    for path in real_paths:
        img = Image.open(path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.inference_mode():
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        features.append(feat.float().cpu().numpy()[0])
        labels.append(0)

    for path in ai_paths:
        img = Image.open(path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.inference_mode():
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        features.append(feat.float().cpu().numpy()[0])
        labels.append(1)

    X = np.array(features)
    y = np.array(labels)

    # Evaluate
    probs = clf.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    auroc = roc_auc_score(y, probs)

    real_acc = accuracy_score(y[y == 0], preds[y == 0])
    ai_acc = accuracy_score(y[y == 1], preds[y == 1])

    print(f"\n[4] Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Real accuracy: {real_acc:.4f}")
    print(f"  AI accuracy: {ai_acc:.4f}")

    # Compare with original keeper
    print(f"\n[5] Comparison with original keeper:")
    print(f"  Original: clean=0.9860, mean=0.9814, worst=0.9710")


if __name__ == "__main__":
    main()
