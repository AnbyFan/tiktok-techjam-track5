#!/usr/bin/env python3
"""
Train and save the patch-based CLIP model for production use.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from PIL import Image
import open_clip
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

REAL_DIR = Path("data/val/real")
AI_DIR = Path("data/wildfake/Images/Diffusion_based/DALLE/DALLE/Advanced/DALLE3")


def extract_clip_features(model, preprocess, image, device):
    """Extract standard CLIP features."""
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.float().cpu().numpy()[0]


def extract_patch_clip_features(model, preprocess, image, device, num_patches=4):
    """Extract CLIP features from multiple patches."""
    w, h = image.size
    patch_w = w // num_patches
    patch_h = h // num_patches
    features = []
    for i in range(num_patches):
        for j in range(num_patches):
            box = (j * patch_w, i * patch_h, (j + 1) * patch_w, (i + 1) * patch_h)
            patch = image.crop(box)
            feat = extract_clip_features(model, preprocess, patch, device)
            features.append(feat)
    return np.mean(features, axis=0)


def main():
    print("=" * 60)
    print("TRAINING AND SAVING PATCH-BASED CLIP MODEL")
    print("=" * 60)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()
    print(f"\n[1] Loaded CLIP on {device}")

    # Get all validation images
    real_paths = sorted(REAL_DIR.glob("*.jpg"))
    ai_paths = sorted(AI_DIR.rglob("*.jpg"))
    print(f"    Real: {len(real_paths)}")
    print(f"    AI: {len(ai_paths)}")

    # Use all available data for training
    train_paths = real_paths + ai_paths
    train_labels = [0] * len(real_paths) + [1] * len(ai_paths)

    # Extract features
    print(f"\n[2] Extracting patch-based CLIP features...")
    features = []
    for i, path in enumerate(train_paths):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(train_paths)}")
        img = Image.open(path).convert("RGB")
        features.append(extract_patch_clip_features(model, preprocess, img, device))

    X_train = np.array(features)
    y_train = np.array(train_labels)

    # Train logistic regression
    print(f"\n[3] Training logistic regression...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluate on training data (for reference)
    probs_train = clf.predict_proba(X_train)[:, 1]
    preds_train = (probs_train >= 0.5).astype(int)
    train_acc = (preds_train == y_train).mean()
    print(f"    Training accuracy: {train_acc:.4f}")

    # Save the model
    print(f"\n[4] Saving model...")
    joblib.dump(clf, "probe_patch_clip.joblib")
    print(f"    Saved to probe_patch_clip.joblib")

    print(f"\n{'='*60}")
    print("MODEL SAVED SUCCESSFULLY!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
