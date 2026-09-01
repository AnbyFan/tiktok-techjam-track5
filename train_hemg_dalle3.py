#!/usr/bin/env python3
"""
Train CLIP probe using Hemg + DALLE3 combined dataset.

Uses:
- AI: Hemg AI images + DALLE3
- Real: Hemg real images
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import pandas as pd
from PIL import Image
import open_clip
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from io import BytesIO

# Paths
AI_DIR = Path("data/wildfake/Images/Diffusion_based/DALLE/DALLE/Advanced/DALLE3")
PARQUET_FILES = [
    r"C:\Users\wongq\.cache\huggingface\hub\datasets--Hemg--AI-Generated-vs-Real-Images-Datasets\snapshots\e270a0ad14b3b18a80a78d64e8ad5ec3eb6f798c\data\train-00003-of-00006-f635132ef309a732.parquet",
    r"C:\Users\wongq\.cache\huggingface\hub\datasets--Hemg--AI-Generated-vs-Real-Images-Datasets\snapshots\e270a0ad14b3b18a80a78d64e8ad5ec3eb6f798c\data\train-00000-of-00006-336b26d54a26e17a.parquet",
]

MAX_HEMG = 10000
MAX_DALLE3 = 2000


def main():
    print("=" * 60)
    print("TRAINING: Hemg + DALLE3 Combined")
    print("=" * 60)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()
    print(f"\n[1] Loaded CLIP on {device}")

    # Load Hemg data
    print("[2] Loading Hemg data...")
    dfs = []
    for pf in PARQUET_FILES:
        df = pd.read_parquet(pf)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Sample balanced subset
    real_df = df[df['label'] == 0].sample(min(MAX_HEMG // 2, len(df[df['label'] == 0])))
    ai_df = df[df['label'] == 1].sample(min(MAX_HEMG // 2, len(df[df['label'] == 1])))
    df = pd.concat([real_df, ai_df], ignore_index=True)
    print(f"    Hemg: {len(df)} images")

    # Get DALLE3 images
    print("[3] Loading DALLE3 images...")
    dalle3_paths = sorted(AI_DIR.rglob("*.jpg"))[:MAX_DALLE3]
    print(f"    DALLE3: {len(dalle3_paths)} images")

    # Extract features
    print("[4] Extracting CLIP features...")
    features = []
    labels = []

    # Hemg images
    for i, row in enumerate(df.itertuples()):
        if (i + 1) % 2000 == 0:
            print(f"    Hemg: {i + 1}/{len(df)}")

        img_data = row.image
        if isinstance(img_data, dict):
            img = Image.open(BytesIO(img_data['bytes']))
        else:
            img = Image.open(BytesIO(img_data))

        img = img.convert('RGB')
        tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.inference_mode():
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        features.append(feat.float().cpu().numpy()[0])
        labels.append(row.label)

    # DALLE3 images
    for i, path in enumerate(dalle3_paths):
        if (i + 1) % 500 == 0:
            print(f"    DALLE3: {i + 1}/{len(dalle3_paths)}")

        img = Image.open(path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.inference_mode():
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        features.append(feat.float().cpu().numpy()[0])
        labels.append(1)  # AI

    X = np.array(features)
    y = np.array(labels)

    print(f"\n[5] Dataset summary:")
    print(f"  Total: {len(X)}")
    print(f"  Real: {np.sum(y == 0)}")
    print(f"  AI: {np.sum(y == 1)}")

    # Train with different class weights
    print(f"\n[6] Training with different class weights...")
    best_acc = 0
    best_w = 0

    for w_ai in [1.0, 2.0, 3.0, 4.0, 4.5, 5.0]:
        clf = LogisticRegression(max_iter=1000, class_weight={0: 1, 1: w_ai})
        clf.fit(X, y)

        probs = clf.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y, preds)

        if acc > best_acc:
            best_acc = acc
            best_w = w_ai
            best_clf = clf

        print(f"  w={w_ai:.1f}: acc={acc:.4f}")

    # Evaluate best model
    probs = best_clf.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    auroc = roc_auc_score(y, probs)

    real_acc = accuracy_score(y[y == 0], preds[y == 0])
    ai_acc = accuracy_score(y[y == 1], preds[y == 1])

    print(f"\n[7] Best results (w={best_w}):")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Real accuracy: {real_acc:.4f}")
    print(f"  AI accuracy: {ai_acc:.4f}")

    # Save model
    import joblib
    model_path = "probe_hemg_dalle3.joblib"
    joblib.dump(best_clf, model_path)
    print(f"\n[8] Saved model to {model_path}")


if __name__ == "__main__":
    main()
