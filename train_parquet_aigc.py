#!/usr/bin/env python3
"""
Train CLIP probe using downloaded parquet dataset.

Uses the Hemg/AI-Generated-vs-Real-Images-Datasets parquet file.
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

PARQUET_PATH = r"C:\Users\wongq\.cache\huggingface\hub\datasets--Hemg--AI-Generated-vs-Real-Images-Datasets\snapshots\e270a0ad14b3b18a80a78d64e8ad5ec3eb6f798c\data\train-00000-of-00006-336b26d54a26e17a.parquet"
MAX_IMAGES = 5000  # Use subset for quick test


def main():
    print("=" * 60)
    print("TRAINING: AI-Generated vs Real Images (Parquet)")
    print("=" * 60)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()
    print(f"\n[1] Loaded CLIP on {device}")

    # Load parquet
    print("[2] Loading parquet...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"    Total images: {len(df)}")
    print(f"    Label distribution:")
    print(f"      0 (Real): {len(df[df['label'] == 0])}")
    print(f"      1 (AI): {len(df[df['label'] == 1])}")

    # Sample subset
    df = df.head(MAX_IMAGES)
    print(f"    Using: {len(df)} images")

    # Extract features
    print("[3] Extracting CLIP features...")
    features = []
    labels = []

    for i, row in df.iterrows():
        if (i + 1) % 1000 == 0:
            print(f"    Processed {i + 1}/{len(df)}")

        # Get image
        img_data = row['image']
        if isinstance(img_data, dict):
            img = Image.open(BytesIO(img_data['bytes']))
        else:
            img = Image.open(BytesIO(img_data))

        img = img.convert('RGB')

        # Preprocess
        tensor = preprocess(img).unsqueeze(0).to(device)

        # Extract features
        with torch.inference_mode():
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        features.append(feat.float().cpu().numpy()[0])
        labels.append(row['label'])

    X = np.array(features)
    y = np.array(labels)

    print(f"\n[4] Dataset summary:")
    print(f"  Total: {len(X)}")
    print(f"  Real: {np.sum(y == 0)}")
    print(f"  AI: {np.sum(y == 1)}")

    # Train logistic regression
    print(f"\n[5] Training logistic regression...")
    clf = LogisticRegression(max_iter=1000, class_weight={0: 1, 1: 4.5})
    clf.fit(X, y)

    # Evaluate
    probs = clf.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    auroc = roc_auc_score(y, probs)

    real_acc = accuracy_score(y[y == 0], preds[y == 0])
    ai_acc = accuracy_score(y[y == 1], preds[y == 1])

    print(f"\n[6] Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Real accuracy: {real_acc:.4f}")
    print(f"  AI accuracy: {ai_acc:.4f}")

    # Save model
    import joblib
    model_path = "probe_parquet_aigc.joblib"
    joblib.dump(clf, model_path)
    print(f"\n[7] Saved model to {model_path}")


if __name__ == "__main__":
    main()
