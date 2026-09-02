#!/usr/bin/env python3
"""
Train CLIP probe using full Hemg/AI-Generated-vs-Real-Images-Datasets.

Uses all 6 parquet files (152,710 images total).
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
from tqdm import tqdm

# All parquet files
PARQUET_FILES = [
    r"C:\Users\wongq\.cache\huggingface\hub\datasets--Hemg--AI-Generated-vs-Real-Images-Datasets\snapshots\e270a0ad14b3b18a80a78d64e8ad5ec3eb6f798c\data\train-00000-of-00006-336b26d54a26e17a.parquet",
    r"C:\Users\wongq\.cache\huggingface\hub\datasets--Hemg--AI-Generated-vs-Real-Images-Datasets\snapshots\e270a0ad14b3b18a80a78d64e8ad5ec3eb6f798c\data\train-00001-of-00006-8ad2d550254dea81.parquet",
    r"C:\Users\wongq\.cache\huggingface\hub\datasets--Hemg--AI-Generated-vs-Real-Images-Datasets\snapshots\e270a0ad14b3b18a80a78d64e8ad5ec3eb6f798c\data\train-00002-of-00006-ac8970f21c0418c1.parquet",
    r"C:\Users\wongq\.cache\huggingface\hub\datasets--Hemg--AI-Generated-vs-Real-Images-Datasets\snapshots\e270a0ad14b3b18a80a78d64e8ad5ec3eb6f798c\data\train-00003-of-00006-f635132ef309a732.parquet",
    r"C:\Users\wongq\.cache\huggingface\hub\datasets--Hemg--AI-Generated-vs-Real-Images-Datasets\snapshots\e270a0ad14b3b18a80a78d64e8ad5ec3eb6f798c\data\train-00004-of-00006-1101eaf5152e1c5f.parquet",
    r"C:\Users\wongq\.cache\huggingface\hub\datasets--Hemg--AI-Generated-vs-Real-Images-Datasets\snapshots\e270a0ad14b3b18a80a78d64e8ad5ec3eb6f798c\data\train-00005-of-00006-4bd152a5ab76dba7.parquet",
]

MAX_IMAGES = 20000  # Use subset for quick test


def main():
    print("=" * 60)
    print("TRAINING: Full Hemg Dataset")
    print("=" * 60)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()
    print(f"\n[1] Loaded CLIP on {device}")

    # Load all parquet files
    print("[2] Loading parquet files...")
    dfs = []
    for pf in PARQUET_FILES:
        df = pd.read_parquet(pf)
        dfs.append(df)
        print(f"    {Path(pf).name}: {len(df)} rows")

    df = pd.concat(dfs, ignore_index=True)
    print(f"    Total: {len(df)} images")
    print(f"    Labels:")
    print(f"      0 (Real): {len(df[df['label'] == 0])}")
    print(f"      1 (AI): {len(df[df['label'] == 1])}")

    # Sample balanced subset
    real_df = df[df['label'] == 0].sample(min(MAX_IMAGES // 2, len(df[df['label'] == 0])))
    ai_df = df[df['label'] == 1].sample(min(MAX_IMAGES // 2, len(df[df['label'] == 1])))
    df = pd.concat([real_df, ai_df], ignore_index=True)
    print(f"    Using: {len(df)} images (balanced)")

    # Extract features
    print("[3] Extracting CLIP features...")
    features = []
    labels = []

    for i, row in enumerate(df.itertuples()):
        if (i + 1) % 2000 == 0:
            print(f"    Processed {i + 1}/{len(df)}")

        # Get image
        img_data = row.image
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
        labels.append(row.label)

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
    model_path = "probe_full_hemg.joblib"
    joblib.dump(clf, model_path)
    print(f"\n[7] Saved model to {model_path}")


if __name__ == "__main__":
    main()
