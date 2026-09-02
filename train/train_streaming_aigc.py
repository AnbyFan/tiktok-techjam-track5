#!/usr/bin/env python3
"""
Train CLIP probe using streaming AIGC-Detection-Benchmark dataset.

Streams data from HuggingFace without downloading the full dataset.
Trains on a subset (e.g., 10k images) to test if streaming works.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from PIL import Image
import open_clip
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

MAX_IMAGES = 10000  # Stream only 10k images for quick test


def main():
    print("=" * 60)
    print("STREAMING TRAINING: AIGC-Detection-Benchmark")
    print("=" * 60)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()
    print(f"\n[1] Loaded CLIP on {device}")

    # Stream dataset
    print("[2] Streaming dataset...")
    ds = load_dataset(
        'TheKernel01/AIGC-Detection-Benchmark',
        split='test',
        streaming=True
    )

    # Collect images and labels
    images = []
    labels = []
    sources = {}

    print("[3] Extracting features from streamed images...")
    for i, example in enumerate(ds):
        if i >= MAX_IMAGES:
            break

        img = example['image']
        source = example.get('source', 'unknown')
        label = 1 if source != 'WhichFaceIsReal' else 0  # 1=AI, 0=Real

        # Track sources
        sources[source] = sources.get(source, 0) + 1

        # Preprocess
        tensor = preprocess(img).unsqueeze(0).to(device)

        # Extract features
        with torch.inference_mode():
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        images.append(feat.float().cpu().numpy()[0])
        labels.append(label)

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{MAX_IMAGES} images")

    X = np.array(images)
    y = np.array(labels)

    print(f"\n[4] Dataset summary:")
    print(f"  Total images: {len(X)}")
    print(f"  Real: {np.sum(y == 0)}")
    print(f"  AI: {np.sum(y == 1)}")
    print(f"  Sources: {sources}")

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
    model_path = "probe_streaming_aigc.joblib"
    joblib.dump(clf, model_path)
    print(f"\n[7] Saved model to {model_path}")


if __name__ == "__main__":
    main()
