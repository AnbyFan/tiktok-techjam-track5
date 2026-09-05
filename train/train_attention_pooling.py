#!/usr/bin/env python3
"""
Train patch-based CLIP model with attention pooling (TAP) instead of mean pooling.

Based on: TAP: Tunable Attention Pooling (CVPR 2026 workshop)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import open_clip
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

REAL_DIR = Path("data/val/real")
PHONE_DIR = Path("data/hdrplus_real")  # real phone photos (Google HDR+) to cut iPhone false positives
AI_DIR = Path("data/wildfake/Images/Diffusion_based/DALLE/DALLE/Advanced/DALLE3")
PHONE_EVAL_HOLDOUT = 30  # phone photos held out of training to measure FP rate
OUTPUT_MODEL = str(Path(__file__).parent.parent / "models" / "model_attention_pooling_phone.joblib")


class AttentionPooling(nn.Module):
    """Learnable attention pooling layer."""

    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (batch, n_patches, dim)
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        # Self-attention with query
        attn_out, _ = self.attn(query, x, x)
        attn_out = self.norm(attn_out + query)
        attn_out = self.proj(attn_out)
        return attn_out.squeeze(1)  # (batch, dim)


def extract_patch_features(model, preprocess, image, device, num_patches=4):
    """Extract CLIP features from multiple patches (returns all patches)."""
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
    print("=" * 60)
    print("TRAINING ATTENTION POOLING MODEL (TAP)")
    print("=" * 60)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()
    print(f"\n[1] Loaded CLIP on {device}")

    # Get all images
    real_paths = sorted(REAL_DIR.glob("*.jpg"))
    phone_paths = sorted(PHONE_DIR.glob("*.jpg")) if PHONE_DIR.exists() else []
    ai_paths = sorted(AI_DIR.rglob("*.jpg"))

    # Hold out some phone photos for FP evaluation; add the rest to the real class
    phone_eval_paths = phone_paths[:PHONE_EVAL_HOLDOUT]
    phone_train_paths = phone_paths[PHONE_EVAL_HOLDOUT:]
    real_paths = real_paths + phone_train_paths

    print(f"    Real (COCO): {len(real_paths) - len(phone_train_paths)}")
    print(f"    Real (phone, train): {len(phone_train_paths)}")
    print(f"    Real (phone, eval holdout): {len(phone_eval_paths)}")
    print(f"    AI: {len(ai_paths)}")

    # Extract features for all images
    print(f"\n[2] Extracting patch features...")
    # Load cached features (run train/extract_features_cache.py first if missing)
    cache_file = Path("data/features_cache.npz")
    if not cache_file.exists():
        print(f"\nERROR: Feature cache not found at {cache_file}")
        print("Run: python train/extract_features_cache.py  (takes ~2 hours, one-time)")
        sys.exit(1)

    print(f"\n[2] Loading cached features from {cache_file}...")
    cache = np.load(cache_file, allow_pickle=True)
    feat_by_path = {p: f for p, f in zip(cache["paths"], cache["features"])}

    # Split cached features by source
    all_features = []
    all_labels = []
    phone_eval_feats = []
    for path in real_paths:  # COCO + phone_train (label 0)
        all_features.append(feat_by_path[str(path)])
        all_labels.append(0)
    for path in phone_eval_paths:  # held-out phone (label 0, eval only)
        phone_eval_feats.append(feat_by_path[str(path)])
    for path in ai_paths:  # AI (label 1)
        all_features.append(feat_by_path[str(path)])
        all_labels.append(1)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        all_features, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")

    # Initialize attention pooling
    dim = X_train[0].shape[1]  # 768 for ViT-L-14
    print(f"\n[3] Initializing attention pooling (dim={dim})...")
    attn_pool = AttentionPooling(dim, n_heads=4).to(device)

    # Prepare tensors
    X_train_t = torch.tensor(np.array(X_train), dtype=torch.float32).to(device)
    y_train_t = torch.tensor(np.array(y_train), dtype=torch.float32).to(device)
    X_test_t = torch.tensor(np.array(X_test), dtype=torch.float32).to(device)
    y_test_t = torch.tensor(np.array(y_test), dtype=torch.float32).to(device)

    # Add classifier head
    classifier = nn.Linear(dim, 1).to(device)

    # Training
    print(f"\n[4] Training...")
    optimizer = torch.optim.Adam(
        list(attn_pool.parameters()) + list(classifier.parameters()),
        lr=1e-4, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    # Train for a few epochs
    n_epochs = 5
    batch_size = 32

    for epoch in range(n_epochs):
        attn_pool.train()
        classifier.train()

        # Shuffle
        perm = torch.randperm(X_train_t.size(0), device=device)
        X_train_shuffled = X_train_t[perm]
        y_train_shuffled = y_train_t[perm]

        total_loss = 0
        correct = 0
        total = 0

        for i in range(0, X_train_shuffled.size(0), batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size].unsqueeze(1)

            # Forward pass
            pooled = attn_pool(X_batch)
            logits = classifier(pooled)

            # Loss
            loss = criterion(logits, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        train_acc = correct / total
        print(f"    Epoch {epoch+1}/{n_epochs}: Loss={total_loss/(X_train_t.size(0)//batch_size+1):.4f}, Acc={train_acc:.4f}")

    # Evaluate on test set
    print(f"\n[5] Evaluating on test set...")
    attn_pool.eval()
    classifier.eval()

    with torch.inference_mode():
        pooled_test = attn_pool(X_test_t)
        logits_test = classifier(pooled_test)
        probs_test = torch.sigmoid(logits_test).cpu().numpy().flatten()
        preds_test = (probs_test >= 0.5).astype(int)

    test_acc = accuracy_score(y_test, preds_test)
    print(f"    Test Accuracy: {test_acc:.4f}")

    # Evaluate on held-out phone photos (all REAL; any prob>=0.5 is a false positive)
    phone_fp = None
    if phone_eval_feats:
        print(f"\n[5b] Evaluating on {len(phone_eval_feats)} held-out phone photos...")
        phone_probs = []
        for feats in phone_eval_feats:
            feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.inference_mode():
                pooled = attn_pool(feats_t)
                logits = classifier(pooled)
                prob = torch.sigmoid(logits).item()
            phone_probs.append(prob)
        n_fp = sum(1 for p in phone_probs if p >= 0.5)
        phone_fp = n_fp / len(phone_probs)
        print(f"    Phone FP rate: {n_fp}/{len(phone_probs)} ({phone_fp*100:.1f}%)")
        print(f"    Mean phone AI-prob: {np.mean(phone_probs):.4f}")

    # Save model
    print(f"\n[6] Saving model...")
    model_state = {
        'attn_pool': attn_pool.state_dict(),
        'classifier': classifier.state_dict(),
        'dim': dim,
        'n_heads': 4
    }
    joblib.dump(model_state, OUTPUT_MODEL)
    print(f"    Saved to {OUTPUT_MODEL}")

    print(f"\n{'='*60}")
    print("ATTENTION POOLING MODEL TRAINED!")
    print(f"Test Accuracy: {test_acc:.4f}")
    if phone_fp is not None:
        print(f"Held-out Phone FP rate: {phone_fp*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
