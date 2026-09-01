#!/usr/bin/env python3
"""
Test different CLIP models for AI image detection.

Models to test:
- ViT-L/14 (current, 768-dim)
- ViT-H/14 (larger, 1024-dim)
- ViT-B/32 (smaller, 512-dim)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import open_clip
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

REAL_DIR = Path("data/val/real")
AI_DIR = Path("data/wildfake/Images/Diffusion_based/DALLE/DALLE/Advanced/DALLE3")


def test_model(model_name, pretrained_tag, real_paths, ai_paths):
    """Test a specific CLIP model."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_tag, device=device)
        model.eval()

        # Extract features
        def extract_features(paths):
            feats = []
            for p in paths:
                img = Image.open(p).convert("RGB")
                tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.inference_mode():
                    feat = model.encode_image(tensor)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                feats.append(feat.float().cpu().numpy()[0])
            return np.array(feats)

        real_feats = extract_features(real_paths[:50])
        ai_feats = extract_features(ai_paths[:50])

        X = np.vstack([real_feats, ai_feats])
        y = np.array([0]*len(real_feats) + [1]*len(ai_feats))

        # Try different class weights
        best_acc = 0
        best_w = 0
        for w in [1.0, 2.0, 3.0, 4.0, 4.5, 5.0]:
            clf = LogisticRegression(max_iter=1000, class_weight={0: 1, 1: w})
            clf.fit(X, y)
            probs = clf.predict_proba(X)[:, 1]
            preds = (probs >= 0.5).astype(int)
            acc = accuracy_score(y, preds)
            if acc > best_acc:
                best_acc = acc
                best_w = w

        feat_dim = X.shape[1]
        print(f"  {model_name}: {best_acc:.2%} (dim={feat_dim}, w={best_w})")

        # Cleanup
        del model
        torch.cuda.empty_cache()

        return best_acc, feat_dim, best_w

    except Exception as e:
        print(f"  {model_name}: ERROR - {str(e)[:100]}")
        return 0, 0, 0


def main():
    print("=" * 60)
    print("CLIP MODEL COMPARISON TEST")
    print("=" * 60)

    real_paths = sorted(REAL_DIR.glob("*.jpg"))
    ai_paths = sorted(AI_DIR.rglob("*.jpg"))
    print(f"\n[1] Using {min(50, len(real_paths))} real + {min(50, len(ai_paths))} AI images")

    # Test different models
    models = [
        ("ViT-B/32", "openai"),
        ("ViT-L/14", "openai"),
        ("ViT-H/14", "laion2b_s32b_b79k"),
    ]

    print(f"\n[2] Testing models...")
    results = []
    for model_name, pretrained in models:
        acc, dim, w = test_model(model_name, pretrained, real_paths, ai_paths)
        results.append((model_name, acc, dim, w))

    # Summary
    print(f"\n[3] Summary:")
    print(f"  {'Model':15s} {'Accuracy':10s} {'Dim':6s} {'Weight':8s}")
    print(f"  {'-'*15} {'-'*10} {'-'*6} {'-'*8}")
    for name, acc, dim, w in results:
        print(f"  {name:15s} {acc:10.2%} {dim:6d} {w:8.1f}")

    best = max(results, key=lambda x: x[1])
    print(f"\n  ✅ Best model: {best[0]} ({best[1]:.2%})")


if __name__ == "__main__":
    main()
