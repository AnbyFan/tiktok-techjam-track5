#!/usr/bin/env python3
"""
Test multi-scale CLIP features for AI image detection.

Instead of extracting features at a single scale (224x224), extract at
multiple scales and concatenate them. This could help the model see both
global patterns (large scale) and local artifacts (small scale).

Scales to test:
- 224x224 (standard)
- 336x336 (medium)
- 448x448 (large)
- Combinations: 224+336, 224+448, 336+448, 224+336+448
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


def extract_multiscale_features(img, model, preprocessors, device):
    """Extract features at multiple scales and concatenate."""
    features = []
    for preprocess in preprocessors:
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        features.append(feat.float().cpu().numpy()[0])

    return np.concatenate(features)


def main():
    print("=" * 60)
    print("MULTISCALE FEATURES TEST")
    print("=" * 60)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess_224 = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()
    print(f"\n[1] Loaded CLIP ViT-L/14 on {device}")

    # Create preprocessors for different scales
    from torchvision import transforms

    def make_preprocessor(size):
        return transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

    preprocess_336 = make_preprocessor(336)
    preprocess_448 = make_preprocessor(448)

    print(f"[2] Created preprocessors for 224, 336, 448")

    # Get test images
    real_paths = sorted(REAL_DIR.glob("*.jpg"))[:50]
    ai_paths = sorted(AI_DIR.rglob("*.jpg"))[:50]
    print(f"[3] Using {len(real_paths)} real + {len(ai_paths)} AI images")

    # Test different feature combinations
    configs = [
        ("224 only", [preprocess_224]),
        ("336 only", [preprocess_336]),
        ("448 only", [preprocess_448]),
        ("224+336", [preprocess_224, preprocess_336]),
        ("224+448", [preprocess_224, preprocess_448]),
        ("336+448", [preprocess_336, preprocess_448]),
        ("224+336+448", [preprocess_224, preprocess_336, preprocess_448]),
    ]

    results = []

    for config_name, preprocessors in configs:
        print(f"\n[4] Testing: {config_name}...")

        # Extract features
        real_feats = []
        ai_feats = []

        for p in real_paths:
            img = Image.open(p).convert("RGB")
            feat = extract_multiscale_features(img, model, preprocessors, device)
            real_feats.append(feat)

        for p in ai_paths:
            img = Image.open(p).convert("RGB")
            feat = extract_multiscale_features(img, model, preprocessors, device)
            ai_feats.append(feat)

        # Combine and train a quick logistic regression
        X = np.vstack([real_feats, ai_feats])
        y = np.array([0]*len(real_feats) + [1]*len(ai_feats))

        clf = LogisticRegression(max_iter=1000, class_weight={0: 1, 1: 4.5})
        clf.fit(X, y)

        # Evaluate
        probs = clf.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y, preds)

        real_acc = accuracy_score(y[:len(real_feats)], preds[:len(real_feats)])
        ai_acc = accuracy_score(y[len(real_feats):], preds[len(real_feats):])

        feat_dim = X.shape[1]
        results.append({
            "config": config_name,
            "feat_dim": feat_dim,
            "accuracy": acc,
            "real_acc": real_acc,
            "ai_acc": ai_acc
        })

        print(f"    Feature dim: {feat_dim}")
        print(f"    Accuracy: {acc:.2%} (real={real_acc:.2%}, ai={ai_acc:.2%})")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  {'Config':15s} {'Dim':6s} {'Accuracy':10s} {'Real':10s} {'AI':10s}")
    print(f"  {'-'*15} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")

    for r in results:
        print(f"  {r['config']:15s} {r['feat_dim']:6d} {r['accuracy']:10.2%} {r['real_acc']:10.2%} {r['ai_acc']:10.2%}")

    # Best config
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\n  ✅ Best config: {best['config']} ({best['accuracy']:.2%})")


if __name__ == "__main__":
    main()
