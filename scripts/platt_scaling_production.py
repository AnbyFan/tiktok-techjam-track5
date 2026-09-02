#!/usr/bin/env python3
"""
Production-ready Platt scaling for Hemg+DALLE3 probe.

This script:
1. Trains a CalibratedClassifierCV on a calibration set
2. Saves the calibrated model
3. Evaluates on validation set
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
from sklearn.calibration import CalibratedClassifierCV

REAL_DIR = Path("data/val/real")
AI_DIR = Path("data/wildfake/Images/Diffusion_based/DALLE/DALLE/Advanced/DALLE3")
HEMG_DALLE3_PROBE = "probe_hemg_dalle3.joblib"
CALIBRATED_MODEL_PATH = "probe_platt_calibrated.joblib"


def extract_features(model, preprocess, paths, device, max_images=500):
    """Extract CLIP features from images."""
    features = []
    for path in sorted(paths)[:max_images]:
        img = Image.open(path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        features.append(feat.float().cpu().numpy()[0])
    return np.array(features)


def main():
    print("=" * 60)
    print("PLATT SCALING: Production Implementation")
    print("=" * 60)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()
    print(f"\n[1] Loaded CLIP on {device}")

    # Load base probe
    base_clf = joblib.load(HEMG_DALLE3_PROBE)
    print(f"[2] Loaded base probe from {HEMG_DALLE3_PROBE}")

    # Get validation images (will be split into calibration + test)
    real_paths = sorted(REAL_DIR.glob("*.jpg"))[:500]
    ai_paths = sorted(AI_DIR.rglob("*.jpg"))[:500]
    print(f"    Real: {len(real_paths)}")
    print(f"    AI: {len(ai_paths)}")

    # Extract features
    print("[3] Extracting features...")
    real_features = extract_features(model, preprocess, real_paths, device)
    ai_features = extract_features(model, preprocess, ai_paths, device)

    X_all = np.vstack([real_features, ai_features])
    y_all = np.array([0]*len(real_paths) + [1]*len(ai_paths))

    # Split into calibration (200) and test (600) sets
    # Use stratified split to maintain class balance
    from sklearn.model_selection import train_test_split
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_all, y_all, test_size=0.6, random_state=42, stratify=y_all)

    print(f"    Calibration set: {len(X_cal)} images")
    print(f"    Test set: {len(X_test)} images")

    # Train Platt scaling calibrator
    print("[4] Training Platt scaling calibrator...")
    calibrator = CalibratedClassifierCV(base_clf, method='sigmoid', cv=3)
    calibrator.fit(X_cal, y_cal)

    # Evaluate on test set
    print("[5] Evaluating on test set...")
    probs = calibrator.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    auroc = roc_auc_score(y_test, probs)
    real_acc = accuracy_score(y_test[y_test == 0], preds[y_test == 0])
    ai_acc = accuracy_score(y_test[y_test == 1], preds[y_test == 1])

    print(f"\n[6] Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Real accuracy: {real_acc:.4f}")
    print(f"  AI accuracy: {ai_acc:.4f}")

    # Compare with original keeper
    print(f"\n[7] Comparison:")
    print(f"  Original keeper: clean=0.9860, mean=0.9814, worst=0.9710")
    print(f"  Platt scaling:   clean={acc:.4f}")

    # Save calibrated model
    joblib.dump(calibrator, CALIBRATED_MODEL_PATH)
    print(f"\n[8] Saved calibrated model to {CALIBRATED_MODEL_PATH}")


if __name__ == "__main__":
    main()
