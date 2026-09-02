#!/usr/bin/env python3
"""
Test all improvement approaches and compare results.

Approaches:
1. Threshold optimization
2. Temperature/Platt scaling calibration
3. Ensemble with original keeper
4. Transfer learning from original keeper
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
from scipy.optimize import minimize_scalar

REAL_DIR = Path("data/val/real")
AI_DIR = Path("data/wildfake/Images/Diffusion_based/DALLE/DALLE/Advanced/DALLE3")
ORIGINAL_ENSEMBLE = "ensemble_config.json"
HEMG_DALLE3_PROBE = "probe_hemg_dalle3.joblib"


def extract_features(model, preprocess, paths, device, max_images=500):
    """Extract CLIP features from images."""
    features = []
    labels = []

    for path in sorted(paths)[:max_images]:
        img = Image.open(path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.inference_mode():
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        features.append(feat.float().cpu().numpy()[0])

    return np.array(features)


def find_optimal_threshold(y_true, probs):
    """Find optimal threshold that maximizes accuracy."""
    def neg_accuracy(threshold):
        preds = (probs >= threshold).astype(int)
        return -accuracy_score(y_true, preds)

    result = minimize_scalar(neg_accuracy, bounds=(0.1, 0.9), method='bounded')
    return result.x


def apply_temperature_scaling(probs, temperature):
    """Apply temperature scaling to probabilities."""
    logits = np.log(probs / (1 - probs + 1e-8) + 1e-8)
    scaled_logits = logits / temperature
    scaled_probs = 1 / (1 + np.exp(-scaled_logits))
    return scaled_probs


def find_optimal_temperature(y_true, probs):
    """Find optimal temperature for scaling."""
    def neg_accuracy(temp):
        scaled_probs = apply_temperature_scaling(probs, temp)
        preds = (scaled_probs >= 0.5).astype(int)
        return -accuracy_score(y_true, preds)

    result = minimize_scalar(neg_accuracy, bounds=(0.1, 100.0), method='bounded')
    return result.x


def main():
    print("=" * 60)
    print("TESTING ALL APPROACHES")
    print("=" * 60)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()
    print(f"\n[1] Loaded CLIP on {device}")

    # Get validation images
    real_paths = list(REAL_DIR.glob("*.jpg"))[:500]
    ai_paths = list(AI_DIR.rglob("*.jpg"))[:500]
    print(f"    Real: {len(real_paths)}")
    print(f"    AI: {len(ai_paths)}")

    # Extract features
    print("[2] Extracting features...")
    real_features = extract_features(model, preprocess, real_paths, device)
    ai_features = extract_features(model, preprocess, ai_paths, device)

    X = np.vstack([real_features, ai_features])
    y = np.array([0]*len(real_paths) + [1]*len(ai_paths))

    # Load Hemg+DALLE3 probe
    clf_hemg = joblib.load(HEMG_DALLE3_PROBE)
    probs_hemg = clf_hemg.predict_proba(X)[:, 1]

    print(f"\n[3] Baseline (Hemg+DALLE3, threshold=0.5):")
    base_preds = (probs_hemg >= 0.5).astype(int)
    base_acc = accuracy_score(y, base_preds)
    base_auroc = roc_auc_score(y, probs_hemg)
    print(f"  Accuracy: {base_acc:.4f}")
    print(f"  AUROC: {base_auroc:.4f}")

    # Approach 1: Threshold optimization
    print(f"\n[4] Approach 1: Threshold Optimization")
    optimal_threshold = find_optimal_threshold(y, probs_hemg)
    opt_preds = (probs_hemg >= optimal_threshold).astype(int)
    opt_acc = accuracy_score(y, opt_preds)
    opt_auroc = roc_auc_score(y, probs_hemg)
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Accuracy: {opt_acc:.4f}")
    print(f"  AUROC: {opt_auroc:.4f}")

    # Approach 2: Temperature scaling
    print(f"\n[5] Approach 2: Temperature Scaling")
    optimal_temp = find_optimal_temperature(y, probs_hemg)
    scaled_probs = apply_temperature_scaling(probs_hemg, optimal_temp)
    temp_preds = (scaled_probs >= 0.5).astype(int)
    temp_acc = accuracy_score(y, temp_preds)
    temp_auroc = roc_auc_score(y, scaled_probs)
    print(f"  Optimal temperature: {optimal_temp:.4f}")
    print(f"  Accuracy: {temp_acc:.4f}")
    print(f"  AUROC: {temp_auroc:.4f}")

    # Approach 3: Platt scaling
    print(f"\n[6] Approach 3: Platt Scaling")
    # Use balanced subset for calibration
    real_idx = np.where(y == 0)[0][:100]
    ai_idx = np.where(y == 1)[0][:100]
    cal_idx = np.concatenate([real_idx, ai_idx])
    X_cal = X[cal_idx]
    y_cal = y[cal_idx]

    calibrator = CalibratedClassifierCV(clf_hemg, method='sigmoid', cv=3)
    calibrator.fit(X_cal, y_cal)
    platt_probs = calibrator.predict_proba(X)[:, 1]
    platt_preds = (platt_probs >= 0.5).astype(int)
    platt_acc = accuracy_score(y, platt_preds)
    platt_auroc = roc_auc_score(y, platt_probs)
    print(f"  Accuracy: {platt_acc:.4f}")
    print(f"  AUROC: {platt_auroc:.4f}")

    # Approach 4: Ensemble with original keeper
    print(f"\n[7] Approach 4: Ensemble with Original Keeper")
    import json
    with open(ORIGINAL_ENSEMBLE) as f:
        ensemble_config = json.load(f)

    # Load original ensemble probes
    original_probs = np.zeros(len(X))
    total_weight = 0
    for member in ensemble_config['members']:
        probe_dir = member['probe']
        probe_path = Path(probe_dir) / "probe.joblib"
        weight = member['weight']
        clf_orig = joblib.load(probe_path)
        original_probs += weight * clf_orig.predict_proba(X)[:, 1]
        total_weight += weight

    original_probs /= total_weight
    orig_preds = (original_probs >= 0.5).astype(int)
    orig_acc = accuracy_score(y, orig_preds)
    orig_auroc = roc_auc_score(y, original_probs)
    print(f"  Original keeper accuracy: {orig_acc:.4f}")
    print(f"  Original keeper AUROC: {orig_auroc:.4f}")

    # Combine with Hemg probe
    combined_probs = 0.5 * original_probs + 0.5 * probs_hemg
    comb_preds = (combined_probs >= 0.5).astype(int)
    comb_acc = accuracy_score(y, comb_preds)
    comb_auroc = roc_auc_score(y, combined_probs)
    print(f"  Combined (50/50) accuracy: {comb_acc:.4f}")
    print(f"  Combined (50/50) AUROC: {comb_auroc:.4f}")

    # Try different weights
    best_weight = 0.5
    best_comb_acc = comb_acc
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        test_probs = w * original_probs + (1-w) * probs_hemg
        test_preds = (test_probs >= 0.5).astype(int)
        test_acc = accuracy_score(y, test_preds)
        if test_acc > best_comb_acc:
            best_comb_acc = test_acc
            best_weight = w

    print(f"  Best weight for original: {best_weight:.2f}")
    print(f"  Best combined accuracy: {best_comb_acc:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*60}")
    print(f"Baseline (Hemg+DALLE3):           {base_acc:.4f}")
    print(f"Threshold optimization:           {opt_acc:.4f}")
    print(f"Temperature scaling:              {temp_acc:.4f}")
    print(f"Platt scaling:                    {platt_acc:.4f}")
    print(f"Original keeper:                  {orig_acc:.4f}")
    print(f"Combined (best weight):           {best_comb_acc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
