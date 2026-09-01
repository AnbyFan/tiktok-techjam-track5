#!/usr/bin/env python3
"""
Test frequency domain (FFT) features for AI image detection.

AI-generated images often have different frequency characteristics
than real photos. Adding FFT-based features could help detect these
subtle differences.

Features to extract:
- Radial power spectrum (average power by frequency radius)
- High-frequency energy ratio
- Spectral entropy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
from scipy.fft import fft2, fftshift
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

REAL_DIR = Path("data/val/real")
AI_DIR = Path("data/wildfake/Images/Diffusion_based/DALLE/DALLE/Advanced/DALLE3")


def extract_fft_features(img, size=224):
    """Extract frequency domain features from an image."""
    # Convert to grayscale and resize
    img_gray = img.convert("L").resize((size, size))
    arr = np.array(img_gray).astype(np.float32) / 255.0

    # Compute 2D FFT
    f_transform = fft2(arr)
    f_shifted = fftshift(f_transform)

    # Power spectrum
    power_spectrum = np.abs(f_shifted) ** 2

    # Radial power spectrum (average by radius from center)
    h, w = power_spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)

    # Bin by radius
    max_radius = min(cy, cx)
    radial_profile = np.zeros(max_radius)
    for r in range(max_radius):
        mask = radius == r
        if np.any(mask):
            radial_profile[r] = np.mean(power_spectrum[mask])

    # Normalize
    radial_profile = radial_profile / (np.sum(radial_profile) + 1e-10)

    # Features
    features = {
        # Radial profile (downsampled to 32 bins)
        "radial_profile": np.interp(
            np.linspace(0, 1, 32),
            np.linspace(0, 1, len(radial_profile)),
            radial_profile
        ),

        # High-frequency energy (outer 25% of spectrum)
        "hf_energy": np.sum(power_spectrum[max_radius//4:, max_radius//4:]) /
                    (np.sum(power_spectrum) + 1e-10),

        # Low-frequency energy (center 25%)
        "lf_energy": np.sum(power_spectrum[:max_radius//4, :max_radius//4]) /
                    (np.sum(power_spectrum) + 1e-10),

        # Spectral entropy
        "spectral_entropy": -np.sum(
            (power_spectrum / (np.sum(power_spectrum) + 1e-10)) *
            np.log(power_spectrum / (np.sum(power_spectrum) + 1e-10) + 1e-10)
        ),

        # Peak frequency location
        "peak_freq_radius": np.argmax(radial_profile) / max_radius,

        # Frequency skewness
        "freq_skewness": np.mean((np.arange(max_radius) - np.mean(np.arange(max_radius))) ** 3) /
                        (np.std(np.arange(max_radius)) ** 3 + 1e-10),
    }

    # Flatten to vector
    feat_vec = np.concatenate([
        features["radial_profile"],
        [features["hf_energy"], features["lf_energy"],
         features["spectral_entropy"], features["peak_freq_radius"],
         features["freq_skewness"]]
    ])

    return feat_vec


def main():
    print("=" * 60)
    print("FREQUENCY DOMAIN FEATURES TEST")
    print("=" * 60)

    # Get test images
    real_paths = sorted(REAL_DIR.glob("*.jpg"))[:100]
    ai_paths = sorted(AI_DIR.rglob("*.jpg"))[:100]
    print(f"\n[1] Using {len(real_paths)} real + {len(ai_paths)} AI images")

    # Extract features
    print("[2] Extracting FFT features...")
    real_feats = []
    ai_feats = []

    for p in real_paths:
        img = Image.open(p)
        feat = extract_fft_features(img)
        real_feats.append(feat)

    for p in ai_paths:
        img = Image.open(p)
        feat = extract_fft_features(img)
        ai_feats.append(feat)

    X = np.vstack([real_feats, ai_feats])
    y = np.array([0]*len(real_feats) + [1]*len(ai_feats))

    print(f"    Feature dimension: {X.shape[1]}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression
    print("[3] Training logistic regression...")
    clf = LogisticRegression(max_iter=1000, class_weight={0: 1, 1: 4.5})
    clf.fit(X_scaled, y)

    # Evaluate
    probs = clf.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)

    real_acc = accuracy_score(y[:len(real_feats)], preds[:len(real_feats)])
    ai_acc = accuracy_score(y[len(real_feats):], preds[len(real_feats):])

    print(f"\n[4] Results:")
    print(f"    Accuracy: {acc:.2%}")
    print(f"    Real accuracy: {real_acc:.2%}")
    print(f"    AI accuracy: {ai_acc:.2%}")

    # Compare with CLIP baseline
    print(f"\n[5] Comparison with CLIP baseline:")
    print(f"    CLIP (224x224): ~91% accuracy (from previous test)")
    print(f"    FFT features:   {acc:.2%} accuracy")

    if acc > 0.91:
        print(f"\n  ✅ FFT features outperform CLIP baseline!")
    else:
        print(f"\n  ❌ FFT features underperform CLIP baseline")
        print(f"  Consider combining CLIP + FFT features")


if __name__ == "__main__":
    main()
