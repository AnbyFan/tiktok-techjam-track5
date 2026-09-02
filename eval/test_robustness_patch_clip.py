#!/usr/bin/env python3
"""
Test robustness of patch-based CLIP model on various transformations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from PIL import Image, ImageFilter
import open_clip
import joblib
from sklearn.metrics import accuracy_score
import io

REAL_DIR = Path("data/val/real")
AI_DIR = Path("data/wildfake/Images/Diffusion_based/DALLE/DALLE/Advanced/DALLE3")


def extract_clip_features(model, preprocess, image, device):
    """Extract standard CLIP features."""
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.float().cpu().numpy()[0]


def extract_patch_clip_features(model, preprocess, image, device, num_patches=4):
    """Extract CLIP features from multiple patches."""
    w, h = image.size
    patch_w = w // num_patches
    patch_h = h // num_patches
    features = []
    for i in range(num_patches):
        for j in range(num_patches):
            box = (j * patch_w, i * patch_h, (j + 1) * patch_w, (i + 1) * patch_h)
            patch = image.crop(box)
            feat = extract_clip_features(model, preprocess, patch, device)
            features.append(feat)
    return np.mean(features, axis=0)


def apply_jpeg_compression(image, quality):
    """Apply JPEG compression."""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')


def apply_gaussian_noise(image, sigma):
    """Apply Gaussian noise."""
    img_array = np.array(image, dtype=np.float32) / 255.0
    noise = np.random.normal(0, sigma, img_array.shape)
    noisy = np.clip(img_array + noise, 0, 1)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def apply_gaussian_blur(image, radius):
    """Apply Gaussian blur."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def resize_image(image, size):
    """Resize image."""
    return image.resize(size, Image.Resampling.LANCZOS)


def test_transformation(clf, model, preprocess, device, transform_func, transform_name, real_paths, ai_paths, n_samples=100):
    """Test model accuracy with a specific transformation."""
    features = []
    labels = []

    for path in real_paths[:n_samples]:
        img = Image.open(path).convert("RGB")
        img = transform_func(img)
        features.append(extract_patch_clip_features(model, preprocess, img, device))
        labels.append(0)

    for path in ai_paths[:n_samples]:
        img = Image.open(path).convert("RGB")
        img = transform_func(img)
        features.append(extract_patch_clip_features(model, preprocess, img, device))
        labels.append(1)

    X = np.array(features)
    y = np.array(labels)

    probs = clf.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)

    print(f"  {transform_name:30s}: {acc:.4f}")
    return acc


def main():
    print("=" * 60)
    print("ROBUSTNESS TESTING: PATCH-BASED CLIP MODEL")
    print("=" * 60)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()
    clf = joblib.load("probe_patch_clip.joblib")
    print(f"\n[1] Loaded model on {device}")

    # Get test images
    real_paths = sorted(REAL_DIR.glob("*.jpg"))
    ai_paths = sorted(AI_DIR.rglob("*.jpg"))
    print(f"    Real: {len(real_paths)}")
    print(f"    AI: {len(ai_paths)}")

    # Baseline (no transformation)
    print(f"\n[2] Baseline (no transformation)")
    test_transformation(clf, model, preprocess, device, lambda x: x, "Original", real_paths, ai_paths)

    # JPEG compression
    print(f"\n[3] JPEG Compression")
    for quality in [95, 85, 75, 65, 55, 45, 35, 25]:
        test_transformation(clf, model, preprocess, device,
                          lambda x, q=quality: apply_jpeg_compression(x, q),
                          f"JPEG Q={quality}", real_paths, ai_paths)

    # Gaussian noise
    print(f"\n[4] Gaussian Noise")
    for sigma in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
        test_transformation(clf, model, preprocess, device,
                          lambda x, s=sigma: apply_gaussian_noise(x, s),
                          f"Noise sigma={sigma}", real_paths, ai_paths)

    # Gaussian blur
    print(f"\n[5] Gaussian Blur")
    for radius in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        test_transformation(clf, model, preprocess, device,
                          lambda x, r=radius: apply_gaussian_blur(x, r),
                          f"Blur r={radius}", real_paths, ai_paths)

    # Resizing
    print(f"\n[6] Resizing")
    for size in [(512, 512), (448, 448), (384, 384), (320, 320), (256, 256)]:
        test_transformation(clf, model, preprocess, device,
                          lambda x, s=size: resize_image(x, s),
                          f"Resize {size[0]}x{size[1]}", real_paths, ai_paths)

    print(f"\n{'='*60}")
    print("ROBUSTNESS TESTING COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
