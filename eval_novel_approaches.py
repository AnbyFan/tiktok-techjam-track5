#!/usr/bin/env python3
"""
Proper evaluation of novel approaches with train/test split.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from PIL import Image, ImageFilter
import open_clip
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from io import BytesIO
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


def extract_npr_features(image, patch_size=7):
    """Extract Neighboring Pixel Relationships features."""
    img_array = np.array(image.convert('L'), dtype=np.float32) / 255.0
    h, w = img_array.shape
    diff_h = np.abs(img_array[1:h, :] - img_array[:h-1, :])
    diff_v = np.abs(img_array[:, 1:w] - img_array[:, :w-1])
    min_h = min(diff_h.shape[0], diff_v.shape[0])
    min_w = min(diff_h.shape[1], diff_v.shape[1])
    diff_h = diff_h[:min_h, :min_w]
    diff_v = diff_v[:min_h, :min_w]
    npr_map = diff_h / (diff_v + 1e-8)
    h, w = npr_map.shape
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    npr_features = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = npr_map[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            npr_features.extend([np.mean(patch), np.std(patch), np.max(patch)])
    return np.array(npr_features)


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


def apply_jpeg_compression(image, quality=75):
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')


def apply_gaussian_blur(image, radius=1.0):
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_gaussian_noise(image, sigma=0.01):
    img_array = np.array(image, dtype=np.float32) / 255.0
    noise = np.random.normal(0, sigma, img_array.shape)
    noisy = np.clip(img_array + noise, 0, 1)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def augment_image(image, augment=True):
    if not augment:
        return image
    if np.random.random() < 0.5:
        quality = np.random.randint(50, 95)
        image = apply_jpeg_compression(image, quality)
    if np.random.random() < 0.3:
        radius = np.random.uniform(0.5, 2.0)
        image = apply_gaussian_blur(image, radius)
    if np.random.random() < 0.3:
        sigma = np.random.uniform(0.005, 0.02)
        image = apply_gaussian_noise(image, sigma)
    return image


def main():
    print("=" * 60)
    print("PROPER EVALUATION OF NOVEL APPROACHES")
    print("=" * 60)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()
    print(f"\n[1] Loaded CLIP on {device}")

    # Get validation images
    real_paths = sorted(REAL_DIR.glob("*.jpg"))[:400]
    ai_paths = sorted(AI_DIR.rglob("*.jpg"))[:400]
    print(f"    Real: {len(real_paths)}")
    print(f"    AI: {len(ai_paths)}")

    # Split into train and test
    real_train, real_test = real_paths[:200], real_paths[200:]
    ai_train, ai_test = ai_paths[:200], ai_paths[200:]

    results = {}

    # Approach 1: Standard CLIP with Platt scaling (baseline)
    print(f"\n[2] Standard CLIP + Platt Scaling (baseline)")
    base_clf = joblib.load("probe_platt_calibrated.joblib")

    # Test set features
    test_features = []
    test_labels = []
    for path in real_test:
        img = Image.open(path).convert("RGB")
        test_features.append(extract_clip_features(model, preprocess, img, device))
        test_labels.append(0)
    for path in ai_test:
        img = Image.open(path).convert("RGB")
        test_features.append(extract_clip_features(model, preprocess, img, device))
        test_labels.append(1)

    X_test = np.array(test_features)
    y_test = np.array(test_labels)
    probs = base_clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    results['standard_clip_platt'] = accuracy_score(y_test, preds)
    print(f"  Test Accuracy: {results['standard_clip_platt']:.4f}")

    # Approach 2: NPR Features
    print(f"\n[3] NPR Features")
    # Train set
    train_features = []
    train_labels = []
    for path in real_train:
        img = Image.open(path).convert("RGB").resize((512, 512))
        train_features.append(extract_npr_features(img))
        train_labels.append(0)
    for path in ai_train:
        img = Image.open(path).convert("RGB").resize((512, 512))
        train_features.append(extract_npr_features(img))
        train_labels.append(1)

    X_npr_train = np.array(train_features)
    y_npr_train = np.array(train_labels)

    # Test set
    test_features = []
    test_labels = []
    for path in real_test:
        img = Image.open(path).convert("RGB").resize((512, 512))
        test_features.append(extract_npr_features(img))
        test_labels.append(0)
    for path in ai_test:
        img = Image.open(path).convert("RGB").resize((512, 512))
        test_features.append(extract_npr_features(img))
        test_labels.append(1)

    X_npr_test = np.array(test_features)
    y_npr_test = np.array(test_labels)

    clf_npr = LogisticRegression(max_iter=1000)
    clf_npr.fit(X_npr_train, y_npr_train)
    probs_npr = clf_npr.predict_proba(X_npr_test)[:, 1]
    preds_npr = (probs_npr >= 0.5).astype(int)
    results['npr'] = accuracy_score(y_npr_test, preds_npr)
    print(f"  Test Accuracy: {results['npr']:.4f}")

    # Approach 3: Patch-based CLIP
    print(f"\n[4] Patch-based CLIP")
    # Train set
    train_features = []
    train_labels = []
    for path in real_train:
        img = Image.open(path).convert("RGB")
        train_features.append(extract_patch_clip_features(model, preprocess, img, device))
        train_labels.append(0)
    for path in ai_train:
        img = Image.open(path).convert("RGB")
        train_features.append(extract_patch_clip_features(model, preprocess, img, device))
        train_labels.append(1)

    X_patch_train = np.array(train_features)
    y_patch_train = np.array(train_labels)

    # Test set
    test_features = []
    test_labels = []
    for path in real_test:
        img = Image.open(path).convert("RGB")
        test_features.append(extract_patch_clip_features(model, preprocess, img, device))
        test_labels.append(0)
    for path in ai_test:
        img = Image.open(path).convert("RGB")
        test_features.append(extract_patch_clip_features(model, preprocess, img, device))
        test_labels.append(1)

    X_patch_test = np.array(test_features)
    y_patch_test = np.array(test_labels)

    clf_patch = LogisticRegression(max_iter=1000)
    clf_patch.fit(X_patch_train, y_patch_train)
    probs_patch = clf_patch.predict_proba(X_patch_test)[:, 1]
    preds_patch = (probs_patch >= 0.5).astype(int)
    results['patch_clip'] = accuracy_score(y_patch_test, preds_patch)
    print(f"  Test Accuracy: {results['patch_clip']:.4f}")

    # Approach 4: Data Augmentation
    print(f"\n[5] Data Augmentation")
    # Train with augmentation
    train_features = []
    train_labels = []
    for path in real_train:
        img = Image.open(path).convert("RGB")
        for _ in range(3):
            img_aug = augment_image(img, augment=True)
            train_features.append(extract_clip_features(model, preprocess, img_aug, device))
            train_labels.append(0)
    for path in ai_train:
        img = Image.open(path).convert("RGB")
        for _ in range(3):
            img_aug = augment_image(img, augment=True)
            train_features.append(extract_clip_features(model, preprocess, img_aug, device))
            train_labels.append(1)

    X_aug_train = np.array(train_features)
    y_aug_train = np.array(train_labels)

    # Test without augmentation
    test_features = []
    test_labels = []
    for path in real_test:
        img = Image.open(path).convert("RGB")
        test_features.append(extract_clip_features(model, preprocess, img, device))
        test_labels.append(0)
    for path in ai_test:
        img = Image.open(path).convert("RGB")
        test_features.append(extract_clip_features(model, preprocess, img, device))
        test_labels.append(1)

    X_aug_test = np.array(test_features)
    y_aug_test = np.array(test_labels)

    clf_aug = LogisticRegression(max_iter=1000)
    clf_aug.fit(X_aug_train, y_aug_train)
    probs_aug = clf_aug.predict_proba(X_aug_test)[:, 1]
    preds_aug = (probs_aug >= 0.5).astype(int)
    results['data_augmentation'] = accuracy_score(y_aug_test, preds_aug)
    print(f"  Test Accuracy: {results['data_augmentation']:.4f}")

    # Approach 5: CLIP + NPR Ensemble
    print(f"\n[6] CLIP + NPR Ensemble")
    # Get CLIP probabilities (using base model)
    base_clf_orig = joblib.load("probe_hemg_dalle3.joblib")
    probs_clip_test = base_clf_orig.predict_proba(X_test)[:, 1]

    # Get NPR probabilities
    probs_npr_test = clf_npr.predict_proba(X_npr_test)[:, 1]

    # Ensemble
    probs_ensemble = 0.5 * probs_clip_test + 0.5 * probs_npr_test
    preds_ensemble = (probs_ensemble >= 0.5).astype(int)
    results['clip_npr_ensemble'] = accuracy_score(y_test, preds_ensemble)
    print(f"  Test Accuracy: {results['clip_npr_ensemble']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY OF NOVEL APPROACHES (Test Set)")
    print(f"{'='*60}")
    for name, acc in results.items():
        print(f"{name:30s}: {acc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
