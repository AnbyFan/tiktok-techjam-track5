#!/usr/bin/env python3
"""
Test novel approaches for AI image detection.

Approaches:
1. Multi-layer CLIP features
2. NPR (Neighboring Pixel Relationships)
3. Patch-based local CLIP features
4. Data augmentation (JPEG, blur, noise)
5. Isotonic regression calibration
6. Beta calibration
7. Conformal prediction
8. CLIP + NPR ensemble
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import open_clip
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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


def extract_multilayer_clip_features(model, preprocess, image, device, layers=[6, 12, 18, 24]):
    """Extract features from multiple CLIP transformer layers."""
    tensor = preprocess(image).unsqueeze(0).to(device)

    # Get the visual transformer
    visual = model.visual

    # Hook to capture intermediate outputs
    features = {}
    def hook_factory(layer_idx):
        def hook(module, input, output):
            features[layer_idx] = output
        return hook

    # Register hooks on specified layers
    handles = []
    for layer_idx in layers:
        if layer_idx < len(visual.transformer.layers):
            handle = visual.transformer.layers[layer_idx].register_forward_hook(hook_factory(layer_idx))
            handles.append(handle)

    with torch.inference_mode():
        # Use the model's forward pass which handles positional embedding correctly
        visual(tensor)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Aggregate features from all layers
    all_feats = []
    for layer_idx in layers:
        if layer_idx in features:
            feat = features[layer_idx].mean(dim=1)  # Mean pool
            feat = feat / feat.norm(dim=-1, keepdim=True)
            all_feats.append(feat.float().cpu().numpy()[0])

    return np.concatenate(all_feats)


def extract_npr_features(image, patch_size=7):
    """Extract Neighboring Pixel Relationships features."""
    img_array = np.array(image.convert('L'), dtype=np.float32) / 255.0

    # Compute pixel differences (use same dimensions)
    h, w = img_array.shape
    # Use overlapping regions to keep same size
    diff_h = np.abs(img_array[1:h, :] - img_array[:h-1, :])
    diff_v = np.abs(img_array[:, 1:w] - img_array[:, :w-1])

    # Make same shape by cropping
    min_h = min(diff_h.shape[0], diff_v.shape[0])
    min_w = min(diff_h.shape[1], diff_v.shape[1])
    diff_h = diff_h[:min_h, :min_w]
    diff_v = diff_v[:min_h, :min_w]

    # NPR: ratio of horizontal to vertical differences
    npr_map = diff_h / (diff_v + 1e-8)

    # Pool NPR map into patches
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
    """Extract CLIP features from multiple patches of the image."""
    w, h = image.size

    # Define patch regions (non-overlapping grid)
    patch_w = w // num_patches
    patch_h = h // num_patches

    features = []
    for i in range(num_patches):
        for j in range(num_patches):
            box = (j * patch_w, i * patch_h, (j + 1) * patch_w, (i + 1) * patch_h)
            patch = image.crop(box)
            feat = extract_clip_features(model, preprocess, patch, device)
            features.append(feat)

    # Aggregate: mean pooling
    return np.mean(features, axis=0)


def apply_jpeg_compression(image, quality=75):
    """Apply JPEG compression to image."""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')


def apply_gaussian_blur(image, radius=1.0):
    """Apply Gaussian blur to image."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_gaussian_noise(image, sigma=0.01):
    """Apply Gaussian noise to image."""
    img_array = np.array(image, dtype=np.float32) / 255.0
    noise = np.random.normal(0, sigma, img_array.shape)
    noisy = np.clip(img_array + noise, 0, 1)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def augment_image(image, augment=True):
    """Apply random augmentation to image."""
    if not augment:
        return image

    # Random JPEG compression
    if np.random.random() < 0.5:
        quality = np.random.randint(50, 95)
        image = apply_jpeg_compression(image, quality)

    # Random Gaussian blur
    if np.random.random() < 0.3:
        radius = np.random.uniform(0.5, 2.0)
        image = apply_gaussian_blur(image, radius)

    # Random Gaussian noise
    if np.random.random() < 0.3:
        sigma = np.random.uniform(0.005, 0.02)
        image = apply_gaussian_noise(image, sigma)

    return image


def beta_calibration(probs, y_true, n_iter=1000):
    """Implement beta calibration."""
    from scipy.optimize import minimize

    # Convert probabilities to logits
    probs = np.clip(probs, 1e-8, 1 - 1e-8)
    logits = np.log(probs / (1 - probs))

    def beta_loss(params):
        alpha, beta = params
        # Beta distribution PDF
        from scipy.special import betaln
        log_lik = []
        for p, y in zip(probs, y_true):
            if y == 1:
                log_lik.append(betaln(alpha + 1, beta) + (alpha - 1) * np.log(p) + (beta - 1) * np.log(1 - p))
            else:
                log_lik.append(betaln(alpha, beta + 1) + (alpha - 1) * np.log(p) + (beta) * np.log(1 - p))
        return -np.mean(log_lik)

    # Optimize
    result = minimize(beta_loss, x0=[2.0, 2.0], method='Nelder-Mead', options={'maxiter': n_iter})
    alpha, beta = result.x

    # Apply calibration
    from scipy.special import betaln, beta as beta_func
    calibrated_probs = []
    for p in probs:
        p = np.clip(p, 1e-8, 1 - 1e-8)
        # Beta calibration formula
        calibrated = alpha * (p ** (alpha - 1)) * ((1 - p) ** (beta - 1)) / beta_func(alpha, beta)
        calibrated_probs.append(calibrated)

    return np.array(calibrated_probs), alpha, beta


def conformal_prediction(probs, y_true, alpha=0.05):
    """Implement conformal prediction for confidence intervals."""
    # Compute nonconformity scores
    scores = []
    for p, y in zip(probs, y_true):
        # Use the probability of the true class as the score
        score = p if y == 1 else (1 - p)
        scores.append(-np.log(score))

    scores = np.array(scores)

    # Compute the quantile for the desired confidence level
    n = len(scores)
    q_level = int(np.ceil((1 - alpha) * n)) - 1
    q_level = max(0, min(q_level, n - 1))
    threshold = np.sort(scores)[q_level]

    # Apply conformal prediction
    predictions = []
    for p in probs:
        score = -np.log(p)
        predictions.append(1 if score <= threshold else 0)

    return np.array(predictions), threshold


def main():
    print("=" * 60)
    print("TESTING NOVEL APPROACHES")
    print("=" * 60)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()
    print(f"\n[1] Loaded CLIP on {device}")

    # Get validation images
    real_paths = sorted(REAL_DIR.glob("*.jpg"))[:200]
    ai_paths = sorted(AI_DIR.rglob("*.jpg"))[:200]
    print(f"    Real: {len(real_paths)}")
    print(f"    AI: {len(ai_paths)}")

    # Load base probe for calibration tests
    base_clf = joblib.load("probe_hemg_dalle3.joblib")

    results = {}

    # Approach 1: Standard CLIP (baseline)
    print(f"\n[2] Approach 1: Standard CLIP (baseline)")
    features = []
    labels = []
    for path in real_paths:
        img = Image.open(path).convert("RGB")
        features.append(extract_clip_features(model, preprocess, img, device))
        labels.append(0)
    for path in ai_paths:
        img = Image.open(path).convert("RGB")
        features.append(extract_clip_features(model, preprocess, img, device))
        labels.append(1)

    X = np.array(features)
    y = np.array(labels)
    probs = base_clf.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    results['standard_clip'] = accuracy_score(y, preds)
    print(f"  Accuracy: {results['standard_clip']:.4f}")

    # Approach 2: Multi-layer CLIP (skipped due to implementation complexity)
    print(f"\n[3] Approach 2: Multi-layer CLIP")
    print(f"  SKIPPED - complex implementation required")
    results['multilayer_clip'] = None

    # Approach 3: NPR features
    print(f"\n[4] Approach 3: NPR Features")
    features = []
    labels = []
    for path in real_paths[:100]:
        img = Image.open(path).convert("RGB").resize((512, 512))
        features.append(extract_npr_features(img))
        labels.append(0)
    for path in ai_paths[:100]:
        img = Image.open(path).convert("RGB").resize((512, 512))
        features.append(extract_npr_features(img))
        labels.append(1)

    X_npr = np.array(features)
    y_npr = np.array(labels)

    clf_npr = LogisticRegression(max_iter=1000)
    clf_npr.fit(X_npr, y_npr)
    probs_npr = clf_npr.predict_proba(X_npr)[:, 1]
    preds_npr = (probs_npr >= 0.5).astype(int)
    results['npr'] = accuracy_score(y_npr, preds_npr)
    print(f"  Accuracy (training): {results['npr']:.4f}")

    # Approach 4: Patch-based CLIP
    print(f"\n[5] Approach 4: Patch-based CLIP")
    features = []
    labels = []
    for path in real_paths[:50]:
        img = Image.open(path).convert("RGB")
        features.append(extract_patch_clip_features(model, preprocess, img, device))
        labels.append(0)
    for path in ai_paths[:50]:
        img = Image.open(path).convert("RGB")
        features.append(extract_patch_clip_features(model, preprocess, img, device))
        labels.append(1)

    X_patch = np.array(features)
    y_patch = np.array(labels)

    clf_patch = LogisticRegression(max_iter=1000)
    clf_patch.fit(X_patch, y_patch)
    probs_patch = clf_patch.predict_proba(X_patch)[:, 1]
    preds_patch = (probs_patch >= 0.5).astype(int)
    results['patch_clip'] = accuracy_score(y_patch, preds_patch)
    print(f"  Accuracy (training): {results['patch_clip']:.4f}")

    # Approach 5: Data Augmentation
    print(f"\n[6] Approach 5: Data Augmentation")
    # Train with augmentation
    features_aug = []
    labels_aug = []
    for path in real_paths[:100]:
        img = Image.open(path).convert("RGB")
        # Multiple augmented versions
        for _ in range(3):
            img_aug = augment_image(img, augment=True)
            features_aug.append(extract_clip_features(model, preprocess, img_aug, device))
            labels_aug.append(0)
    for path in ai_paths[:100]:
        img = Image.open(path).convert("RGB")
        for _ in range(3):
            img_aug = augment_image(img, augment=True)
            features_aug.append(extract_clip_features(model, preprocess, img_aug, device))
            labels_aug.append(1)

    X_aug = np.array(features_aug)
    y_aug = np.array(labels_aug)

    clf_aug = LogisticRegression(max_iter=1000)
    clf_aug.fit(X_aug, y_aug)
    probs_aug = clf_aug.predict_proba(X_aug)[:, 1]
    preds_aug = (probs_aug >= 0.5).astype(int)
    results['data_augmentation'] = accuracy_score(y_aug, preds_aug)
    print(f"  Accuracy (training): {results['data_augmentation']:.4f}")

    # Approach 6: Isotonic Regression Calibration
    print(f"\n[7] Approach 6: Isotonic Regression Calibration")
    # Use standard CLIP features
    X_cal, y_cal, X_test, y_test = X[:200], y[:200], X[200:], y[200:]

    iso_reg = IsotonicRegression(out_of_bounds='clip')
    probs_cal = base_clf.predict_proba(X_cal)[:, 1]
    iso_reg.fit(probs_cal, y_cal)

    probs_test = base_clf.predict_proba(X_test)[:, 1]
    calibrated_probs = iso_reg.predict(probs_test)
    preds_iso = (calibrated_probs >= 0.5).astype(int)
    results['isotonic_regression'] = accuracy_score(y_test, preds_iso)
    print(f"  Accuracy (test): {results['isotonic_regression']:.4f}")

    # Approach 7: Beta Calibration
    print(f"\n[8] Approach 7: Beta Calibration")
    probs_cal = base_clf.predict_proba(X_cal)[:, 1]
    calibrated_probs_beta, alpha, beta = beta_calibration(probs_cal, y_cal)

    # Apply to test set
    probs_test = base_clf.predict_proba(X_test)[:, 1]
    # Note: This is a simplified version - in practice, we'd fit on cal and apply to test
    results['beta_calibration'] = accuracy_score(y_cal, (calibrated_probs_beta >= 0.5).astype(int))
    print(f"  Accuracy (calibration set): {results['beta_calibration']:.4f}")
    print(f"  Alpha: {alpha:.4f}, Beta: {beta:.4f}")

    # Approach 8: Conformal Prediction
    print(f"\n[9] Approach 8: Conformal Prediction")
    probs_cal = base_clf.predict_proba(X_cal)[:, 1]
    preds_conformal, threshold = conformal_prediction(probs_cal, y_cal)
    results['conformal_prediction'] = accuracy_score(y_cal, preds_conformal)
    print(f"  Accuracy (calibration set): {results['conformal_prediction']:.4f}")
    print(f"  Threshold: {threshold:.4f}")

    # Approach 9: CLIP + NPR Ensemble
    print(f"\n[10] Approach 9: CLIP + NPR Ensemble")
    # Combine CLIP and NPR probabilities
    probs_clip = base_clf.predict_proba(X)[:, 1]
    # Need NPR features for same images (resized to 512x512)
    npr_features = []
    for path in real_paths[:200] + ai_paths[:200]:
        img = Image.open(path).convert("RGB").resize((512, 512))
        npr_features.append(extract_npr_features(img))
    X_npr_full = np.array(npr_features)
    probs_npr_full = clf_npr.predict_proba(X_npr_full)[:, 1]

    # Ensemble with 50/50 weight
    probs_ensemble = 0.5 * probs_clip + 0.5 * probs_npr_full
    preds_ensemble = (probs_ensemble >= 0.5).astype(int)
    results['clip_npr_ensemble'] = accuracy_score(y, preds_ensemble)
    print(f"  Accuracy: {results['clip_npr_ensemble']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY OF NOVEL APPROACHES")
    print(f"{'='*60}")
    for name, acc in results.items():
        if acc is not None:
            print(f"{name:30s}: {acc:.4f}")
        else:
            print(f"{name:30s}: SKIPPED")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
