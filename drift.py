#!/usr/bin/env python3
"""
Feature-drift augmentation for Track 5 (RIGID-style).

Core idea: real and AI images behave differently under tiny perturbations.
AI images show LARGER CLIP-embedding drift under small noise (their features
sit in sharper, less stable regions of the embedding space). We expose this as
two extra scalar features so a probe can lean on it.

For a clean preprocessed tensor x:
    f_clean = normalize(CLIP(x))
    for i in range(K):
        f_i   = normalize(CLIP(x + eps_i))     # eps_i ~ N(0, sigma^2)
        d_i   = 1 - cos(f_clean, f_i)
    drift_mean = mean(d_i)
    drift_std  = std(d_i)
    feature    = [f_clean, drift_mean, drift_std]   # 768 + 2 = 770-d

The perturbation is added in CLIP's preprocessed tensor space (post-normalize),
which is where the backbone actually operates.
"""

import numpy as np
import torch


def add_drift_features(model, f_clean, tensors, k, sigma, device, rng=None):
    """
    Compute drift features and append to clean features.

    Args:
        model: CLIP model (in eval mode).
        f_clean: (B, 768) L2-normalized clean features (numpy float32).
        tensors: (B, 3, H, W) preprocessed clean tensors (CPU).
        k: number of random perturbations per image.
        sigma: std of the Gaussian perturbation (preprocessed-tensor space).
        device: torch device string.
        rng: numpy Generator for reproducible noise (optional).

    Returns:
        (B, 770) numpy float32 array = [f_clean, drift_mean, drift_std].
    """
    B = f_clean.shape[0]
    drifts = np.empty((B, k), dtype=np.float32)

    with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.float16, enabled=(device == "cuda")):
        x = tensors.to(device, non_blocking=True)
        fc = torch.as_tensor(f_clean, dtype=torch.float32).to(device, non_blocking=True)
        fc = fc / fc.norm(dim=-1, keepdim=True)
        for i in range(k):
            if rng is not None:
                noise_np = rng.normal(0.0, sigma, tensors.shape).astype(np.float32)
                noise = torch.from_numpy(noise_np).to(device, non_blocking=True)
            else:
                noise = torch.randn_like(x) * sigma
            f_noisy = model.encode_image(x + noise)
            f_noisy = f_noisy / f_noisy.norm(dim=-1, keepdim=True)
            cos_sim = (fc * f_noisy).sum(dim=-1)
            drifts[:, i] = (1.0 - cos_sim).float().cpu().numpy()

    drift_mean = drifts.mean(axis=1, keepdims=True)
    drift_std = drifts.std(axis=1, keepdims=True)
    return np.concatenate([f_clean, drift_mean, drift_std], axis=1)
