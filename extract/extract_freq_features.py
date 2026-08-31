#!/usr/bin/env python3
"""
Extract frequency-domain features (radial power spectrum + phase coherence)
for images in a directory. These features are more stable under JPEG compression
and can be concatenated with CLIP features for improved robustness.

Features extracted per image:
- 64 radial power spectrum bins (log-scale)
- 8 phase coherence stats (mean/std of phase gradient magnitude)
Total: 72 additional dimensions

Usage:
    python extract_freq_features.py --input-dir data/val/real --out features_freq/real_val
    python extract_freq_features.py --input-dir data/wildfake/Images/Diffusion_based/DALLE/DALLE/Advanced/DALLE3 --out features_freq/ai_dalle3
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import fft2, fftshift
from tqdm import tqdm

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def compute_radial_power_spectrum(gray, n_bins=64):
    """Compute radial power spectrum with log-spaced bins."""
    # 2D FFT
    f = fft2(gray)
    f_shifted = fftshift(f)
    
    # Power spectrum
    power = np.abs(f_shifted) ** 2
    
    # Create radial bins
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = min(cy, cx)
    
    # Log-spaced bin edges
    bin_edges = np.logspace(0, np.log10(r_max), n_bins + 1)
    
    # Average power in each radial bin
    radial_power = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if mask.sum() > 0:
            radial_power[i] = power[mask].mean()
    
    # Log transform for better distribution
    radial_power = np.log1p(radial_power)
    
    return radial_power


def compute_phase_coherence(gray, n_stats=8):
    """Compute phase coherence statistics."""
    f = fft2(gray)
    f_shifted = fftshift(f)
    phase = np.angle(f_shifted)
    
    # Phase gradient magnitude
    dy, dx = np.gradient(phase)
    grad_mag = np.sqrt(dx ** 2 + dy ** 2)
    
    # Statistics
    stats = np.array([
        grad_mag.mean(),
        grad_mag.std(),
        np.percentile(grad_mag, 10),
        np.percentile(grad_mag, 25),
        np.percentile(grad_mag, 50),
        np.percentile(grad_mag, 75),
        np.percentile(grad_mag, 90),
        (grad_mag > grad_mag.mean()).mean(),  # fraction above mean
    ])
    
    return stats


def extract_features_for_image(img, n_freq_bins=64, n_phase_stats=8):
    """Extract frequency features from a PIL image."""
    # Convert to grayscale, resize to 256x256 for consistency
    gray = img.convert("L").resize((256, 256), Image.BICUBIC)
    arr = np.asarray(gray).astype(np.float32) / 255.0
    
    # Radial power spectrum
    radial = compute_radial_power_spectrum(arr, n_bins=n_freq_bins)
    
    # Phase coherence
    phase_stats = compute_phase_coherence(arr, n_stats=n_phase_stats)
    
    # Concatenate
    return np.concatenate([radial, phase_stats])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-freq-bins", type=int, default=64)
    p.add_argument("--n-phase-stats", type=int, default=8)
    p.add_argument("--max-images", type=int, default=None)
    args = p.parse_args()
    
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    
    # List images
    paths = sorted(p for p in Path(args.input_dir).rglob("*") if p.suffix.lower() in EXTS)
    if args.max_images:
        paths = paths[:args.max_images]
    
    print(f"[init] {len(paths)} images from {args.input_dir}")
    print(f"[init] Features: {args.n_freq_bins} freq bins + {args.n_phase_stats} phase stats = {args.n_freq_bins + args.n_phase_stats} dims")
    
    feats = []
    metas = []
    
    for path in tqdm(paths, desc="extracting", unit="img"):
        try:
            img = Image.open(path).convert("RGB")
            feat = extract_features_for_image(img, args.n_freq_bins, args.n_phase_stats)
            feats.append(feat)
            metas.append(str(path.relative_to(args.input_dir)))
        except Exception as e:
            print(f"[warn] failed {path}: {e}")
            continue
    
    if not feats:
        print("[error] no features extracted")
        return
    
    feats = np.stack(feats).astype(np.float32)
    np.save(out / "features_00000.npy", feats)
    
    # Save metadata
    with (out / "meta_00000.csv").open("w") as f:
        f.write("path\n")
        for m in metas:
            f.write(f"{m}\n")
    
    manifest = {
        "shards": ["features_00000.npy"],
        "n_images": len(feats),
        "feature_dim": args.n_freq_bins + args.n_phase_stats,
        "input_dir": args.input_dir,
        "n_freq_bins": args.n_freq_bins,
        "n_phase_stats": args.n_phase_stats,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    
    print(f"[done] {len(feats)} images, {feats.shape[1]} dims")
    print(f"[done] saved to {out}/")


if __name__ == "__main__":
    main()
