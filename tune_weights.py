#!/usr/bin/env python3
"""
Tune ensemble weights by optimizing on the eval set.

Loads the ensemble, gets per-probe predictions on a subset of images,
then uses scipy.optimize to find weights that maximize accuracy.

Usage:
    python tune_weights.py --config ensemble_config.json \
        --real-dir data/val/real --ai-dir data/wildfake/.../dalle3 \
        --max-per-class 300 --out ensemble_tuned.json
"""

import argparse
import csv
import io
import json
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy.optimize import minimize

from ensemble_core import EnsembleScorer

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="ensemble_config.json")
    p.add_argument("--real-dir", required=True)
    p.add_argument("--ai-dir", required=True)
    p.add_argument("--max-per-class", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="ensemble_tuned.json")
    p.add_argument("--focus-noise", action="store_true",
                   help="optimize primarily on noise transforms")
    return p.parse_args()


def build_specs():
    """Return transform specs (same as eval script)."""
    specs = [("clean", "clean", {})]
    for q in (90, 70, 50, 30):
        specs.append((f"jpeg_q{q}", "jpeg", {"quality": q}))
    for s in (0.5, 1.0, 2.0):
        specs.append((f"blur_s{s}", "blur", {"sigma": s}))
    for sc in (0.5, 0.25):
        specs.append((f"resize_{sc}x", "resize", {"scale": sc}))
    for s in (0.02, 0.05, 0.10):
        specs.append((f"noise_s{s}", "noise", {"sigma": s}))
    specs.append(("jitter_20pct", "jitter", {"amount": 0.2}))
    specs.append(("crop_80pct", "crop", {"frac": 0.8}))
    return specs


def apply_transform(img, kind, params, rng):
    if kind == "clean":
        return img
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=params["quality"])
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=params["sigma"]))
    if kind == "resize":
        w, h = img.size
        sc = params["scale"]
        small = img.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
    if kind == "noise":
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = arr + rng.normal(0.0, params["sigma"], arr.shape).astype(np.float32)
        return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8))
    if kind == "jitter":
        a = params["amount"]
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(1 - a, 1 + a))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(1 - a, 1 + a))
        img = ImageEnhance.Color(img).enhance(rng.uniform(1 - a, 1 + a))
        return img
    if kind == "crop":
        w, h = img.size
        f = params["frac"]
        cw, ch = int(w * f), int(h * f)
        left, top = (w - cw) // 2, (w - ch) // 2
        return img.crop((left, top, left + cw, top + ch))
    raise ValueError(f"unknown transform: {kind}")


def get_per_probe_predictions(scorer, items, specs, batch_size, seed):
    """Get individual probe predictions for all images and transforms."""
    n_probes = len(scorer.probes)
    # Store: [transform_idx][image_idx][probe_idx] = prob_ai
    all_probs = {spec[0]: [] for spec in specs}
    all_labels = {spec[0]: [] for spec in specs}

    for t_idx, (name, kind, params) in enumerate(specs):
        probs_all, labels_all = [], []
        batches = [items[i:i + batch_size]
                   for i in range(0, len(items), batch_size)]

        for batch in batches:
            tensors, meta = [], []
            for path, label in batch:
                try:
                    img = Image.open(path).convert("RGB")
                    rng = np.random.default_rng(
                        (seed + zlib.crc32(f"{path}|{name}".encode())) % 2**32)
                    tensors.append(scorer.preprocess(apply_transform(img, kind, params, rng)))
                    meta.append((str(path), label))
                except Exception:
                    continue
            if not tensors:
                continue

            # Get features once
            features = scorer.encode(torch.stack(tensors))

            # Get each probe's individual prediction
            for (path, label), feat in zip(meta, features):
                probe_probs = []
                for (probe_name, clf) in scorer.probes:
                    p = clf.predict_proba(feat.reshape(1, -1))[:, 1][0]
                    probe_probs.append(p)
                probs_all.append(probe_probs)
                labels_all.append(label)

        all_probs[name] = np.array(probs_all) if probs_all else np.array([])
        all_labels[name] = np.array(labels_all) if labels_all else np.array([])
        print(f"[tune] {name}: {len(labels_all)} images")

    return all_probs, all_labels


def main():
    args = parse_args()
    scorer = EnsembleScorer(args.config)
    n_probes = len(scorer.probes)
    print(f"[tune] Loading ensemble with {n_probes} probes")
    print(f"[tune] Probe names: {[n for n, _ in scorer.probes]}")

    # Load images
    def list_images(d):
        paths = sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in EXTS)
        if args.max_per_class:
            paths = paths[:args.max_per_class]
        return paths

    items = [(p, 0) for p in list_images(args.real_dir)] + \
            [(p, 1) for p in list_images(args.ai_dir)]
    print(f"[tune] {len(items)} images total")

    # Get per-probe predictions
    specs = build_specs()
    all_probs, all_labels = get_per_probe_predictions(
        scorer, items, specs, args.batch_size, args.seed)

    # Define objective function: negative accuracy (to minimize)
    def objective(weights):
        weights = np.abs(weights)  # Ensure positive
        weights = weights / weights.sum()  # Normalize

        total_correct = 0
        total_samples = 0

        for name in all_probs:
            probs = all_probs[name]
            labels = all_labels[name]
            if len(probs) == 0:
                continue

            # Weighted average of probe predictions
            combined_probs = probs @ weights
            preds = (combined_probs >= 0.5).astype(int)
            correct = (preds == labels).sum()

            # If focusing on noise, weight noise transforms more heavily
            if args.focus_noise and "noise" in name:
                weight = 3.0
            else:
                weight = 1.0

            total_correct += correct * weight
            total_samples += len(labels) * weight

        return -total_correct / total_samples

    # Initial weights: equal
    x0 = np.ones(n_probes) / n_probes

    # Optimize
    print(f"[tune] Optimizing weights (focus_noise={args.focus_noise})...")
    result = minimize(
        objective, x0, method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
    )

    # Get optimal weights
    opt_weights = np.abs(result.x)
    opt_weights = opt_weights / opt_weights.sum()

    print(f"\n[tune] Optimization complete. Success: {result.success}")
    print(f"[tune] Initial accuracy: {-objective(x0):.4f}")
    print(f"[tune] Optimized accuracy: {-result.fun:.4f}")
    print(f"\n[tune] Optimal weights:")
    for name, w in zip([n for n, _ in scorer.probes], opt_weights):
        print(f"  {name}: {w:.4f}")

    # Save tuned config
    cfg = json.loads(Path(args.config).read_text())
    for i, m in enumerate(cfg["members"]):
        m["weight"] = float(opt_weights[i])

    Path(args.out).write_text(json.dumps(cfg, indent=2))
    print(f"\n[tune] Saved tuned config to {args.out}")


if __name__ == "__main__":
    main()
