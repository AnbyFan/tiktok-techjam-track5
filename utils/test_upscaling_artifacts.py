#!/usr/bin/env python3
"""
Test different upscaling methods to isolate what causes false positives
when real images are upscaled to 1024x1024.

Hypothesis: Different interpolation kernels create different artifacts
that the model may be detecting as "AI-like".
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
from ensemble_core import EnsembleScorer

REAL_DIR = Path("data/val/real")


def test_resampling_filters(scorer, real_paths, target_size=(1024, 1024)):
    """Test different resampling filters on upscaled real images."""
    filters = {
        "LANCZOS": Image.Resampling.LANCZOS,
        "BICUBIC": Image.Resampling.BICUBIC,
        "BILINEAR": Image.Resampling.BILINEAR,
        "NEAREST": Image.Resampling.NEAREST,
    }

    print("\n" + "=" * 60)
    print("RESAMPLING FILTER TEST")
    print("=" * 60)
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    print(f"Testing {len(real_paths)} real images\n")

    results = {}

    for filter_name, filter_method in filters.items():
        correct = 0
        total = 0
        false_positives = []

        for p in real_paths:
            img = Image.open(p).convert("RGB")
            img_upscaled = img.resize(target_size, filter_method)
            tensor = scorer.preprocess(img_upscaled)
            prob = scorer.score_tensors(torch.stack([tensor]))[0]
            is_ai = prob >= scorer.threshold

            total += 1
            if not is_ai:  # Correctly classified as real
                correct += 1
            else:
                false_positives.append((p.name, prob))

        acc = correct / total
        results[filter_name] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "false_positives": false_positives
        }

        fp_names = [fp[0] for fp in false_positives[:3]]
        print(f"  {filter_name:10s}: {acc:.2%} ({correct}/{total})")
        if false_positives:
            print(f"             FP examples: {', '.join(fp_names)}")

    return results


def test_upscale_with_downscale(scorer, real_paths):
    """Test if upscaling then downscaling (round-trip) changes results."""
    print("\n" + "=" * 60)
    print("UPSCALE-THEN-DOWNSCALE TEST")
    print("=" * 60)
    print("Testing if round-trip resizing causes artifacts\n")

    # Original
    correct_original = 0
    total = len(real_paths)

    for p in real_paths:
        img = Image.open(p).convert("RGB")
        tensor = scorer.preprocess(img)
        prob = scorer.score_tensors(torch.stack([tensor]))[0]
        if prob < scorer.threshold:
            correct_original += 1

    acc_original = correct_original / total
    print(f"  Original:        {acc_original:.2%} ({correct_original}/{total})")

    # Upscale to 1024, then downscale to original size
    correct_roundtrip = 0
    for p in real_paths:
        img = Image.open(p).convert("RGB")
        orig_size = img.size
        img_up = img.resize((1024, 1024), Image.Resampling.LANCZOS)
        img_rt = img_up.resize(orig_size, Image.Resampling.LANCZOS)
        tensor = scorer.preprocess(img_rt)
        prob = scorer.score_tensors(torch.stack([tensor]))[0]
        if prob < scorer.threshold:
            correct_roundtrip += 1

    acc_roundtrip = correct_roundtrip / total
    print(f"  Round-trip:      {acc_roundtrip:.2%} ({correct_roundtrip}/{total})")

    # Upscale to 2048, then downscale to 1024
    correct_2048 = 0
    for p in real_paths:
        img = Image.open(p).convert("RGB")
        img_up = img.resize((2048, 2048), Image.Resampling.LANCZOS)
        img_1024 = img_up.resize((1024, 1024), Image.Resampling.LANCZOS)
        tensor = scorer.preprocess(img_1024)
        prob = scorer.score_tensors(torch.stack([tensor]))[0]
        if prob < scorer.threshold:
            correct_2048 += 1

    acc_2048 = correct_2048 / total
    print(f"  2048→1024:       {acc_2048:.2%} ({correct_2048}/{total})")


def test_jpeg_after_upscale(scorer, real_paths):
    """Test if JPEG compression after upscaling masks the artifacts."""
    print("\n" + "=" * 60)
    print("JPEG COMPRESSION AFTER UPSCALE TEST")
    print("=" * 60)
    print("Testing if JPEG re-encoding masks upscaling artifacts\n")

    import io

    qualities = [30, 50, 70, 90]
    results = {}

    for q in qualities:
        correct = 0
        total = 0

        for p in real_paths:
            img = Image.open(p).convert("RGB")
            img_up = img.resize((1024, 1024), Image.Resampling.LANCZOS)

            # Save as JPEG with specified quality, then reload
            buf = io.BytesIO()
            img_up.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img_jpeg = Image.open(buf).convert("RGB")

            tensor = scorer.preprocess(img_jpeg)
            prob = scorer.score_tensors(torch.stack([tensor]))[0]
            total += 1
            if prob < scorer.threshold:
                correct += 1

        acc = correct / total
        results[q] = acc
        print(f"  JPEG q={q:2d}: {acc:.2%} ({correct}/{total})")

    return results


def test_native_high_res_reals(scorer):
    """Test if we have any native high-res real images in our dataset."""
    print("\n" + "=" * 60)
    print("NATIVE HIGH-RES REAL IMAGES TEST")
    print("=" * 60)

    # Find real images that are already >= 1000px
    high_res_reals = []
    for p in REAL_DIR.glob("*.jpg"):
        with Image.open(p) as img:
            if min(img.size) >= 1000:
                high_res_reals.append(p)

    print(f"\n  Found {len(high_res_reals)} real images >= 1000px")

    if high_res_reals:
        correct = 0
        total = 0
        for p in high_res_reals[:20]:
            img = Image.open(p).convert("RGB")
            tensor = scorer.preprocess(img)
            prob = scorer.score_tensors(torch.stack([tensor]))[0]
            total += 1
            if prob < scorer.threshold:
                correct += 1

        acc = correct / total
        print(f"  Accuracy on native high-res: {acc:.2%} ({correct}/{total})")


def main():
    print("=" * 60)
    print("UPSCALING ARTIFACT ANALYSIS")
    print("=" * 60)

    # Load ensemble
    print("\n[1] Loading ensemble...")
    scorer = EnsembleScorer("ensemble_config.json")
    print(f"    Loaded {len(scorer.probes)} probes")

    # Get real image paths
    real_paths = sorted(REAL_DIR.glob("*.jpg"))[:50]
    print(f"    Using {len(real_paths)} real images")

    # Run tests
    results_filters = test_resampling_filters(scorer, real_paths)
    test_upscale_with_downscale(scorer, real_paths)
    results_jpeg = test_jpeg_after_upscale(scorer, real_paths)
    test_native_high_res_reals(scorer)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n  Resampling filters (lower = more false positives):")
    for name, r in sorted(results_filters.items(), key=lambda x: x[1]["accuracy"]):
        print(f"    {name:10s}: {r['accuracy']:.2%}")

    print("\n  JPEG quality (higher quality should preserve artifacts):")
    for q, acc in sorted(results_jpeg.items()):
        print(f"    q={q:2d}:     {acc:.2%}")


if __name__ == "__main__":
    main()
