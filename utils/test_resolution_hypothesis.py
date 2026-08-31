#!/usr/bin/env python3
"""
Test whether the model uses image resolution as a shortcut for AI detection.

Hypothesis: If we resize AI images down to match real image resolution,
the model should misclassify them more often (if resolution is a shortcut).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
from ensemble_core import EnsembleScorer

REAL_DIR = Path("data/val/real")
AI_DIR = Path("data/wildfake/Images/Diffusion_based/DALLE/DALLE/Advanced/DALLE3")


def get_image_stats(paths, max_samples=50):
    """Get resolution stats for a set of images."""
    sizes = []
    for p in paths[:max_samples]:
        with Image.open(p) as img:
            sizes.append(img.size)

    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]

    return {
        "count": len(sizes),
        "avg_width": sum(widths) / len(widths),
        "avg_height": sum(heights) / len(heights),
        "min_width": min(widths),
        "max_width": max(widths),
        "min_height": min(heights),
        "max_height": max(heights),
    }


def test_resolution_hypothesis():
    print("=" * 60)
    print("RESOLUTION HYPOTHESIS TEST")
    print("=" * 60)

    # Load ensemble
    print("\n[1] Loading ensemble...")
    scorer = EnsembleScorer("ensemble_config.json")
    print(f"    Loaded {len(scorer.probes)} probes")

    # Get image paths
    real_paths = sorted(REAL_DIR.glob("*.jpg"))[:50]
    ai_paths = sorted(AI_DIR.rglob("*.jpg"))[:50]

    # Print resolution stats
    print("\n[2] Resolution stats:")
    real_stats = get_image_stats(real_paths)
    ai_stats = get_image_stats(ai_paths)

    print(f"    Real images:")
    print(f"      Avg: {real_stats['avg_width']:.0f}x{real_stats['avg_height']:.0f}")
    print(f"      Range: {real_stats['min_width']}x{real_stats['min_height']} to {real_stats['max_width']}x{real_stats['max_height']}")

    print(f"    AI images:")
    print(f"      Avg: {ai_stats['avg_width']:.0f}x{ai_stats['avg_height']:.0f}")
    print(f"      Range: {ai_stats['min_width']}x{ai_stats['min_height']} to {ai_stats['max_width']}x{ai_stats['max_height']}")

    # Test 1: Original resolution
    print("\n[3] Testing at ORIGINAL resolution:")
    correct_original = 0
    total_original = 0

    # Test real images (should be classified as REAL)
    for p in real_paths:
        img = Image.open(p).convert("RGB")
        tensor = scorer.preprocess(img)
        prob = scorer.score_tensors(torch.stack([tensor]))[0]
        is_ai = prob >= scorer.threshold
        if not is_ai:  # Correctly classified as real
            correct_original += 1
        total_original += 1

    # Test AI images (should be classified as AI)
    for p in ai_paths:
        img = Image.open(p).convert("RGB")
        tensor = scorer.preprocess(img)
        prob = scorer.score_tensors(torch.stack([tensor]))[0]
        is_ai = prob >= scorer.threshold
        if is_ai:  # Correctly classified as AI
            correct_original += 1
        total_original += 1

    acc_original = correct_original / total_original
    print(f"    Accuracy: {correct_original}/{total_original} ({acc_original:.2%})")

    # Test 2: Resize AI images to match real resolution
    print("\n[4] Testing AI images RESIZED to 640x480 (matching real):")
    correct_resized = 0
    total_resized = 0
    target_size = (640, 480)

    for p in ai_paths:
        img = Image.open(p).convert("RGB")
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        tensor = scorer.preprocess(img_resized)
        prob = scorer.score_tensors(torch.stack([tensor]))[0]
        is_ai = prob >= scorer.threshold
        if is_ai:  # Still correctly classified as AI
            correct_resized += 1
        total_resized += 1

    acc_resized = correct_resized / total_resized
    print(f"    AI detection accuracy: {correct_resized}/{total_resized} ({acc_resized:.2%})")

    # Test 3: Resize real images UP to match AI resolution
    print("\n[5] Testing real images RESIZED UP to 1024x1024 (matching AI):")
    correct_upscaled = 0
    total_upscaled = 0

    for p in real_paths:
        img = Image.open(p).convert("RGB")
        img_upscaled = img.resize((1024, 1024), Image.Resampling.LANCZOS)
        tensor = scorer.preprocess(img_upscaled)
        prob = scorer.score_tensors(torch.stack([tensor]))[0]
        is_ai = prob >= scorer.threshold
        if not is_ai:  # Still correctly classified as real
            correct_upscaled += 1
        total_upscaled += 1

    acc_upscaled = correct_upscaled / total_upscaled
    print(f"    Real detection accuracy: {correct_upscaled}/{total_upscaled} ({acc_upscaled:.2%})")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print(f"    Original accuracy:        {acc_original:.2%}")
    print(f"    AI resized down:          {acc_resized:.2%} (drop: {acc_original - acc_resized:.2%})")
    print(f"    Real resized up:          {acc_upscaled:.2%} (drop: {acc_original - acc_upscaled:.2%})")

    if acc_resized < acc_original - 0.05:
        print("\n    ⚠️  RESOLUTION IS LIKELY A SHORTCUT!")
        print("    AI images resized down are misclassified more often.")
    else:
        print("\n    ✅ Resolution does NOT appear to be a major shortcut.")
        print("    Model still detects AI images well at lower resolution.")

    if acc_upscaled < acc_original - 0.05:
        print("\n    ⚠️  Upscaled real images are being misclassified as AI!")
        print("    This suggests resolution IS being used as a feature.")
    else:
        print("\n    ✅ Real images upscaled are still correctly classified.")


if __name__ == "__main__":
    test_resolution_hypothesis()
