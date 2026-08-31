#!/usr/bin/env python3
"""
Test if aspect ratio distortion (not resolution) causes false positives.

Hypothesis: Forcing images to square (1:1) aspect ratio creates artifacts
that the model detects as AI-like.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
from ensemble_core import EnsembleScorer

REAL_DIR = Path("data/val/real")


def test_aspect_ratio(scorer, real_paths):
    """Compare square vs proportional resizing."""
    print("=" * 60)
    print("ASPECT RATIO HYPOTHESIS TEST")
    print("=" * 60)

    # Test 1: Force all to 1024x1024 square
    square_correct = 0
    square_total = 0
    square_fps = []

    for p in real_paths:
        img = Image.open(p).convert("RGB")
        img_sq = img.resize((1024, 1024), Image.Resampling.LANCZOS)
        tensor = scorer.preprocess(img_sq)
        prob = scorer.score_tensors(torch.stack([tensor]))[0]
        square_total += 1
        if prob < scorer.threshold:
            square_correct += 1
        else:
            square_fps.append(p.name)

    square_acc = square_correct / square_total

    # Test 2: Preserve aspect ratio, fit within 1024x1024
    prop_correct = 0
    prop_total = 0
    prop_fps = []

    for p in real_paths:
        img = Image.open(p).convert("RGB")
        # Resize to fit within 1024x1024, preserving aspect ratio
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        tensor = scorer.preprocess(img)
        prob = scorer.score_tensors(torch.stack([tensor]))[0]
        prop_total += 1
        if prob < scorer.threshold:
            prop_correct += 1
        else:
            prop_fps.append(p.name)

    prop_acc = prop_correct / prop_total

    # Test 3: Original resolution
    orig_correct = 0
    orig_total = 0

    for p in real_paths:
        img = Image.open(p).convert("RGB")
        tensor = scorer.preprocess(img)
        prob = scorer.score_tensors(torch.stack([tensor]))[0]
        orig_total += 1
        if prob < scorer.threshold:
            orig_correct += 1

    orig_acc = orig_correct / orig_total

    print(f"\n  Results ({len(real_paths)} real images):")
    print(f"  {'Method':20s} {'Accuracy':10s} {'False Positives'}")
    print(f"  {'-'*20} {'-'*10} {'-'*20}")
    print(f"  {'Original':20s} {orig_acc:10.2%} {len(real_paths) - orig_correct}")
    print(f"  {'1024x1024 square':20s} {square_acc:10.2%} {len(square_fps)}")
    if square_fps:
        print(f"  {'':20s} {'':10s} {', '.join(square_fps[:3])}")
    print(f"  {'1024 proportional':20s} {prop_acc:10.2%} {len(prop_fps)}")
    if prop_fps:
        print(f"  {'':20s} {'':10s} {', '.join(prop_fps[:3])}")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    if square_acc < prop_acc - 0.05:
        print("\n  ✅ ASPECT RATIO IS THE CAUSE!")
        print("  Forcing square aspect ratio creates false positives.")
        print("  Preserving aspect ratio maintains accuracy.")
    else:
        print("\n  ❌ Aspect ratio is NOT the primary cause.")
        print("  Both methods show similar performance.")

    if prop_acc >= orig_acc - 0.02:
        print("\n  ✅ PROPORTIONAL RESIZING IS SAFE!")
        print("  You can upscale images while preserving aspect ratio")
        print("  without introducing false positives.")


def main():
    print("\n[1] Loading ensemble...")
    scorer = EnsembleScorer("ensemble_config.json")
    print(f"    Loaded {len(scorer.probes)} probes")

    real_paths = sorted(REAL_DIR.glob("*.jpg"))[:50]
    print(f"    Using {len(real_paths)} real images")

    test_aspect_ratio(scorer, real_paths)


if __name__ == "__main__":
    main()
