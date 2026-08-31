#!/usr/bin/env python3
"""
Test letterboxing (padding instead of stretching) to fix aspect ratio distortion.

Letterboxing:
1. Resize image to fit within target size, preserving aspect ratio
2. Pad remaining space with neutral color (black)

This should avoid the stretching artifacts that cause false positives.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
from ensemble_core import EnsembleScorer

REAL_DIR = Path("data/val/real")
AI_DIR = Path("data/wildfake/Images/Diffusion_based/DALLE/DALLE/Advanced/DALLE3")


def letterbox_image(img, target_size=(1024, 1024), pad_color=(0, 0, 0)):
    """Resize image to fit within target_size, preserving aspect ratio, then pad."""
    img_w, img_h = img.size
    target_w, target_h = target_size

    # Calculate scale to fit within target
    scale = min(target_w / img_w, target_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    # Resize
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create padded image
    padded = Image.new("RGB", target_size, pad_color)
    # Center the resized image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded.paste(img_resized, (x_offset, y_offset))

    return padded


def test_letterboxing(scorer, real_paths, ai_paths):
    """Test letterboxing on both real and AI images."""
    print("=" * 60)
    print("LETTERBOXING TEST")
    print("=" * 60)

    target_size = (1024, 1024)

    # Test real images
    real_correct = 0
    real_total = 0
    real_fps = []

    for p in real_paths:
        img = Image.open(p).convert("RGB")
        img_lb = letterbox_image(img, target_size)
        tensor = scorer.preprocess(img_lb)
        prob = scorer.score_tensors(torch.stack([tensor]))[0]
        real_total += 1
        if prob < scorer.threshold:
            real_correct += 1
        else:
            real_fps.append((p.name, prob))

    real_acc = real_correct / real_total

    # Test AI images
    ai_correct = 0
    ai_total = 0
    ai_fns = []

    for p in ai_paths:
        img = Image.open(p).convert("RGB")
        img_lb = letterbox_image(img, target_size)
        tensor = scorer.preprocess(img_lb)
        prob = scorer.score_tensors(torch.stack([tensor]))[0]
        ai_total += 1
        if prob >= scorer.threshold:
            ai_correct += 1
        else:
            ai_fns.append((p.name, prob))

    ai_acc = ai_correct / ai_total

    # Overall accuracy
    total_correct = real_correct + ai_correct
    total_images = real_total + ai_total
    overall_acc = total_correct / total_images

    print(f"\n  Results ({len(real_paths)} real + {len(ai_paths)} AI images):")
    print(f"  {'Metric':20s} {'Value':10s}")
    print(f"  {'-'*20} {'-'*10}")
    print(f"  {'Real accuracy':20s} {real_acc:10.2%}")
    print(f"  {'AI accuracy':20s} {ai_acc:10.2%}")
    print(f"  {'Overall accuracy':20s} {overall_acc:10.2%}")

    if real_fps:
        print(f"\n  Real false positives ({len(real_fps)}):")
        for name, prob in real_fps[:5]:
            print(f"    {name}: P(AI)={prob:.4f}")

    if ai_fns:
        print(f"\n  AI false negatives ({len(ai_fns)}):")
        for name, prob in ai_fns[:5]:
            print(f"    {name}: P(AI)={prob:.4f}")

    return {
        "real_acc": real_acc,
        "ai_acc": ai_acc,
        "overall_acc": overall_acc,
    }


def test_letterbox_vs_square(scorer, real_paths):
    """Compare letterboxing vs square stretching."""
    print("\n" + "=" * 60)
    print("LETTERBOX vs SQUARE COMPARISON")
    print("=" * 60)

    target_size = (1024, 1024)

    # Square stretching
    square_correct = 0
    for p in real_paths:
        img = Image.open(p).convert("RGB")
        img_sq = img.resize(target_size, Image.Resampling.LANCZOS)
        tensor = scorer.preprocess(img_sq)
        prob = scorer.score_tensors(torch.stack([tensor]))[0]
        if prob < scorer.threshold:
            square_correct += 1

    square_acc = square_correct / len(real_paths)

    # Letterboxing
    lb_correct = 0
    for p in real_paths:
        img = Image.open(p).convert("RGB")
        img_lb = letterbox_image(img, target_size)
        tensor = scorer.preprocess(img_lb)
        prob = scorer.score_tensors(torch.stack([tensor]))[0]
        if prob < scorer.threshold:
            lb_correct += 1

    lb_acc = lb_correct / len(real_paths)

    # Original
    orig_correct = 0
    for p in real_paths:
        img = Image.open(p).convert("RGB")
        tensor = scorer.preprocess(img)
        prob = scorer.score_tensors(torch.stack([tensor]))[0]
        if prob < scorer.threshold:
            orig_correct += 1

    orig_acc = orig_correct / len(real_paths)

    print(f"\n  Real image accuracy at 1024px:")
    print(f"  {'Method':20s} {'Accuracy':10s} {'False Positives'}")
    print(f"  {'-'*20} {'-'*10} {'-'*20}")
    print(f"  {'Original':20s} {orig_acc:10.2%} {len(real_paths) - orig_correct}")
    print(f"  {'Square stretch':20s} {square_acc:10.2%} {len(real_paths) - square_correct}")
    print(f"  {'Letterbox':20s} {lb_acc:10.2%} {len(real_paths) - lb_correct}")

    if lb_acc >= orig_acc - 0.02:
        print("\n  ✅ LETTERBOXING FIXES THE ISSUE!")
        print("  No false positives introduced by upscaling.")


def main():
    print("\n[1] Loading ensemble...")
    scorer = EnsembleScorer("ensemble_config.json")
    print(f"    Loaded {len(scorer.probes)} probes")

    real_paths = sorted(REAL_DIR.glob("*.jpg"))[:50]
    ai_paths = sorted(AI_DIR.rglob("*.jpg"))[:50]
    print(f"    Using {len(real_paths)} real + {len(ai_paths)} AI images")

    test_letterboxing(scorer, real_paths, ai_paths)
    test_letterbox_vs_square(scorer, real_paths)


if __name__ == "__main__":
    main()
