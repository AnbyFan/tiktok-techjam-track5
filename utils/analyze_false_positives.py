#!/usr/bin/env python3
"""
Analyze the specific images that become false positives when upscaled.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
from ensemble_core import EnsembleScorer

REAL_DIR = Path("data/val/real")

# The 3 false positive images from previous test
FP_IMAGES = [
    "000000000139.jpg",
    "000000003156.jpg",
    "000000003255.jpg",
]


def analyze_fp_images(scorer):
    print("=" * 60)
    print("FALSE POSITIVE IMAGE ANALYSIS")
    print("=" * 60)

    for fname in FP_IMAGES:
        p = REAL_DIR / fname
        if not p.exists():
            print(f"\n  {fname}: NOT FOUND")
            continue

        img = Image.open(p).convert("RGB")
        orig_size = img.size

        # Test at original resolution
        tensor_orig = scorer.preprocess(img)
        prob_orig = scorer.score_tensors(torch.stack([tensor_orig]))[0]

        # Test at 1024x1024
        img_1024 = img.resize((1024, 1024), Image.Resampling.LANCZOS)
        tensor_1024 = scorer.preprocess(img_1024)
        prob_1024 = scorer.score_tensors(torch.stack([tensor_1024]))[0]

        # Test at different aspect ratios (preserve original ratio)
        w, h = orig_size
        if w > h:
            # Wide image - resize to 1024 wide, proportional height
            new_h = int(1024 * h / w)
            img_wide = img.resize((1024, new_h), Image.Resampling.LANCZOS)
        else:
            # Tall image - resize to 1024 tall, proportional width
            new_w = int(1024 * w / h)
            img_wide = img.resize((new_w, 1024), Image.Resampling.LANCZOS)

        tensor_wide = scorer.preprocess(img_wide)
        prob_wide = scorer.score_tensors(torch.stack([tensor_wide]))[0]

        print(f"\n  {fname}:")
        print(f"    Original size: {orig_size[0]}x{orig_size[1]}")
        print(f"    P(AI) original:    {prob_orig:.4f} {'❌ FP' if prob_orig >= scorer.threshold else '✅ OK'}")
        print(f"    P(AI) 1024x1024:   {prob_1024:.4f} {'❌ FP' if prob_1024 >= scorer.threshold else '✅ OK'}")
        print(f"    P(AI) 1024-prop:   {prob_wide:.4f} {'❌ FP' if prob_wide >= scorer.threshold else '✅ OK'}")


def test_content_patterns(scorer):
    """Check if there's a content pattern (e.g., faces, text, etc.)"""
    print("\n" + "=" * 60)
    print("CONTENT PATTERN TEST")
    print("=" * 60)
    print("Testing if certain content types are more prone to FP\n")

    # Get all real images and test at 1024x1024
    real_paths = sorted(REAL_DIR.glob("*.jpg"))
    fp_images = []
    ok_images = []

    for p in real_paths:
        img = Image.open(p).convert("RGB")
        img_1024 = img.resize((1024, 1024), Image.Resampling.LANCZOS)
        tensor = scorer.preprocess(img_1024)
        prob = scorer.score_tensors(torch.stack([tensor]))[0]

        if prob >= scorer.threshold:
            fp_images.append((p.name, prob, img.size))
        else:
            ok_images.append((p.name, prob, img.size))

    print(f"  Total images tested: {len(real_paths)}")
    print(f"  False positives: {len(fp_images)}")
    print(f"  Correct: {len(ok_images)}")

    if fp_images:
        print("\n  False positive details:")
        for name, prob, size in sorted(fp_images, key=lambda x: -x[1]):
            print(f"    {name}: P(AI)={prob:.4f}, size={size[0]}x{size[1]}")


def main():
    print("\n[1] Loading ensemble...")
    scorer = EnsembleScorer("ensemble_config.json")
    print(f"    Loaded {len(scorer.probes)} probes")

    analyze_fp_images(scorer)
    test_content_patterns(scorer)


if __name__ == "__main__":
    main()
