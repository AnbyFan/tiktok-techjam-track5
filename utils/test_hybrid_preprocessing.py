#!/usr/bin/env python3
"""
Test hybrid preprocessing: use mirror padding only for extreme aspect ratios.

Strategy:
- Standard CLIP preprocessing for normal aspect ratios (0.67 <= AR <= 1.5)
- Mirror padding for extreme aspect ratios (AR > 1.5 or AR < 0.67)

This should fix the aspect ratio issue without hurting overall performance.
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


def mirror_pad_square(img, target_size=224):
    """Resize and mirror pad to square."""
    img_w, img_h = img.size
    scale = min(target_size / img_w, target_size / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    padded = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    x_off, y_off = (target_size - new_w) // 2, (target_size - new_h) // 2
    padded.paste(img_resized, (x_off, y_off))

    arr = np.array(padded)
    img_arr = np.array(img_resized)

    if y_off > 0:
        pad_h = min(y_off, new_h)
        arr[:y_off, x_off:x_off+new_w] = img_arr[-pad_h:, :, :][::-1, :, :]
    if y_off + new_h < target_size:
        pad_h = min(target_size - y_off - new_h, new_h)
        arr[y_off+new_h:, x_off:x_off+new_w] = img_arr[:pad_h, :, :][::-1, :, :]
    if x_off > 0:
        pad_w = min(x_off, new_w)
        arr[:, :x_off] = arr[:, x_off:x_off+pad_w][:, ::-1, :]
    if x_off + new_w < target_size:
        pad_w = min(target_size - x_off - new_w, new_w)
        arr[:, x_off+new_w:] = arr[:, x_off+new_w-pad_w:x_off+new_w][:, ::-1, :]

    return Image.fromarray(arr)


def hybrid_preprocess(img, scorer, ar_threshold_low=0.67, ar_threshold_high=1.5):
    """Use mirror padding only for extreme aspect ratios."""
    w, h = img.size
    ar = w / h

    # Extreme aspect ratio -> use mirror padding
    if ar > ar_threshold_high or ar < ar_threshold_low:
        img_padded = mirror_pad_square(img)
        return scorer.preprocess(img_padded)
    else:
        # Normal aspect ratio -> standard preprocessing
        return scorer.preprocess(img)


def test_method(scorer, real_paths, ai_paths, preprocess_func, method_name):
    """Test a preprocessing method on both real and AI images."""
    real_correct = 0
    real_total = 0
    ai_correct = 0
    ai_total = 0

    for p in real_paths:
        try:
            img = Image.open(p).convert("RGB")
            tensor = preprocess_func(img, scorer)
            prob = scorer.score_tensors(torch.stack([tensor]))[0]
            real_total += 1
            if prob < scorer.threshold:
                real_correct += 1
        except Exception:
            pass

    for p in ai_paths:
        try:
            img = Image.open(p).convert("RGB")
            tensor = preprocess_func(img, scorer)
            prob = scorer.score_tensors(torch.stack([tensor]))[0]
            ai_total += 1
            if prob >= scorer.threshold:
                ai_correct += 1
        except Exception:
            pass

    real_acc = real_correct / real_total if real_total > 0 else 0
    ai_acc = ai_correct / ai_total if ai_total > 0 else 0
    overall_acc = (real_correct + ai_correct) / (real_total + ai_total)

    return {
        "method": method_name,
        "real_acc": real_acc,
        "ai_acc": ai_acc,
        "overall_acc": overall_acc,
        "real_correct": real_correct,
        "real_total": real_total,
        "ai_correct": ai_correct,
        "ai_total": ai_total
    }


def main():
    print("=" * 60)
    print("HYBRID PREPROCESSING TEST")
    print("=" * 60)

    scorer = EnsembleScorer("ensemble_config.json")
    print(f"\n[1] Loaded {len(scorer.probes)} probes")

    real_paths = sorted(REAL_DIR.glob("*.jpg"))[:100]
    ai_paths = sorted(AI_DIR.rglob("*.jpg"))[:100]
    print(f"    Testing with {len(real_paths)} real + {len(ai_paths)} AI images")

    # Define preprocessing methods
    methods = [
        ("Standard", lambda img, s: s.preprocess(img)),
        ("Mirror Pad (all)", lambda img, s: s.preprocess(mirror_pad_square(img))),
        ("Hybrid (AR 0.67-1.5)", lambda img, s: hybrid_preprocess(img, s, 0.67, 1.5)),
        ("Hybrid (AR 0.5-2.0)", lambda img, s: hybrid_preprocess(img, s, 0.5, 2.0)),
        ("Hybrid (AR 0.33-3.0)", lambda img, s: hybrid_preprocess(img, s, 0.33, 3.0)),
    ]

    results = []
    for name, func in methods:
        print(f"\n[2] Testing: {name}...")
        r = test_method(scorer, real_paths, ai_paths, func, name)
        results.append(r)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  {'Method':25s} {'Real':10s} {'AI':10s} {'Overall':10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    for r in results:
        print(f"  {r['method']:25s} {r['real_acc']:10.2%} {r['ai_acc']:10.2%} {r['overall_acc']:10.2%}")

    # Best method
    best = max(results, key=lambda x: x['overall_acc'])
    print(f"\n  ✅ Best method: {best['method']} ({best['overall_acc']:.2%})")


if __name__ == "__main__":
    main()
