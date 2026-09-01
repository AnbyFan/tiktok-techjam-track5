#!/usr/bin/env python3
"""
Test if mirror padding improves performance on:
1. High-resolution images (1024px+)
2. Images with extreme aspect ratios (very wide or very tall)

This targets the specific cases where aspect ratio distortion was a problem.
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


def categorize_images(real_paths, ai_paths):
    """Categorize images by resolution and aspect ratio."""
    categories = {
        "high_res_real": [],      # Real images >= 1000px
        "high_res_ai": [],        # AI images >= 1000px
        "wide_real": [],          # Aspect ratio > 1.5
        "tall_real": [],          # Aspect ratio < 0.67
        "wide_ai": [],
        "tall_ai": [],
    }

    for p in real_paths:
        with Image.open(p) as img:
            w, h = img.size
            ar = w / h
            if min(w, h) >= 1000:
                categories["high_res_real"].append(p)
            if ar > 1.5:
                categories["wide_real"].append(p)
            elif ar < 0.67:
                categories["tall_real"].append(p)

    for p in ai_paths:
        with Image.open(p) as img:
            w, h = img.size
            ar = w / h
            if min(w, h) >= 1000:
                categories["high_res_ai"].append(p)
            if ar > 1.5:
                categories["wide_ai"].append(p)
            elif ar < 0.67:
                categories["tall_ai"].append(p)

    return categories


def test_category(scorer, paths, label, method_name, use_mirror_pad=False):
    """Test a category of images with a specific method."""
    correct = 0
    total = 0

    for p in paths:
        try:
            img = Image.open(p).convert("RGB")

            if use_mirror_pad:
                img_proc = mirror_pad_square(img)
                tensor = scorer.preprocess(img_proc)
            else:
                tensor = scorer.preprocess(img)

            prob = scorer.score_tensors(torch.stack([tensor]))[0]
            pred_ai = prob >= scorer.threshold

            total += 1
            # Correct if prediction matches true label
            if (label == 1 and pred_ai) or (label == 0 and not pred_ai):
                correct += 1
        except Exception:
            pass

    acc = correct / total if total > 0 else 0
    return {
        "method": method_name,
        "category": label,
        "accuracy": acc,
        "correct": correct,
        "total": total
    }


def main():
    print("=" * 60)
    print("HIGH-RES & ASPECT RATIO TEST")
    print("=" * 60)

    scorer = EnsembleScorer("ensemble_config.json")
    print(f"\n[1] Loaded {len(scorer.probes)} probes")

    real_paths = sorted(REAL_DIR.glob("*.jpg"))
    ai_paths = sorted(AI_DIR.rglob("*.jpg"))
    print(f"    Real images: {len(real_paths)}")
    print(f"    AI images: {len(ai_paths)}")

    # Categorize
    print("\n[2] Categorizing images...")
    cats = categorize_images(real_paths, ai_paths)

    for name, paths in cats.items():
        print(f"    {name}: {len(paths)} images")

    # Test each category
    print("\n[3] Testing methods...")
    results = []

    test_configs = [
        ("high_res_real", cats["high_res_real"], 0, "High-res real (>=1000px)"),
        ("high_res_ai", cats["high_res_ai"], 1, "High-res AI (>=1000px)"),
        ("wide_real", cats["wide_real"], 0, "Wide real (AR>1.5)"),
        ("tall_real", cats["tall_real"], 0, "Tall real (AR<0.67)"),
        ("wide_ai", cats["wide_ai"], 1, "Wide AI (AR>1.5)"),
        ("tall_ai", cats["tall_ai"], 1, "Tall AI (AR<0.67)"),
    ]

    for key, paths, label, desc in test_configs:
        if len(paths) < 5:
            print(f"\n  {desc}: Skipping (only {len(paths)} images)")
            continue

        # Standard method
        std_result = test_category(scorer, paths[:50], label, "Standard", use_mirror_pad=False)

        # Mirror pad method
        mp_result = test_category(scorer, paths[:50], label, "Mirror Pad", use_mirror_pad=True)

        results.append({
            "description": desc,
            "standard": std_result,
            "mirror_pad": mp_result
        })

        diff = mp_result["accuracy"] - std_result["accuracy"]
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"\n  {desc}:")
        print(f"    Standard:   {std_result['accuracy']:.2%} ({std_result['correct']}/{std_result['total']})")
        print(f"    Mirror Pad: {mp_result['accuracy']:.2%} ({mp_result['correct']}/{mp_result['total']}) {arrow} {diff:+.2%}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  {'Category':30s} {'Standard':10s} {'Mirror Pad':12s} {'Diff':8s}")
    print(f"  {'-'*30} {'-'*10} {'-'*12} {'-'*8}")

    improvements = 0
    total_tests = 0

    for r in results:
        std_acc = r["standard"]["accuracy"]
        mp_acc = r["mirror_pad"]["accuracy"]
        diff = mp_acc - std_acc
        total_tests += 1

        if diff > 0.01:
            improvements += 1
            marker = "✅"
        elif diff < -0.01:
            marker = "❌"
        else:
            marker = "➖"

        print(f"  {r['description']:30s} {std_acc:10.2%} {mp_acc:12.2%} {diff:+8.2%} {marker}")

    print(f"\n  Mirror pad improvements: {improvements}/{total_tests} categories")


if __name__ == "__main__":
    main()
