#!/usr/bin/env python3
"""
Full robustness evaluation with mirror padding preprocessing.

Compares:
1. Original ensemble (standard preprocessing)
2. Mirror padding ensemble
3. Hybrid (mirror pad only for extreme AR)

Metrics: clean accuracy, mean transformed accuracy, worst transform accuracy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from PIL import Image
from ensemble_core import EnsembleScorer
from robustness_utils import get_all_transforms, apply_transform

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


def hybrid_preprocess(img, ar_low=0.67, ar_high=1.5):
    """Use mirror padding only for extreme aspect ratios."""
    w, h = img.size
    ar = w / h

    if ar > ar_high or ar < ar_low:
        return mirror_pad_square(img)
    else:
        return img


def evaluate(scorer, real_paths, ai_paths, preprocess_func, name):
    """Evaluate a preprocessing method."""
    transforms = get_all_transforms()

    # Clean accuracy
    real_correct = 0
    real_total = 0
    ai_correct = 0
    ai_total = 0

    for p in real_paths[:100]:
        try:
            img = Image.open(p).convert("RGB")
            img_proc = preprocess_func(img)
            tensor = scorer.preprocess(img_proc)
            prob = scorer.score_tensors(torch.stack([tensor]))[0]
            real_total += 1
            if prob < scorer.threshold:
                real_correct += 1
        except:
            pass

    for p in ai_paths[:100]:
        try:
            img = Image.open(p).convert("RGB")
            img_proc = preprocess_func(img)
            tensor = scorer.preprocess(img_proc)
            prob = scorer.score_tensors(torch.stack([tensor]))[0]
            ai_total += 1
            if prob >= scorer.threshold:
                ai_correct += 1
        except:
            pass

    clean_acc = (real_correct + ai_correct) / (real_total + ai_total)

    # Transformed accuracy
    transform_accs = []

    for transform_name, transform in transforms:
        t_real_correct = 0
        t_real_total = 0
        t_ai_correct = 0
        t_ai_total = 0

        for p in real_paths[:50]:
            try:
                img = Image.open(p).convert("RGB")
                img_proc = preprocess_func(img)
                img_transformed = transform(img_proc)
                tensor = scorer.preprocess(img_transformed)
                prob = scorer.score_tensors(torch.stack([tensor]))[0]
                t_real_total += 1
                if prob < scorer.threshold:
                    t_real_correct += 1
            except:
                pass

        for p in ai_paths[:50]:
            try:
                img = Image.open(p).convert("RGB")
                img_proc = preprocess_func(img)
                img_transformed = transform(img_proc)
                tensor = scorer.preprocess(img_transformed)
                prob = scorer.score_tensors(torch.stack([tensor]))[0]
                t_ai_total += 1
                if prob >= scorer.threshold:
                    t_ai_correct += 1
            except:
                pass

        t_acc = (t_real_correct + t_ai_correct) / (t_real_total + t_ai_total)
        transform_accs.append((transform_name, t_acc))

    mean_transform = np.mean([a for _, a in transform_accs])
    worst_transform = min(transform_accs, key=lambda x: x[1])

    return {
        "name": name,
        "clean_acc": clean_acc,
        "mean_transform": mean_transform,
        "worst_transform": worst_transform[0],
        "worst_acc": worst_transform[1]
    }


def main():
    print("=" * 60)
    print("FULL ROBUSTNESS EVALUATION: PREPROCESSING COMPARISON")
    print("=" * 60)

    scorer = EnsembleScorer("ensemble_config.json")
    print(f"\n[1] Loaded {len(scorer.probes)} probes")

    real_paths = sorted(REAL_DIR.glob("*.jpg"))
    ai_paths = sorted(AI_DIR.rglob("*.jpg"))
    print(f"    Real images: {len(real_paths)}")
    print(f"    AI images: {len(ai_paths)}")

    # Define preprocessing methods
    methods = [
        ("Standard", lambda img: img),
        ("Mirror Pad (all)", lambda img: mirror_pad_square(img)),
        ("Hybrid (AR 0.67-1.5)", lambda img: hybrid_preprocess(img, 0.67, 1.5)),
    ]

    results = []
    for name, func in methods:
        print(f"\n[2] Evaluating: {name}...")
        r = evaluate(scorer, real_paths, ai_paths, func, name)
        results.append(r)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  {'Method':25s} {'Clean':8s} {'Mean':8s} {'Worst':8s} {'Worst Transform':20s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*20}")

    for r in results:
        print(f"  {r['name']:25s} {r['clean_acc']:8.4f} {r['mean_transform']:8.4f} {r['worst_acc']:8.4f} {r['worst_transform']:20s}")

    # Compare with original keeper
    print(f"\n  Original keeper: clean=0.9860, mean=0.9814, worst=0.9710")


if __name__ == "__main__":
    main()
