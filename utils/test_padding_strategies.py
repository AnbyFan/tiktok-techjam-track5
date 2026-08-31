#!/usr/bin/env python3
"""
Test different padding strategies for letterboxing.

Strategies:
1. Black padding (current)
2. White padding
3. Gray padding (128,128,128)
4. Mirror/reflection padding
5. Edge replication padding
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
from ensemble_core import EnsembleScorer

REAL_DIR = Path("data/val/real")


def letterbox_black(img, target_size=(1024, 1024)):
    """Black padding."""
    img_w, img_h = img.size
    target_w, target_h = target_size
    scale = min(target_w / img_w, target_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    padded = Image.new("RGB", target_size, (0, 0, 0))
    x_off, y_off = (target_w - new_w) // 2, (target_h - new_h) // 2
    padded.paste(img_resized, (x_off, y_off))
    return padded


def letterbox_white(img, target_size=(1024, 1024)):
    """White padding."""
    img_w, img_h = img.size
    target_w, target_h = target_size
    scale = min(target_w / img_w, target_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    padded = Image.new("RGB", target_size, (255, 255, 255))
    x_off, y_off = (target_w - new_w) // 2, (target_h - new_h) // 2
    padded.paste(img_resized, (x_off, y_off))
    return padded


def letterbox_gray(img, target_size=(1024, 1024)):
    """Gray padding."""
    img_w, img_h = img.size
    target_w, target_h = target_size
    scale = min(target_w / img_w, target_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    padded = Image.new("RGB", target_size, (128, 128, 128))
    x_off, y_off = (target_w - new_w) // 2, (target_h - new_h) // 2
    padded.paste(img_resized, (x_off, y_off))
    return padded


def letterbox_mirror(img, target_size=(1024, 1024)):
    """Mirror/reflection padding."""
    img_w, img_h = img.size
    target_w, target_h = target_size
    scale = min(target_w / img_w, target_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create canvas
    padded = Image.new("RGB", target_size, (0, 0, 0))
    x_off, y_off = (target_w - new_w) // 2, (target_h - new_h) // 2

    # Paste original
    padded.paste(img_resized, (x_off, y_off))

    # Add mirror padding
    arr = np.array(padded)
    img_arr = np.array(img_resized)

    # Top padding (mirror)
    if y_off > 0:
        pad_h = min(y_off, img_h)
        arr[:y_off, x_off:x_off+new_w] = img_arr[-pad_h:, :, :][::-1, :, :]

    # Bottom padding (mirror)
    if y_off + new_h < target_h:
        pad_h = min(target_h - y_off - new_h, img_h)
        arr[y_off+new_h:, x_off:x_off+new_w] = img_arr[:pad_h, :, :][::-1, :, :]

    # Left padding (mirror)
    if x_off > 0:
        pad_w = min(x_off, img_w)
        arr[:, :x_off] = arr[:, x_off:x_off+pad_w][:, ::-1, :]

    # Right padding (mirror)
    if x_off + new_w < target_w:
        pad_w = min(target_w - x_off - new_w, img_w)
        arr[:, x_off+new_w:] = arr[:, x_off+new_w-pad_w:x_off+new_w][:, ::-1, :]

    return Image.fromarray(arr)


def letterbox_edge(img, target_size=(1024, 1024)):
    """Edge replication padding."""
    img_w, img_h = img.size
    target_w, target_h = target_size
    scale = min(target_w / img_w, target_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create canvas with edge color
    arr_resized = np.array(img_resized)

    # Get edge colors
    top_edge = arr_resized[0, :, :]
    bottom_edge = arr_resized[-1, :, :]
    left_edge = arr_resized[:, 0, :]
    right_edge = arr_resized[:, -1, :]

    # Create full canvas
    padded_arr = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Fill with edge colors
    x_off, y_off = (target_w - new_w) // 2, (target_h - new_h) // 2

    # Top region
    if y_off > 0:
        padded_arr[:y_off, :] = np.tile(top_edge, (y_off, 1, 1))[:y_off, :target_w]

    # Bottom region
    if y_off + new_h < target_h:
        padded_arr[y_off+new_h:, :] = np.tile(bottom_edge, (target_h-y_off-new_h, 1, 1))[:target_h-y_off-new_h, :target_w]

    # Left region
    if x_off > 0:
        padded_arr[:, :x_off] = np.tile(left_edge, (1, x_off, 1))[:target_h, :x_off]

    # Right region
    if x_off + new_w < target_w:
        padded_arr[:, x_off+new_w:] = np.tile(right_edge, (1, target_w-x_off-new_w, 1))[:target_h, x_off+new_w:]

    # Center
    padded_arr[y_off:y_off+new_h, x_off:x_off+new_w] = arr_resized

    return Image.fromarray(padded_arr)


def test_padding_strategy(scorer, real_paths, strategy_name, strategy_func):
    """Test a single padding strategy."""
    correct = 0
    total = 0
    fps = []

    for p in real_paths:
        img = Image.open(p).convert("RGB")
        img_padded = strategy_func(img, (1024, 1024))
        tensor = scorer.preprocess(img_padded)
        prob = scorer.score_tensors(torch.stack([tensor]))[0]
        total += 1
        if prob < scorer.threshold:
            correct += 1
        else:
            fps.append((p.name, prob))

    acc = correct / total
    return {
        "name": strategy_name,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "false_positives": fps
    }


def main():
    print("\n[1] Loading ensemble...")
    scorer = EnsembleScorer("ensemble_config.json")
    print(f"    Loaded {len(scorer.probes)} probes")

    real_paths = sorted(REAL_DIR.glob("*.jpg"))[:50]
    print(f"    Using {len(real_paths)} real images")

    # Test original
    orig_correct = 0
    for p in real_paths:
        img = Image.open(p).convert("RGB")
        tensor = scorer.preprocess(img)
        prob = scorer.score_tensors(torch.stack([tensor]))[0]
        if prob < scorer.threshold:
            orig_correct += 1
    orig_acc = orig_correct / len(real_paths)

    print("\n" + "=" * 60)
    print("PADDING STRATEGY COMPARISON")
    print("=" * 60)
    print(f"\n  Real image accuracy at 1024px:")
    print(f"  {'Strategy':20s} {'Accuracy':10s} {'FPs':5s}")
    print(f"  {'-'*20} {'-'*10} {'-'*5}")
    print(f"  {'Original':20s} {orig_acc:10.2%} {len(real_paths) - orig_correct:5d}")

    strategies = [
        ("Square stretch", lambda img, size: img.resize(size, Image.Resampling.LANCZOS)),
        ("Black padding", letterbox_black),
        ("White padding", letterbox_white),
        ("Gray padding", letterbox_gray),
        ("Mirror padding", letterbox_mirror),
        ("Edge padding", letterbox_edge),
    ]

    results = []
    for name, func in strategies:
        r = test_padding_strategy(scorer, real_paths, name, func)
        results.append(r)
        print(f"  {r['name']:20s} {r['accuracy']:10.2%} {len(r['false_positives']):5d}")
        if r['false_positives']:
            fp_names = [fp[0] for fp in r['false_positives'][:3]]
            print(f"  {'':20s} {'':10s} {', '.join(fp_names)}")

    # Best strategy
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\n  ✅ Best strategy: {best['name']} ({best['accuracy']:.2%})")


if __name__ == "__main__":
    main()
