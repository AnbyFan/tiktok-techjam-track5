#!/usr/bin/env python3
"""
Robustness evaluation harness with an N-probe ENSEMBLE (Track 5).

Same 15-config transforms as eval_robustness.py, but scores with a weighted
average of N probes' AI probabilities (loaded from a JSON config) to test
whether ensembling improves robustness / cross-generator generalization.

Usage:
    python eval_robustness_ensemble.py \
        --real-dir data/val/real \
        --ai-dir data/wildfake/Images/Diffusion_based/DALLE/DALLE/Advanced/DALLE3/dalle3 \
        --config ensemble_config.json --max-per-class 500 --out reports/dalle3_ensemble
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
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from ensemble_core import EnsembleScorer

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real-dir", required=True)
    p.add_argument("--ai-dir", required=True)
    p.add_argument("--config", default="ensemble_config.json",
                   help="JSON config listing member probes + weights")
    p.add_argument("--out", default="reports/ensemble_run1")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None,
                   help="override; default reads config (recommended)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_specs():
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


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = build_specs()

    scorer = EnsembleScorer(args.config)
    threshold = args.threshold if args.threshold is not None else scorer.threshold
    print(f"[ensemble] {len(scorer.probes)} members: "
          f"{[n for n, _ in scorer.probes]}  (threshold={threshold})")

    def list_images(d):
        paths = sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in EXTS)
        if args.max_per_class:
            paths = paths[: args.max_per_class]
        return paths

    items = [(p, 0) for p in list_images(args.real_dir)] + \
            [(p, 1) for p in list_images(args.ai_dir)]
    n_real = sum(1 for _, l in items if l == 0)
    print(f"[data] real={n_real} ai={len(items) - n_real} "
          f"x {len(specs)} transforms = {len(items) * len(specs)} scorings")
    if n_real == 0 or n_real == len(items):
        raise SystemExit("Need images in BOTH --real-dir and --ai-dir.")

    pred_f = (out / "predictions.csv").open("w", newline="")
    pred_w = csv.writer(pred_f)
    pred_w.writerow(["image_path", "true_label", "true_class", "transform",
                     "prob_ai", "pred", "correct"])

    report_rows = []
    for name, kind, params in specs:
        probs_all, labels_all = [], []
        skipped = 0
        batches = [items[i:i + args.batch_size]
                   for i in range(0, len(items), args.batch_size)]
        for batch in tqdm(batches, desc=name, unit="batch", leave=False):
            tensors, meta = [], []
            for path, label in batch:
                try:
                    img = Image.open(path).convert("RGB")
                    rng = np.random.default_rng(
                        (args.seed + zlib.crc32(f"{path}|{name}".encode())) % 2**32)
                    tensors.append(scorer.preprocess(apply_transform(img, kind, params, rng)))
                    meta.append((str(path), label))
                except Exception:
                    skipped += 1
            if not tensors:
                continue
            probs = scorer.score_tensors(torch.stack(tensors))
            for (path, label), prob in zip(meta, probs):
                pred = int(prob >= threshold)
                pred_w.writerow([path, label, "ai" if label else "real",
                                 name, f"{prob:.6f}", pred, int(pred == label)])
                probs_all.append(prob)
                labels_all.append(label)

        y = np.array(labels_all)
        p = (np.array(probs_all) >= threshold).astype(int)
        acc = accuracy_score(y, p)
        real_acc = accuracy_score(y[y == 0], p[y == 0])
        ai_acc = accuracy_score(y[y == 1], p[y == 1])
        auroc = roc_auc_score(y, probs_all)
        report_rows.append({"transform": name, "n": len(y), "accuracy": acc,
                            "real_acc": real_acc, "ai_acc": ai_acc,
                            "auroc": auroc, "skipped": skipped})
        tqdm.write(f"[{name:14s}] acc={acc:.4f}  real={real_acc:.4f}  "
                   f"ai={ai_acc:.4f}  auroc={auroc:.4f}")

    pred_f.close()

    with (out / "robustness_report.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        w.writeheader()
        w.writerows(report_rows)

    lines = ["| Transform | N | Accuracy | Real acc | AI acc | AUROC |",
             "|---|---|---|---|---|---|"]
    for r in report_rows:
        lines.append(f"| {r['transform']} | {r['n']} | {r['accuracy']:.4f} "
                     f"| {r['real_acc']:.4f} | {r['ai_acc']:.4f} | {r['auroc']:.4f} |")
    (out / "robustness_report.md").write_text("\n".join(lines) + "\n")

    clean = report_rows[0]["accuracy"]
    transformed = [r["accuracy"] for r in report_rows[1:]]
    worst = min(report_rows[1:], key=lambda r: r["accuracy"])
    print(f"\n[summary] clean={clean:.4f}  "
          f"mean-transformed={np.mean(transformed):.4f}  "
          f"worst={worst['transform']} ({worst['accuracy']:.4f})")
    print(f"[saved] {out/'robustness_report.md'}  <- paste-ready table")


if __name__ == "__main__":
    main()
