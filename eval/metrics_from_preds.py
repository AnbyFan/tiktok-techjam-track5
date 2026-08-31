#!/usr/bin/env python3
"""Compute per-transform metrics from a predictions.csv (frozen t=0.5)."""
import argparse
import csv
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    args = ap.parse_args()

    rows = defaultdict(list)
    with open(args.preds, newline="") as f:
        for r in csv.DictReader(f):
            tc = str(r.get("true_class", "")).strip().lower()
            y = 1 if tc in ("ai", "1", "ai_generated", "synthetic") else 0
            p = float(r["prob_ai"])
            pred = 1 if p >= 0.5 else 0
            rows[r["transform"]].append((y, p, pred))

    def auroc(pairs):
        pos = [p for y, p, _ in pairs if y == 1]
        neg = [p for y, p, _ in pairs if y == 0]
        if not pos or not neg:
            return float("nan")
        wins = sum(1 for a in pos for b in neg if a > b) + \
               0.5 * sum(1 for a in pos for b in neg if a == b)
        return wins / (len(pos) * len(neg))

    order = ["clean", "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30",
             "blur_s0.5", "blur_s1.0", "blur_s2.0", "resize_0.5x",
             "resize_0.25x", "noise_s0.02", "noise_s0.05", "noise_s0.1",
             "jitter_20pct", "crop_80pct"]
    print(f"{'transform':<14} {'acc':>7} {'real':>7} {'ai':>7} {'auroc':>7}")
    xform_accs = []
    for t in order:
        if t not in rows:
            continue
        pairs = rows[t]
        y = [a for a, _, _ in pairs]
        pred = [c for _, _, c in pairs]
        acc = sum(1 for a, c in zip(y, pred) if a == c) / len(y)
        ra = sum(1 - c for a, c in zip(y, pred) if a == 0) / max(1, sum(1 for a in y if a == 0))
        aa = sum(c for a, c in zip(y, pred) if a == 1) / max(1, sum(1 for a in y if a == 1))
        au = auroc(pairs)
        if t != "clean":
            xform_accs.append(acc)
        print(f"{t:<14} {acc:.4f} {ra:.4f} {aa:.4f} {au:.4f}")
    if xform_accs:
        print(f"\nmean-transformed = {sum(xform_accs)/len(xform_accs):.4f}")
        worst = min(xform_accs)
        print(f"worst = {worst:.4f}")


if __name__ == "__main__":
    main()
