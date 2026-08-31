#!/usr/bin/env python3
"""Analyze threshold headroom from a predictions.csv (clean rows only).

Reads prob_ai + true_label for the 'clean' transform and reports accuracy
across candidate thresholds, plus the accuracy-optimal threshold. This tells
us whether calibration (Platt scaling) could recover accuracy at the frozen
t=0.5 by reshaping the probability curve.
"""
import csv
import sys
import numpy as np

path = sys.argv[1]
transform_filter = sys.argv[2] if len(sys.argv) > 2 else "clean"

probs, labels = [], []
with open(path, newline="") as f:
    for row in csv.DictReader(f):
        if row["transform"] != transform_filter:
            continue
        probs.append(float(row["prob_ai"]))
        labels.append(int(row["true_label"]))

y = np.array(labels)
p = np.array(probs)
print(f"[data] {len(y)} rows, transform={transform_filter}, "
      f"n_real={int((y==0).sum())}, n_ai={int((y==1).sum())}")

def metrics(t):
    pred = (p >= t).astype(int)
    acc = (pred == y).mean()
    ra = (1 - pred[y == 0]).mean()   # reals correctly called real
    aa = pred[y == 1].mean()         # AI correctly called AI
    return acc, ra, aa

acc05, ra05, aa05 = metrics(0.5)
print(f"[t=0.500] accuracy={acc05:.4f}  real={ra05:.4f}  ai={aa05:.4f}")

# Sweep thresholds
best_t, best_acc = None, -1
for t in np.arange(0.10, 0.91, 0.01):
    acc, _, _ = metrics(t)
    if acc > best_acc:
        best_acc, best_t = acc, t
print(f"[best ] t={best_t:.2f} accuracy={best_acc:.4f}  "
      f"(headroom vs 0.5: {best_acc-acc05:+.4f})")

# Show a coarse sweep
print("\n  t     acc    real_acc  ai_acc")
for t in (0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8):
    acc, ra, aa = metrics(t)
    print(f" {t:.2f}  {acc:.4f}  {ra:.4f}    {aa:.4f}")
