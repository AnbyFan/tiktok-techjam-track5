#!/usr/bin/env python3
"""
Class-weighted logistic regression probe trainer.

Same as train_probe.py but adds --ai-weight to upweight the AI class loss.
Used to shift the decision boundary so the FROZEN t=0.5 cut lands well when
the base probe under-predicts AI (high AUROC, low AI recall at t=0.5).

Usage:
    python train_probe_cw.py --features features/sid_set --out probe_v7_cw2 \
        --ai-weight 2.0
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--out", default="probe_cw")
    p.add_argument("--real-label", type=int, default=0)
    p.add_argument("--ai-label", type=int, default=1)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--ai-weight", type=float, default=1.0,
                   help="class_weight for the AI class (real=1.0)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_feature_dir(d):
    d = Path(d)
    manifest = json.loads((d / "manifest.json").read_text())
    feats, labels = [], []
    for shard in manifest["shards"]:
        f = np.load(d / shard)
        meta_path = d / shard.replace("features_", "meta_").replace(".npy", ".csv")
        with meta_path.open() as fh:
            rows = list(csv.DictReader(fh))
        feats.append(f)
        labels.extend(int(r["label"]) for r in rows)
    return np.concatenate(feats), np.array(labels), manifest


def main():
    args = parse_args()
    Xs, ys = [], []
    for d in args.features:
        X, y, manifest = load_feature_dir(d)
        Xs.append(X)
        ys.append(y)
    X = np.concatenate(Xs)
    y_raw = np.concatenate(ys)

    mask = np.isin(y_raw, [args.real_label, args.ai_label])
    X = X[mask]
    y = (y_raw[mask] == args.ai_label).astype(int)
    n_real, n_ai = int((y == 0).sum()), int((y == 1).sum())
    print(f"[data] usable: {len(y)}  (real={n_real}, ai={n_ai})  ai_weight={args.ai_weight}")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=args.val_size + args.test_size,
        stratify=y, random_state=args.seed)
    rel_test = args.test_size / (args.val_size + args.test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_test, stratify=y_tmp, random_state=args.seed)

    clf = LogisticRegression(C=args.C, max_iter=3000,
                             class_weight={0: 1.0, 1: args.ai_weight})
    clf.fit(X_train, y_train)

    def report(Xs_, ys_, name):
        probs = clf.predict_proba(Xs_)[:, 1]
        preds = (probs >= args.threshold).astype(int)
        acc = accuracy_score(ys_, preds)
        ra = (1 - preds[ys_ == 0]).mean()
        aa = preds[ys_ == 1].mean()
        print(f"[{name}] acc={acc:.4f}  real={ra:.4f}  ai={aa:.4f}  n={len(ys_)}")
        return acc

    report(X_train, y_train, "train")
    report(X_val, y_val, "val")
    report(X_test, y_test, "test")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / "probe.joblib")
    config = {
        "feature_dirs": [str(d) for d in args.features],
        "feature_dim": int(X.shape[1]),
        "real_label": args.real_label,
        "ai_label": args.ai_label,
        "threshold": args.threshold,
        "C": args.C,
        "ai_weight": args.ai_weight,
        "seed": args.seed,
    }
    (out / "probe_config.json").write_text(json.dumps(config, indent=2))
    print(f"[saved] {out/'probe.joblib'}")


if __name__ == "__main__":
    main()
