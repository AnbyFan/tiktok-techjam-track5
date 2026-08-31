#!/usr/bin/env python3
"""
Train a probe with feature-level noise augmentation.
Adds Gaussian noise to CLIP features during training to improve
robustness to image-level noise transforms.

Usage:
    python train_noisy.py --features features/sid_set features/comfy_aug --out probe_v5_noisy
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--out", default="probe_v5_noisy")
    p.add_argument("--real-label", type=int, default=0)
    p.add_argument("--ai-label", type=int, default=1)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--noise-sigma", type=float, default=0.05,
                   help="std of Gaussian noise added to features (0=off)")
    p.add_argument("--noise-prob", type=float, default=0.3,
                   help="probability of adding noise to each training sample")
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
    rng = np.random.default_rng(args.seed)

    Xs, ys = [], []
    for d in args.features:
        X, y, manifest = load_feature_dir(d)
        print(f"[load] {d}: {X.shape[0]} samples x {X.shape[1]} dims")
        Xs.append(X)
        ys.append(y)
    X = np.concatenate(Xs)
    y_raw = np.concatenate(ys)

    mask = np.isin(y_raw, [args.real_label, args.ai_label])
    X = X[mask]
    y = (y_raw[mask] == args.ai_label).astype(int)
    print(f"[data] usable: {len(y)}  (real={int((y==0).sum())}, ai={int((y==1).sum())})")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=args.val_size + args.test_size,
        stratify=y, random_state=args.seed)
    rel_test = args.test_size / (args.val_size + args.test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_test, stratify=y_tmp, random_state=args.seed)
    print(f"[split] train={len(y_train)} val={len(y_val)} test={len(y_test)}")

    # Feature-level noise augmentation on training set only
    if args.noise_sigma > 0:
        print(f"[augment] adding feature noise: sigma={args.noise_sigma}, prob={args.noise_prob}")
        n_orig = len(X_train)
        noise_mask = rng.random(n_orig) < args.noise_prob
        X_noisy = X_train[noise_mask].copy()
        X_noisy += rng.normal(0, args.noise_sigma, X_noisy.shape)
        # Re-normalize after noise (CLIP features are L2-normalized)
        X_noisy = X_noisy / np.linalg.norm(X_noisy, axis=1, keepdims=True)
        X_train_aug = np.vstack([X_train, X_noisy])
        y_train_aug = np.concatenate([y_train, y_train[noise_mask]])
        print(f"[augment] {n_orig} -> {len(X_train_aug)} training samples "
              f"(+{len(X_noisy)} noisy)")
    else:
        X_train_aug, y_train_aug = X_train, y_train

    clf = LogisticRegression(C=args.C, max_iter=2000)
    clf.fit(X_train_aug, y_train_aug)

    for name, Xs_, ys_ in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        probs = clf.predict_proba(Xs_)[:, 1]
        preds = (probs >= args.threshold).astype(int)
        acc = accuracy_score(ys_, preds)
        auroc = roc_auc_score(ys_, probs)
        print(f"[{name}] acc={acc:.4f}  auroc={auroc:.4f}  n={len(ys_)}")

    test_probs = clf.predict_proba(X_test)[:, 1]
    print("\n[test] confusion matrix:")
    print(confusion_matrix(y_test, (test_probs >= args.threshold).astype(int)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / "probe.joblib")
    config = {
        "feature_dirs": [str(d) for d in args.features],
        "feature_dim": int(X.shape[1]),
        "model_type": "logistic_regression_noisy",
        "noise_sigma": args.noise_sigma,
        "noise_prob": args.noise_prob,
        "real_label": args.real_label,
        "ai_label": args.ai_label,
        "threshold": args.threshold,
        "C": args.C,
        "seed": args.seed,
        "test_acc_at_threshold": float(accuracy_score(y_test, (test_probs >= args.threshold).astype(int))),
    }
    (out / "probe_config.json").write_text(json.dumps(config, indent=2))
    print(f"\n[saved] {out/'probe.joblib'}")


if __name__ == "__main__":
    main()
