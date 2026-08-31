#!/usr/bin/env python3
"""
Train a linear probe (logistic regression) on CLIP features cached by
extract_features.py. Reports clean-data accuracy and per-class metrics,
and saves the probe + config for the inference/scoring script.

Usage:
    # Train on one feature cache
    python train_probe.py --features features/sid_set

    # Combine multiple caches later (e.g. add CIFAKE / own generations)
    python train_probe.py --features features/sid_set features/cifake --out probe_v2

Requires: pip install scikit-learn joblib numpy
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", nargs="+", required=True,
                   help="one or more feature dirs produced by extract_features.py")
    p.add_argument("--out", default="probe_v1")
    p.add_argument("--real-label", type=int, default=0,
                   help="must match what you used in extract_features.py")
    p.add_argument("--ai-label", type=int, default=1)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="decision threshold on P(AI); freeze this once chosen")
    p.add_argument("--C", type=float, default=1.0, help="inverse regularization")
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
        if len(rows) != len(f):
            raise ValueError(f"shard/meta row mismatch in {d}: {shard}")
        feats.append(f)
        labels.extend(int(r["label"]) for r in rows)
    return np.concatenate(feats), np.array(labels), manifest


def main():
    args = parse_args()

    Xs, ys = [], []
    for d in args.features:
        X, y, manifest = load_feature_dir(d)
        print(f"[load] {d}: {X.shape[0]} samples x {X.shape[1]} dims "
              f"(model={manifest.get('model')}, split={manifest.get('split')})")
        Xs.append(X)
        ys.append(y)
    X = np.concatenate(Xs)
    y_raw = np.concatenate(ys)

    # Binarize: keep only the two target classes, map to 0=real / 1=AI
    mask = np.isin(y_raw, [args.real_label, args.ai_label])
    X = X[mask]
    y = (y_raw[mask] == args.ai_label).astype(int)
    n_real, n_ai = int((y == 0).sum()), int((y == 1).sum())
    print(f"[data] usable: {len(y)}  (real={n_real}, ai={n_ai})")
    if n_real == 0 or n_ai == 0:
        raise SystemExit("One class is empty -- check --real-label/--ai-label.")

    # Stratified train / val / test split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=args.val_size + args.test_size,
        stratify=y, random_state=args.seed)
    rel_test = args.test_size / (args.val_size + args.test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_test, stratify=y_tmp, random_state=args.seed)
    print(f"[split] train={len(y_train)} val={len(y_val)} test={len(y_test)}")

    clf = LogisticRegression(C=args.C, max_iter=2000)
    print("[train] fitting logistic regression ...")
    clf.fit(X_train, y_train)

    def report(Xs_, ys_, name):
        probs = clf.predict_proba(Xs_)[:, 1]
        preds = (probs >= args.threshold).astype(int)
        acc = accuracy_score(ys_, preds)
        print(f"[{name}] acc={acc:.4f}  n={len(ys_)}  (threshold={args.threshold})")
        return acc

    report(X_train, y_train, "train")
    report(X_val, y_val, "val")
    test_acc = report(X_test, y_test, "test")

    print("\n[test] confusion matrix (rows=true, cols=pred; 0=real, 1=AI):")
    print(confusion_matrix(y_test, (clf.predict_proba(X_test)[:, 1] >= args.threshold).astype(int)))
    print("\n[test] per-class metrics:")
    print(classification_report(y_test,
                                (clf.predict_proba(X_test)[:, 1] >= args.threshold).astype(int),
                                target_names=["real", "ai"], digits=4))

    # Threshold sweep on val -- pick once, then freeze for all robustness evals
    print("[val] accuracy vs threshold (choose deliberately, then freeze):")
    val_probs = clf.predict_proba(X_val)[:, 1]
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        acc_t = accuracy_score(y_val, (val_probs >= t).astype(int))
        print(f"    t={t:.1f}  acc={acc_t:.4f}")

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
        "seed": args.seed,
        "test_acc_at_threshold": float(test_acc),
    }
    (out / "probe_config.json").write_text(json.dumps(config, indent=2))
    print(f"\n[saved] {out/'probe.joblib'}")
    print(f"[saved] {out/'probe_config.json'}")
    print("[next] feed these probabilities into the transform-table eval "
          "harness, using the SAME frozen threshold for every transform.")


if __name__ == "__main__":
    main()
