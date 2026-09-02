#!/usr/bin/env python3
"""
Ensemble prediction script (Track 5) -- combines N probes for robustness.

Loads one shared CLIP backbone plus N linear probes from a JSON config and
reports a weighted-average P(AI) per image. All members share the backbone, so
a single forward pass serves every probe.

Usage:
    # Default config (ensemble_config.json)
    python predict_ensemble.py --input-dir path/to/images --out predictions.json

    # Custom config
    python predict_ensemble.py --input-dir path/to/images \
        --config ensemble_config.json --out predictions.json
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from ensemble_core import EnsembleScorer

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--config", default="ensemble_config.json",
                   help="JSON config listing member probes + weights")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out", default="predictions_ensemble.json")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    paths = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in EXTS)
    print(f"[data] {len(paths)} images")

    scorer = EnsembleScorer(args.config)
    print(f"[ensemble] {len(scorer.probes)} members: "
          f"{[n for n, _ in scorer.probes]}  (threshold={scorer.threshold})")

    results = []
    batches = [paths[i:i + args.batch_size]
               for i in range(0, len(paths), args.batch_size)]
    for batch in tqdm(batches, desc="scoring", unit="batch"):
        tensors, rels = [], []
        for path in batch:
            try:
                img = Image.open(path).convert("RGB")
                tensors.append(scorer.preprocess(img))
                rels.append(str(path.relative_to(input_dir)))
            except Exception:
                continue
        if not tensors:
            continue
        probs = scorer.score_tensors(torch.stack(tensors))
        for rel, prob in zip(rels, probs):
            results.append({"image_path": rel, "pred": round(float(prob), 4)})

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2))
    n_ai = sum(1 for r in results if r["pred"] >= scorer.threshold)
    print(f"\n[done] scored={len(results)}")
    print(f"[summary] flagged AI at t={scorer.threshold}: "
          f"{n_ai}/{len(results)} ({100 * n_ai / max(len(results), 1):.1f}%)")


if __name__ == "__main__":
    main()
