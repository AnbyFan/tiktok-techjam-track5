#!/usr/bin/env python3
"""
Track 5 submission scoring script.

Takes an image directory as input and writes a JSON file with a confidence
score per image — the likelihood that it is AIGC-generated — using the
frozen CLIP ViT-L/14 backbone + trained linear probe.

Usage:
    python predict.py --input-dir path/to/images --probe probe_v4 --out predictions.json

Output format (JSON list):
    [
      {"image_path": "subdir/img1.png", "pred": 0.9123},
      {"image_path": "img2.jpg",        "pred": 0.0431},
      ...
    ]

`pred` is the model's P(AI-generated) in [0, 1]; `image_path` is relative
to --input-dir. The frozen decision threshold lives in
<probe>/probe_config.json and is used only for the printed summary — the
JSON keeps raw scores so any threshold can be applied downstream.

Requires: torch, open_clip_torch, scikit-learn (joblib), numpy, Pillow, tqdm
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import open_clip
import joblib
from PIL import Image
from tqdm import tqdm

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, help="directory of images to score")
    p.add_argument("--probe", required=True,
                   help="dir with probe.joblib + probe_config.json (from train_probe.py)")
    p.add_argument("--out", default="predictions.json")
    p.add_argument("--model", default="ViT-L-14",
                   help="must match the model used for feature extraction")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"not a directory: {input_dir}")

    paths = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in EXTS)
    if not paths:
        raise SystemExit(f"no images ({'/'.join(sorted(EXTS))}) found under {input_dir}")
    print(f"[data] {len(paths)} images under {input_dir}")

    probe_dir = Path(args.probe)
    clf = joblib.load(probe_dir / "probe.joblib")
    cfg = json.loads((probe_dir / "probe_config.json").read_text())
    threshold = cfg.get("threshold", 0.5)
    print(f"[probe] {probe_dir/'probe.joblib'}  "
          f"(feature_dim={cfg.get('feature_dim')}, threshold={threshold})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device)
    model.eval()
    print(f"[init] {args.model} ({args.pretrained}) on {device}")

    def encode(imgs):
        with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=(device == "cuda")):
            t = torch.stack(imgs).to(device, non_blocking=True)
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()

    results = []
    skipped = 0
    batches = [paths[i:i + args.batch_size]
               for i in range(0, len(paths), args.batch_size)]
    for batch in tqdm(batches, desc="scoring", unit="batch"):
        tensors, rels = [], []
        for path in batch:
            try:
                img = Image.open(path).convert("RGB")
                tensors.append(preprocess(img))
                rels.append(str(path.relative_to(input_dir)))
            except Exception:
                skipped += 1
        if not tensors:
            continue
        probs = clf.predict_proba(encode(tensors))[:, 1]
        for rel, prob in zip(rels, probs):
            results.append({"image_path": rel, "pred": round(float(prob), 4)})

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2))

    n_ai = sum(1 for r in results if r["pred"] >= threshold)
    print(f"\n[done] scored={len(results)} skipped={skipped}")
    print(f"[summary] flagged AI at t={threshold}: {n_ai}/{len(results)} "
          f"({100 * n_ai / max(len(results), 1):.1f}%)")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
