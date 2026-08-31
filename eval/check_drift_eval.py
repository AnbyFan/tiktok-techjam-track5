#!/usr/bin/env python3
"""
Quick analysis: does feature-drift separate AI from real on the EVAL set
(clean transform only)? Analysis only -- no training.

Usage:
    python check_drift_eval.py \
        --real-dir data/val/real \
        --ai-dir data/wildfake/Images/Diffusion_based/DALLE/DALLE/Advanced/DALLE3/dalle3 \
        --max-per-class 300 --drift-k 4 --drift-sigma 0.2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

from drift import add_drift_features

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real-dir", required=True)
    p.add_argument("--ai-dir", required=True)
    p.add_argument("--max-per-class", type=int, default=300)
    p.add_argument("--drift-k", type=int, default=4)
    p.add_argument("--drift-sigma", type=float, default=None,
                   help="single sigma; if omitted, sweeps a default set")
    p.add_argument("--model", default="ViT-L-14")
    p.add_argument("--pretrained", default="openai")
    return p.parse_args()


def list_images(d, limit):
    paths = sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in EXTS)
    return paths[:limit] if limit else paths


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device)
    model.eval()

    items = ([(p, 0) for p in list_images(args.real_dir, args.max_per_class)] +
             [(p, 1) for p in list_images(args.ai_dir, args.max_per_class)])
    print(f"[data] real={sum(1 for _, l in items if l == 0)} "
          f"ai={sum(1 for _, l in items if l == 1)}")

    tensors, labels = [], []
    for path, label in items:
        img = Image.open(path).convert("RGB")
        tensors.append(preprocess(img))
        labels.append(label)
    labels = np.array(labels)
    t = torch.stack(tensors)

    with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.float16, enabled=(device == "cuda")):
        f = model.encode_image(t.to(device, non_blocking=True))
        f = f / f.norm(dim=-1, keepdim=True)
    f_clean = f.float().cpu().numpy()

    sigmas = ([args.drift_sigma] if args.drift_sigma is not None
              else [0.05, 0.1, 0.2, 0.3, 0.5, 0.8])
    from sklearn.metrics import roc_auc_score
    for sigma in sigmas:
        feats = add_drift_features(model, f_clean, t, args.drift_k,
                                   sigma, device, rng=np.random.default_rng(42))
        dm, ds = feats[:, 768], feats[:, 769]
        auroc_dm = roc_auc_score(labels, dm)   # >0.5 => higher drift = more AI
        auroc_ds = roc_auc_score(labels, ds)
        print(f"sigma={sigma:4.2f}  drift_mean: real={dm[labels==0].mean():.5f} "
              f"ai={dm[labels==1].mean():.5f} diff={dm[labels==1].mean()-dm[labels==0].mean():+.5f} "
              f"AUROC={auroc_dm:.4f}  |  drift_std: real={ds[labels==0].mean():.5f} "
              f"ai={ds[labels==1].mean():.5f} AUROC={auroc_ds:.4f}")


if __name__ == "__main__":
    main()
