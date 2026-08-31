#!/usr/bin/env python3
"""Extract CLIP features from local SANA images folder."""
import json
from pathlib import Path
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def main():
    input_dir = Path("Y:/qwencode/sana_1k/images")
    out_dir = Path("features/sana")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    model.eval()
    print(f"[init] CLIP on {device}")

    paths = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in EXTS)
    print(f"[init] {len(paths)} images")

    feats = []
    batch_imgs = []
    batch_paths = []

    def encode(imgs):
        with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=(device == "cuda")):
            t = torch.stack(imgs).to(device, non_blocking=True)
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()

    for path in tqdm(paths, desc="extracting", unit="img"):
        try:
            img = Image.open(path).convert("RGB")
            batch_imgs.append(preprocess(img))
            batch_paths.append(f"sana_{len(feats)+len(batch_imgs):06d}")
        except Exception as e:
            print(f"[warn] {path}: {e}")
            continue

        if len(batch_imgs) >= 64:
            feats.append(encode(batch_imgs))
            batch_imgs.clear()

    if batch_imgs:
        feats.append(encode(batch_imgs))

    all_feats = np.concatenate(feats).astype(np.float32)
    np.save(out_dir / "features_00000.npy", all_feats)

    with (out_dir / "meta_00000.csv").open("w") as f:
        f.write("img_id,label,dataset,transform\n")
        for i in range(len(all_feats)):
            f.write(f"sana_{i:06d},1,sana,clean\n")

    manifest = {
        "counts": {"1": len(all_feats)},
        "shards": ["features_00000.npy"],
        "dataset": "sana",
        "split": "train",
        "model": "ViT-L-14",
        "pretrained": "openai",
        "total_images": len(all_feats),
        "feature_dim": 768
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[done] {len(all_feats)} images, shape={all_feats.shape}")

if __name__ == "__main__":
    main()
