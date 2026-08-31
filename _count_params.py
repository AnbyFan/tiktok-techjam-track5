import json
import torch
import open_clip
import joblib
import numpy as np

# 1. CLIP backbone
model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
total_clip = sum(p.numel() for p in model.parameters())
# Vision tower only (what encode_image actually uses)
vision_params = sum(p.numel() for p in model.visual.parameters())
text_params = total_clip - vision_params
print("=== CLIP ViT-L/14 (openai) ===")
print(f"  full model:      {total_clip:>12,}")
print(f"  vision tower:    {vision_params:>12,}   <- used for image detection")
print(f"  text tower:      {text_params:>12,}   <- not used")

# 2. Probes in the current ensemble
cfg = json.load(open("ensemble_config.json"))
print("\n=== Probes ===")
probe_total = 0
for m in cfg["members"]:
    probe = joblib.load(f"{m['probe']}/probe.joblib")
    coef = np.asarray(probe.coef_)
    intercept = np.asarray(probe.intercept_)
    n = coef.size + intercept.size
    probe_total += n
    print(f"  {m['probe']:<32} {n:>6,} params  (coef {coef.size} + bias {intercept.size})")

print(f"  {'TOTAL probes':<32} {probe_total:>6,} params")

# 3. Totals
print("\n=== Totals vs 2B limit ===")
print(f"  vision tower + probes: {vision_params + probe_total:>12,}  ({(vision_params+probe_total)/2e9*100:.3f}% of 2B)")
print(f"  full CLIP + probes:    {total_clip + probe_total:>12,}  ({(total_clip+probe_total)/2e9*100:.3f}% of 2B)")
print(f"  2B limit:              {2_000_000_000:>12,}")
