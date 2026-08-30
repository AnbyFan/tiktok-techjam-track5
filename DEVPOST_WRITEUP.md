# AI Image Detection Under Real-World Transformations

## TikTok TechJam 2026 — Track 5 Submission

## Overview

We built a robust AI-vs-real image classifier that maintains **97.19% mean accuracy** across 15 real-world transforms (JPEG compression, Gaussian blur, resize, noise, color jitter, crop) using a frozen decision threshold.

## Architecture

```
Input Image → CLIP ViT-L/14 (frozen) → 768-dim features → Logistic Regression Probe → P(AI)
```

- **Backbone**: OpenAI CLIP ViT-L/14 (~427M params, well under 2B limit)
- **Probe**: Logistic regression with class weighting (ai_weight=4.5)
- **Threshold**: Frozen at 0.5 (no tuning on eval data)
- **Training data**: 3 sources combined — SID_Set + ComfyUI + Flux (5 augmented copies per AI image)

## Key Innovations

### 1. Transform Augmentation

Image-level transform augmentation during training — for each AI training image, create 5 random transformed copies matching the competition's spec (JPEG, blur, resize, noise, jitter, crop).

### 2. Multi-Source Training Data

Combining three independent AI image sources (SID_Set, ComfyUI, Flux) with higher class weighting (ai_weight=4.5) significantly improved generalization.

**Results (v4 baseline → v11 final)**:
- Mean transform accuracy: 0.75 → **0.9719** (+0.22)
- Clean accuracy: 0.811 → **0.965** (+0.154)
- Worst-case transform: 0.73 → **0.959** (+0.229)
- Clean AUROC: ~0.78 → **0.996** (+0.216)

## Tools & Libraries

| Component | Version | Purpose |
|---|---|---|
| PyTorch | 2.13.0+cu126 | Deep learning framework |
| OpenCLIP | 3.3.0 | CLIP model implementation |
| scikit-learn | 1.7.2 | Logistic regression probe |
| Hugging Face Datasets | 2.19+ | Data streaming |
| Pillow | 10.0+ | Image preprocessing |
| NumPy | 1.26+ | Numerical operations |
| tqdm | 4.66+ | Progress bars |

## Datasets

### Training Data
- **SID_Set** (Hugging Face): 8,000 images (4,000 real + 4,000 AI)
  - Real: Natural images from COCO, ImageNet
  - AI: Diffusion model outputs (Stable Diffusion, Midjourney, DALL·E)
- **ComfyUI** (own generator): 8,000 AI images (5 augmented copies each = 40,000 features)
- **Flux_AIGC_Dataset** (Hugging Face): 3,000 AI images (Flux 1 Schnell outputs)

Total training features: ~60,000 (with augmentation)

### Evaluation Data (Never used in training)
- **COCO val2017**: 5,000 real images
- **WildFake DALL·E 3 Advanced**: 8,843 AI images (500 used for benchmark)
- **ComfyUI/Flux**: 2,000 held-out AI images (own-generator test)

## Evaluation Protocol

### 15 Transforms Tested
1. Clean (baseline)
2. JPEG q90, q70, q50, q30
3. Gaussian blur σ0.5, σ1.0, σ2.0
4. Resize 0.5×, 0.25× (then upscale)
5. Gaussian noise σ0.02, σ0.05, σ0.10
6. Color jitter ±20%
7. Center crop 80%

### Results (DALL·E 3 benchmark, t=0.5 frozen)

| Metric | Value |
|---|---|
| Clean accuracy | 0.965 |
| Mean transform accuracy | **0.9719** |
| Worst transform | 0.959 (crop_80pct) |
| Clean AUROC | 0.996 |

## Iteration History

| Version | Approach | Clean | Mean-Xform | Worst |
|---|---|---|---|---|
| v4 | Baseline (mixed data) | 0.811 | ~0.75 | ~0.73 |
| v5 | Drop CIFAKE | 0.880 | 0.867 | 0.823 |
| v6 | SID_Set only | 0.891 | 0.850 | 0.735 |
| v7 | Class weighting (w=5) | 0.920 | 0.892 | 0.781 |
| v8 | Transform augmentation (w=1.5) | 0.917 | 0.9329 | 0.919 |
| v9 | Stratified augmentation | 0.883 | 0.8956 | 0.872 |
| v10 | Combo5 + w3.5 | 0.959 | 0.9632 | 0.945 |
| **v11** | **All features + w4.5** | **0.965** | **0.9719** | **0.959** |

## Limitations

1. **Noise remains challenging**: While improved, noise transforms still cause the most errors.
2. **Compression ambiguity**: Heavy JPEG compression (q30-q50) affects both real and AI images.
3. **Threshold fixed**: Cannot adapt threshold per transform (competition rule).

## Reproduction

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Extract features (if not already done)
python extract_features_aug.py --input-dir data/sid_set --output-dir features/sid_set_aug
python extract_features_folder.py --input-dir data/comfy_train/ai --output-dir features/comfy_aug5 --augment --aug-copies 5
python extract_flux_features.py --output-dir features/flux --max-images 3000

# Train probe
python train_probe_cw.py --features features/sid_set_aug features/comfy_aug5 features/flux --out probe_v11_all_w45 --ai-weight 4.5

# Evaluate
python eval_robustness.py --real-dir data/val/real --ai-dir data/wildfake/.../DALLE3 --probe probe_v11_all_w45 --out reports/eval

# Predict on new images
python predict.py --input-dir path/to/images --probe probe_v11_all_w45 --out predictions.json
```

## Team

[Your names here]

## Demo Video

[Link to YouTube video here]
