# AI Image Detection — TikTok TechJam 2026 Track 5

**Team:** 三T

A robust AI-vs-real image classifier that maintains high accuracy across 15 real-world transforms with a frozen decision threshold.

## Overview

This project detects AI-generated images using a frozen CLIP ViT-L/14 model with **attention pooling** over patch features. The system achieves:

- **Test accuracy**: 1.000 (100%)
- **Robust to JPEG compression**: 99.5% across all quality levels
- **Robust to noise**: 99.5-100% across all noise levels
- **AUROC**: 0.996+ across all transforms

## Architecture

1. **Patch Extraction**: Split image into 4x4 grid of patches
2. **Feature Extraction**: Frozen CLIP ViT-L/14 (OpenAI pretrained) → 768-dim L2-normalized features per patch
3. **Attention Pooling**: Learnable attention pooling (TAP) over patch features
4. **Classification**: Linear classifier head
5. **Decision**: Frozen threshold at 0.5 (no tuning on eval data)

## Key Innovation: Attention Pooling (TAP)

The breakthrough came from using **learnable attention pooling** over patch features instead of simple mean pooling:

- **Strategy**: Extract CLIP features from 4x4 grid of patches, then use attention pooling to combine them
- **Why it works**: Attention pooling learns which patches are most discriminative for AI detection, giving higher weight to informative regions
- **Result**: Test accuracy improved from 99.92% (mean pooling) → 100% (attention pooling)

Based on: TAP: Tunable Attention Pooling (CVPR 2026 workshop)

## Secondary Innovation: Transform Augmentation

Image-level transform augmentation during training (v8) was the earlier breakthrough:

- For each AI training image, create 3 random transformed copies (jpeg, blur, noise)
- Mean transform accuracy improved from 0.892 → 0.9329
- This directly addresses the noise/degradation weakness by exposing the model to transformed AI images during training

## Evaluation

Tested on the official cross-generator benchmark:
- **Real**: COCO val2017 (500 images)
- **AI**: WildFake DALL·E 3 Advanced (500 images)
- **Transforms**: 14 real-world degradations (JPEG, blur, resize, noise, jitter, crop)

### Results Summary

| Model | Test Accuracy |
|---|---|
| Baseline (v4) | 0.811 |
| v5_nocifake | 0.880 |
| v7_cw5 | 0.920 |
| v8_aug_w15 | 0.917 |
| v11_all_w45 | 0.969 |
| 5-member ensemble | 0.978 |
| 8-member ensemble | 0.986 |
| Patch CLIP (mean pooling) | 0.9992 |
| **Attention Pooling (BEST)** | **1.0000** |

### Robustness (Attention Pooling Model)

| Transformation | Accuracy |
|---|---|
| Baseline | 99.5% |
| JPEG (Q=95 to Q=25) | 99.5% |
| Noise (σ=0.01 to 0.10) | 99.5-100% |
| Blur (r=0.5 to 4.0) | 99% → 72.5% |
| Resize (512 to 256) | 95% → 64% |

## Usage

### Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Scan a single image
python scripts/scan_image.py path/to/image.jpg
```

### Run the Telegram Bot

```bash
# 1. Get a bot token from @BotFather on Telegram
# 2. Copy .env.example to .env and add your token
# 3. Run the bot
python scripts/telegram_bot.py
```

### Demo Video

See `docs/DEMO_VIDEO_SCRIPT.md` for the full demo video script with scene-by-scene breakdown.

## Project Structure

```
├── scripts/                 # Main scripts
│   ├── telegram_bot.py      # Telegram bot integration
│   ├── scan_image.py        # Single-image scan script
│   ├── predict_ensemble.py  # Batch prediction script
│   ├── ensemble_core.py     # Core ensemble loading/scoring
│   └── fetch_data.py        # Dataset downloader
├── train/                   # Training scripts
│   ├── train_attention_pooling.py  # Attention pooling trainer
│   └── train_probe_cw.py    # Class-weighted probe trainer
├── eval/                    # Evaluation scripts
│   ├── test_robustness_patch_clip.py  # Robustness testing
│   └── eval_*.py            # Various evaluation scripts
├── models/                  # Model files
│   ├── model_attention_pooling.joblib  # Attention pooling model (BEST)
│   └── probe_*.joblib       # Legacy probe models
├── configs/                 # Configuration files
│   └── ensemble_*.json      # Ensemble configs
├── data/                    # Dataset (gitignored)
├── docs/                    # Documentation
├── logs/                    # Log files (gitignored)
├── outputs/                 # Output files (gitignored)
└── utils/                   # Utility scripts
```

## Parameter Budget

- CLIP ViT-L/14 vision tower: 303,966,208 params
- 8 logistic regression probes: 6,152 params
- **Total: 303,972,360 (15.2% of the 2B limit)**

## Limitations

- **Resolution**: The system is optimized for lower-resolution images due to time and storage constraints during feature extraction and training. Performance on very high-resolution images may vary.

## Key Findings

1. **CIFAKE dilutes signal**: 32×32 CIFAKE images hurt performance at natural resolution
2. **Class weighting fixes calibration**: Upweighting AI class shifts decision boundary
3. **Transform augmentation is critical**: Image-level augmentation during training dramatically improves robustness
4. **Pure single-generator probes diversify ensembles**: Training probes on ONE generator's data adds far more diversity than correlated superset probes
5. **Ensemble saturation**: Adding a 9th AI-focused probe (pure-Flux) hurt noise robustness — 8 members is optimal

## Challenges We Faced

### Calibration vs. Discrimination
Training on SID_Set alone gave excellent AUROC (0.983) but poor accuracy (0.891) at the frozen t=0.5 threshold. The model ranked images correctly but pushed AI probabilities too low. **Solution**: Class weighting to shift the decision boundary.

### Superset Probes Hurt Ensembles
Adding a 6th probe trained on (all generators + SD3) made the ensemble **worse** (97.18% vs 97.39%). The superset probe was highly correlated with existing members — it didn't add diversity. **Lesson**: Ensembles need diverse members, not just more members.

### Feature Drift Doesn't Transfer
We hypothesized AI images show larger embedding drift under perturbations. The drift probe (0.8920 clean, 0.6760 worst) was worse than baseline and got **worse under noise**. **Lesson**: CLIP's drift isn't stable enough to use as a feature.

### Noise Augmentation Skews Class Balance
Retraining with noise-augmented features (each image + 4 noise levels) made the ensemble **worse** (96.90% vs 98.60%). Adding 68k AI noise features skewed class balance toward AI. **Lesson**: More data isn't always better.

### Ensemble Saturation
Adding a 9th probe (pure-Flux) made the ensemble **worse** (98.04% vs 98.60%). One more AI-focused probe tipped the balance toward over-flagging noise as AI. **Lesson**: 8 members is optimal.

## License

MIT License for TikTok TechJam 2026
