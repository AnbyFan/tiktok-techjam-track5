# AI Image Detection — TikTok TechJam 2026 Track 5

A robust AI-vs-real image classifier that maintains high accuracy across 15 real-world transforms with a frozen decision threshold.

## Overview

This project detects AI-generated images using a frozen CLIP ViT-L/14 model and an **8-member ensemble** of lightweight logistic regression probes. The system achieves:

- **Clean accuracy**: 0.986 (98.6%)
- **Mean transform accuracy**: 0.9814 (98.14%)
- **Worst-case transform**: 0.9710 (noise_s0.05 / noise_s0.1)
- **AUROC**: 0.996+ across all transforms

## Architecture

1. **Feature Extraction**: Frozen CLIP ViT-L/14 (OpenAI pretrained) → 768-dim L2-normalized features (single forward pass shared by all probes)
2. **Classification**: 8-member ensemble of class-weighted logistic regression probes
3. **Ensemble**: Weighted-average P(AI) across members
4. **Decision**: Frozen threshold at 0.5 (no tuning on eval data)

## Key Innovation: Pure Single-Generator Probes

The breakthrough came from ensemble diversity via **pure single-generator probes**:

- **Strategy**: Train individual probes on ONE generator's AI images + real negatives only
- **Why it works**: Pure probes learn generator-specific signatures that diversify the ensemble far more than correlated "all-generators" superset probes
- **Result**: Mean transform accuracy improved from 0.9739 → 0.9814 (+0.0075); worst-case from 0.9680 → 0.9710 (+0.003)

The ensemble combines:
- 3 "all-generators" probes (different class weights: w4.0, w4.5, w5.0)
- 2 "all-generators + extra dataset" probes (Sana, Midjourney)
- 3 **pure single-generator** probes (SD3, Midjourney, Sana)

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

| Model | Clean | Mean-Transform | Worst |
|---|---|---|---|
| Baseline (v4) | 0.811 | ~0.75 | ~0.73 |
| v5_nocifake | 0.880 | 0.867 | 0.823 |
| v7_cw5 | 0.920 | 0.892 | 0.781 |
| v8_aug_w15 | 0.917 | 0.9329 | 0.919 |
| v11_all_w45 | 0.969 | 0.9719 | 0.959 |
| 5-member ensemble | 0.978 | 0.9739 | 0.968 |
| **8-member ensemble (BEST)** | **0.986** | **0.9814** | **0.9710** |

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
python scan_image.py path/to/image.jpg
```

### Run the Telegram Bot

```bash
# 1. Get a bot token from @BotFather on Telegram
# 2. Copy .env.example to .env and add your token
# 3. Run the bot
python telegram_bot.py
```

## Project Structure

```
├── ensemble_config.json     # Ensemble member config (8 members)
├── probes/                  # Ensemble member probes (8+ dirs)
├── reports/                 # Evaluation reports
├── features_freq/           # Frequency-domain features
├── extract/                 # Feature extraction scripts
├── train/                   # Training scripts
├── eval/                    # Evaluation scripts
├── utils/                   # Utility scripts
├── ensemble_core.py         # Core ensemble loading/scoring
├── predict_ensemble.py      # Batch prediction script
├── scan_image.py            # Single-image scan script
├── telegram_bot.py          # Telegram bot integration
├── train_probe_cw.py        # Class-weighted probe trainer
└── fetch_data.py            # Dataset downloader
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

## License

MIT License for TikTok TechJam 2026
