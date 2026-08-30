# AI Image Detection — TikTok TechJam 2026 Track 5

A robust AI-vs-real image classifier that maintains high accuracy across 15 real-world transforms with a frozen decision threshold.

## Overview

This project detects AI-generated images using a frozen CLIP ViT-L/14 model and a lightweight logistic regression probe. The system achieves:

- **Clean accuracy**: 0.917 (91.7%)
- **Mean transform accuracy**: 0.9329 (93.29%)
- **Worst-case transform**: 0.919 (resize_0.5x)
- **AUROC**: 0.985+ across all transforms

## Architecture

1. **Feature Extraction**: Frozen CLIP ViT-L/14 (OpenAI pretrained) → 768-dim L2-normalized features
2. **Classification**: Logistic regression probe with class weighting
3. **Decision**: Frozen threshold at 0.5 (no tuning on eval data)

## Key Innovation: Transform Augmentation

The breakthrough came from image-level transform augmentation during training:

- **v8 approach**: For each AI training image, create 3 random transformed copies (jpeg_q50, blur_s1.0, noise_s0.05)
- **Result**: Mean transform accuracy improved from 0.892 → 0.9329 (+0.041)
- **Worst-case**: Improved from 0.781 → 0.919 (+0.138)

This directly addresses the noise/degradation weakness by exposing the model to transformed AI images during training.

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
| **v8_aug_w15 (BEST)** | 0.917 | **0.9329** | **0.919** |
| v9_stratified | 0.883 | 0.8956 | 0.872 |

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
# 1. Extract features (if not already done)
python extract_features_aug.py --input-dir data/sid_set --output-dir features/sid_set_aug

# 2. Train probe
python train_probe_cw.py --features features/sid_set_aug --out probe_v8_aug_w15_local --ai-weight 1.5

# 3. Evaluate robustness
python eval_robustness.py --real-dir data/val/real --ai-dir data/val/ai --probe probe_v8_aug_w15_local --out reports/eval_results
```

### Predict on New Images

```bash
python predict.py --probe probe_v8_aug_w15_local --image path/to/image.jpg
```

## Project Structure

```
├── data/                    # Datasets (real + AI images)
├── features/                # Extracted CLIP features
├── probe_v8_aug_w15/       # Best model (locked)
├── reports/                 # Evaluation results
├── eval_robustness.py      # Robustness evaluation harness
├── train_probe_cw.py       # Class-weighted probe trainer
├── extract_features_aug.py # Feature extraction with augmentation
└── JOURNAL.md              # Iteration history and findings
```

## Key Findings

1. **CIFAKE dilutes signal**: 32×32 CIFAKE images hurt performance at natural resolution
2. **Class weighting fixes calibration**: Upweighting AI class shifts decision boundary
3. **Transform augmentation is critical**: Image-level augmentation during training dramatically improves robustness
4. **Random > Stratified**: Random transform sampling outperforms deterministic stratified augmentation

## License

MIT License for TikTok TechJam 2026
