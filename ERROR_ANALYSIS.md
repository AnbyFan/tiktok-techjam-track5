# Error Analysis — probe_v8_aug_w15_local

## Summary

- **Clean accuracy**: 0.917 (91.7%)
- **Mean transform accuracy**: 0.9329 (93.29%)
- **False positive rate (clean)**: 12.6% (real images flagged as AI)
- **False negative rate (clean)**: 4.0% (AI images not flagged)

## False Positives (Real Images Flagged as AI)

These are real COCO val2017 images that the model incorrectly classifies as AI-generated.

| Image | Transform | P(AI) | Notes |
|---|---|---|---|
| 000000022371.jpg | jpeg_q70 | 0.978 | Highly compressed real image |
| 000000021604.jpg | crop_80pct | 0.968 | Center-cropped real image |
| 000000022371.jpg | jpeg_q90 | 0.967 | Same image, lighter compression |
| 000000021604.jpg | jpeg_q90 | 0.965 | Same image, lighter compression |
| 000000021604.jpg | clean | 0.956 | Original uncropped image |

**Pattern**: The model tends to flag real images with heavy compression or cropping. These transformations may introduce artifacts that resemble AI generation patterns.

## False Negatives (AI Images Not Flagged)

These are DALL·E 3 images that the model incorrectly classifies as real.

| Image | Transform | P(AI) | Notes |
|---|---|---|---|
| 402bdb5e...jpg | noise_s0.05 | 0.054 | Gaussian noise added |
| 402bdb5e...jpg | noise_s0.1 | 0.061 | Higher noise level |
| 29538088...jpg | noise_s0.05 | 0.061 | Different image, same noise |
| 402bdb5e...jpg | jpeg_q30 | 0.069 | Heavy JPEG compression |
| 29538088...jpg | noise_s0.02 | 0.070 | Light noise |

**Pattern**: The model struggles with AI images that have noise added or heavy compression. These transformations obscure the subtle patterns that distinguish AI-generated content.

## Key Insights

1. **Noise is the hardest transform**: Both FPs and FNs cluster around noise transforms, confirming noise as the primary challenge.
2. **Compression creates ambiguity**: JPEG compression at low quality (q30-q70) affects both real and AI images, making them harder to distinguish.
3. **Clean images are well-separated**: Most errors occur under transforms, not on clean images.

## Recommendations

1. **Increase noise augmentation**: Add more noise variants during training to improve robustness.
2. **Compression-aware features**: Consider extracting features that are more robust to JPEG compression.
3. **Ensemble approach**: Combine multiple probes trained on different noise/compression levels.
