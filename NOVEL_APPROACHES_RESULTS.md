# Novel Approaches Results

## Date: 2026-08-30

## Evaluation Setup
- **Train set**: 200 real + 200 AI images
- **Test set**: 200 real + 200 AI images
- **Total**: 400 real + 400 AI images

## Results Summary

| Approach | Test Accuracy | Notes |
|----------|---------------|-------|
| Standard CLIP + Platt Scaling | 0.9925 | Baseline |
| NPR Features | 0.6325 | Poor performance |
| **Patch-based CLIP** | **1.0000** | **Winner!** |
| Data Augmentation | 0.9875 | Slightly worse than baseline |
| CLIP + NPR Ensemble | 0.6375 | Poor performance |

## Detailed Findings

### 1. Standard CLIP + Platt Scaling (Baseline)
- **Accuracy**: 99.25%
- Uses frozen CLIP ViT-L-14 features with logistic regression probe
- Platt scaling calibration for improved probability estimates

### 2. NPR (Neighboring Pixel Relationships)
- **Accuracy**: 63.25%
- Extracts pixel-difference based features
- **Result**: Poor performance, not recommended

### 3. Patch-based CLIP Features ⭐
- **Accuracy**: 100.00%
- Extracts CLIP features from 4x4 grid of patches
- Aggregates using mean pooling
- **Result**: Perfect accuracy on test set!
- **Why it works**: Catches localized artifacts that get diluted in global embeddings

### 4. Data Augmentation
- **Accuracy**: 98.75%
- Applied JPEG compression, Gaussian blur, and noise during training
- **Result**: Slightly worse than baseline (may need more tuning)

### 5. CLIP + NPR Ensemble
- **Accuracy**: 63.75%
- Combined CLIP and NPR probabilities with 50/50 weight
- **Result**: Poor performance due to weak NPR features

## Conclusion

**Patch-based CLIP features are the winner!** Achieved 100% test accuracy, outperforming the standard CLIP + Platt scaling baseline (99.25%).

The key insight: Extracting CLIP features from multiple patches instead of one global embedding allows the model to detect localized artifacts that would otherwise be diluted.

## Next Steps
1. Implement patch-based CLIP in production pipeline
2. Test on robustness transforms (compression, noise, etc.)
3. Consider combining with Platt scaling for even better calibration

## Files Created
- `test_novel_approaches.py` - Initial testing script
- `eval_novel_approaches.py` - Proper evaluation with train/test split
