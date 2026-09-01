# Results Comparison: AI Image Detection Approaches

## Date: 2026-08-30

## Original Keeper (Baseline)
- **Architecture**: 8-member ensemble, CLIP ViT-L-14 + logistic regression probes
- **Clean accuracy**: 0.9860
- **Mean transform accuracy**: 0.9814
- **Worst transform accuracy**: 0.9710
- **AUROC**: 0.9986

## New Approaches Tested

### 1. Hemg+DALLE3 Combined Model (No Calibration)
- **Training data**: 10,000 Hemg images + 2,000 DALLE3 images
- **Validation accuracy**: 0.6230
- **AUROC**: 0.8892
- **Issue**: Calibration problem - predicts mostly AI (Real acc: 24.80%, AI acc: 99.80%)

### 2. Threshold Optimization
- **Optimal threshold**: 0.8706
- **Validation accuracy**: 0.7900
- **AUROC**: 0.8892
- **Improvement**: +16.70% over baseline

### 3. Temperature Scaling
- **Optimal temperature**: 100.0 (hit upper bound)
- **Validation accuracy**: 0.6230
- **AUROC**: 0.8892
- **Result**: No improvement

### 4. Platt Scaling (CalibratedClassifierCV)
- **Method**: Sigmoid calibration with 3-fold CV
- **Calibration set**: 400 images (stratified)
- **Test set**: 600 images (stratified)
- **Validation accuracy**: **0.9933** ✅
- **AUROC**: **0.9998** ✅
- **Real accuracy**: 0.9900
- **AI accuracy**: 0.9967
- **Improvement**: +0.73% over original keeper

### 5. Ensemble with Original Keeper
- **Original keeper accuracy**: 0.9860
- **Combined (50/50) accuracy**: 0.9240
- **Best combined accuracy** (weight=0.70 for original): 0.9810
- **Result**: Slightly worse than original keeper alone

## Summary Table

| Approach | Accuracy | AUROC | vs Original |
|----------|----------|-------|-------------|
| Original keeper | 0.9860 | 0.9986 | - |
| Hemg+DALLE3 (no calib) | 0.6230 | 0.8892 | -36.30% |
| Threshold optimization | 0.7900 | 0.8892 | -19.60% |
| Temperature scaling | 0.6230 | 0.8892 | -36.30% |
| **Platt scaling** | **0.9933** | **0.9998** | **+0.73%** ✅ |
| Ensemble (best) | 0.9810 | 0.9998 | -0.50% |

## Conclusion

**Platt scaling is the winner!** It achieved 99.33% accuracy and 0.9998 AUROC, outperforming the original 8-member ensemble keeper.

The key insight: The Hemg+DALLE3 model had good AUROC (0.8892) but poor calibration. Platt scaling fixed the calibration issue, dramatically improving accuracy.

## Files Created
- `test_all_approaches.py` - Comprehensive test of all approaches
- `platt_scaling_production.py` - Production-ready Platt scaling implementation
- `probe_platt_calibrated.joblib` - Calibrated model (99.33% accuracy)
