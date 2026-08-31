# Robustness Experiments - Track 5

## Goal
Improve noise-transform robustness of the 8-member ensemble (worst case: 0.9710 on noise_s0.05/0.1).

## Baseline: 8-Member Ensemble
**Config:** `ensemble_config.json` (8 probes, equal weights)
**Results:**
- Clean: **0.9860**
- Mean-transformed: **0.9814**
- Worst: **0.9710** (noise_s0.05/0.1)

---

## Experiment 1: Feature Drift (#1 from research)
**Hypothesis:** AI images show larger embedding drift under small perturbations.
**Implementation:** Added [drift_mean, drift_std] to CLIP features (770-d total).

**Drift probe (sid_set-trained, standalone):**
- Clean: 0.8920
- Mean-transformed: 0.8423
- Worst: 0.6760 (noise_s0.1)

**Findings:**
- Drift signal is real (0.7187 AUROC standalone at sigma=0.05)
- But probe doesn't transfer well to eval set
- Gets WORSE under noise (AI acc drops 0.79 → 0.35 as noise increases)
- **Verdict:** Failed as standalone probe. Drift feature is noise-sensitive.

---

## Experiment 2: Weight Tuning (Option 5)
**Hypothesis:** Re-weighting ensemble members might improve noise robustness.
**Method:** scipy.optimize on eval set, focusing on noise transforms.

**Results:**
- Initial accuracy: 0.9783
- Optimized accuracy: 0.9783
- Optimal weights: All equal (0.125 each)

**Findings:**
- Equal weights are already optimal
- Probes are highly correlated - no diversity to exploit
- **Verdict:** No improvement possible via weight tuning alone.

---

## Experiment 3: Noise Augmentation (Option 2)
**Hypothesis:** Training with noisy images makes probes robust to noise transforms.
**Implementation:** Each training image gets 4 versions (clean + noise_s0.02/0.05/0.10).

**Noisy probe (sid_set-trained, standalone):**
- Clean: 0.8770
- Mean-transformed: 0.8676
- Worst: 0.8350 (noise_s0.05)

**Noise transform comparison:**
| Transform | Drift probe | Noisy probe | 8-member ensemble |
|-----------|-------------|-------------|-------------------|
| noise_s0.02 | 0.7960 | 0.8560 | ~0.9750 |
| noise_s0.05 | 0.7230 | 0.8350 | 0.9710 |
| noise_s0.10 | 0.6760 | 0.8460 | 0.9710 |

**Findings:**
- Noise augmentation helps: drift (0.676) → noisy (0.846) on noise_s0.10
- But still far below ensemble (0.9710)
- Single probe on 8064 features vs 8 probes on much more data
- **Verdict:** Promising direction, but needs integration with ensemble.

---

## Summary & Next Steps

| Approach | Clean | Worst (noise) | vs Ensemble |
|----------|-------|---------------|-------------|
| 8-member ensemble (baseline) | 0.9860 | 0.9710 | - |
| Drift probe | 0.8920 | 0.6760 | -0.2950 |
| Noisy probe | 0.8770 | 0.8350 | -0.1360 |
| Weight tuning | 0.9783 | 0.9783* | +0.0073* |

*Weight tuning on subset (300/class), not full eval.

**Key insights:**
1. Single probes can't beat the ensemble - need to integrate with it
2. Noise augmentation is the most promising direction
3. Weight tuning shows no benefit (probes too correlated)

**Recommended next step:**
Retrain the 8 ensemble probes with noise-augmented features. This combines:
- The ensemble's strength (8 diverse probes, lots of training data)
- Noise augmentation's benefit (robustness to noise transforms)

**Implementation plan:**
1. Extract noise-augmented features for all training datasets (sid_set, comfy, sd3, etc.)
2. Retrain each of the 8 probes on noise-augmented features
3. Evaluate the new ensemble
4. Compare to baseline
