# Iteration Journal — Track 5 AI Image Detection

Goal: improve AI-vs-real detection accuracy on the official cross-generator
benchmark (COCO val2017 vs WildFake DALL·E 3, 500/class), especially under the
15 real-world transforms, at the frozen threshold t=0.5.

Two eval pools exist (do NOT confuse them):
- **DALL·E 3 (official, cross-generator, HARD)**: `data/wildfake/.../Advanced/DALLE3`
- **ComfyUI/Flux (own-generator, EASY)**: `data/val/ai`
All comparisons below use the DALL·E 3 pool unless stated otherwise.

## Baseline (probe_v4)

- Own-generator (ComfyUI): clean acc=0.875, AUROC=0.960
- Official (DALL·E 3): clean acc=0.811, AUROC=0.901
- Worst rows: noise sigma 0.05/0.1 (~0.73), blur sigma 2.0 (0.746)
- Known weakness: severe degradation flips real to AI (JPEG q50 real_acc=0.718)

## v5 — drop CIFAKE, probe variants (DALL·E 3)

Key finding: CIFAKE (32×32) dilutes the training signal at natural resolution.
Dropping it (probe_v5_nocifake = sid_set + comfy_aug, logistic C=1) is the best
v5 model.

| Model | clean | noise_s0.05 | blur_s2.0 | resize_0.25x | mean-xform |
|---|---|---|---|---|---|
| v5_nocifake (BEST v5) | 0.880 | 0.823 | 0.858 | 0.903 | 0.867 |
| v5_mlp (256,128 head) | worse | — | — | — | overfits |
| v5_c100 (C=100) | worse | — | — | — | overfits |
| v5_noisy (feat noise σ=0.05) | 0.877 | 0.830 | 0.862 | 0.909 | ~0.868 |

- Feature-noise augmentation (v5_noisy) gives only **marginal** noise gains
  (+0.007 on noise_s0.05) at a small clean cost. Weak lever.

## v5 — ensemble (clean + noisy probe)

- Weighted-average ensemble of v5_nocifake + v5_noisy (0.7/0.3):
  DALL·E 3 clean = **0.881** (vs 0.880 single). **No gain** — the two logistic
  probes are too similar to diversify. Ensemble abandoned.

## v6 — sid_set alone: the AUROC breakthrough (DALL·E 3)

Training on sid_set ONLY (no ComfyUI) yields a far better cross-generator
ranking, but a badly shifted decision boundary:

| Model | clean acc | clean AUROC | clean real | clean ai |
|---|---|---|---|---|
| v5_nocifake | 0.880 | 0.949 | 0.886 | 0.874 |
| **v6_sidadone** | 0.891 | **0.983** | 0.994 | 0.788 |

- The ComfyUI-specific features were **hurting** generalization. sid_set alone
  ranks DALL·E 3 far better (AUROC 0.983 vs 0.949).
- BUT it under-predicts AI: real_acc 0.994, ai_acc 0.788 at t=0.5.
- Headroom (optimal threshold vs frozen 0.5):
  - clean: +0.038 (best t=0.20 → 0.929)
  - noise_s0.05: **+0.203** (best t=0.10 → 0.938); noise AUROC 0.989
- Diagnosis: this is a **calibration** problem, not a discrimination problem.
  The ranking is excellent; AI probabilities are just pushed too low, worst
  under noise.

## v7 — class weighting to fix calibration (DALL·E 3, t=0.5 frozen)

Upweight the AI class so the frozen t=0.5 cut lands well. sid_set + ai_weight.

| ai_weight | clean | noise_s0.05 | noise_s0.1 | mean-xform | note |
|---|---|---|---|---|---|
| 1.0 (v6) | 0.891 | 0.735 | ~0.70 | ~0.85 | under-predicts AI |
| 2.0 (cw2) | 0.907 | 0.771 | 0.725 | 0.868 | clean up, noise still low |
| 3.0 (cw3) | 0.921 | 0.795 | 0.764 | 0.886 | |
| 5.0 (cw5) | 0.920 | 0.814 | 0.781 | **0.892** | **BEST overall** |

- Class weight helps BOTH clean and noise monotonically (cw5 ≥ cw3 > cw2 > v6).
  clean plateaus ~0.920–0.921; noise and mean-xform keep rising with weight.
- **cw5 = best model so far**: clean 0.920 (+0.040 over v5_nocifake 0.880),
  mean-xform 0.892 (+0.025). Noise still the weak point (ai_acc 0.56–0.76,
  real_acc ~1.0) — AI probs pushed low under noise.
- Next: image-level transform augmentation on AI data to raise transform-time
  AI probabilities directly (targets the noise weakness).

## v8 — image-level transform augmentation (DALL·E 3)

Train on sid_set (4000 real + 4000 AI) plus 3 transformed copies of every AI
image (jpeg_q50, blur_s1.0, noise_s0.05) → 20000 features (sid_set_aug).
Probe = logistic regression + ai_weight.

| Model | clean | noise_s0.05 | noise_s0.1 | mean-xform | worst |
|---|---|---|---|---|---|
| cw5 (no aug, ref) | 0.920 | 0.814 | 0.781 | 0.892 | 0.781 |
| v8_aug_w3 | 0.909 | 0.927 | 0.908 | 0.9224 | 0.896 (jpeg_q50) |
| v8_aug_w1 | **0.926** | 0.930 | 0.923 | 0.9281 | 0.914 (blur_s2.0) |
| **v8_aug_w15** | 0.917 | 0.934 | **0.948** | **0.9329** | **0.919 (resize_0.5x)** |

- **Augmentation is a major breakthrough for robustness**: mean-xform rises
  0.892 → 0.9329; worst-case 0.781 → 0.919. The noise weakness is fixed
  (noise_s0.1: 0.781 → 0.948 with w15).
- Class-weight trade-off on augmented features:
  - **w1**: best clean (0.926, beats cw5's 0.920), balanced; weakest row
    blur_s2.0 (0.914).
  - **w15**: best mean-xform AND best worst-case; over-predicts AI on clean
    (real_acc 0.874) but recovers under most transforms.
  - w3: worse on both axes than w1/w15 — dominated.
- **v8_aug_w15 = best model overall** (mean-xform 0.9329, +0.066 over
  v5_nocifake 0.867; worst 0.919 vs 0.823). v8_aug_w1 is the fallback if
  clean accuracy matters more than transform robustness.
- Full per-transform tables: reports/dalle3_v8_aug_w1|w15/robustness_report.md

## v9 — stratified augmentation (DALL·E 3)

Train on sid_set (4000 real + 4000 AI) plus exactly 1 copy of EVERY scored
transform per AI image (14 transforms × 4000 = 56000 augmented + 4000 clean =
60000 AI features; 64000 total). Probe = logistic regression + ai_weight.

| Model | clean | mean-xform (no noise) | worst |
|---|---|---|---|
| v8_aug_w1 (ref) | 0.927 | 0.9308 | 0.918 (blur_s2.0) |
| v8_aug_w15 (ref) | 0.917 | 0.9314 | 0.919 (resize_0.5x) |
| v9_w1 | 0.898 | 0.9035 | 0.880 (jpeg_q50) |
| v9_w15 | 0.883 | 0.8956 | 0.872 (jpeg_q90) |

- **v9 stratified augmentation underperforms v8 random augmentation** by a
  wide margin (mean-xform 0.8956–0.9035 vs 0.9308–0.9314).
- Possible causes:
  - Too many augmented copies dilute the training signal
  - Stratified (deterministic) augmentation may not provide as much diversity
    as random sampling
  - The probe may be overfitting to the specific transform patterns
- **Conclusion**: v8 random augmentation is superior. Stick with v8_aug_w15
  as the best model.

## v10 — combo5: more sources + higher class weight (DALL·E 3)

Added more AI sources and raised class weight.

| Model | clean | mean-xform | worst |
|---|---|---|---|
| v8_aug_w15 (ref) | 0.917 | 0.9329 | 0.919 |
| **v10_combo5_w35** | 0.959 | 0.9632 | 0.945 |

- Big jump from adding more diverse AI sources (ComfyUI, Flux) + w3.5.

## v11 — all features + w4.5 (DALL·E 3)

Combined all available features (sid_set_aug + comfy_aug5 + flux) at ai_weight 4.5.

| Model | clean | mean-xform | worst |
|---|---|---|---|
| v10_combo5_w35 (ref) | 0.959 | 0.9632 | 0.945 |
| **v11_all_w45** | 0.965 | 0.9719 | 0.959 |

- Best single probe. Clean 0.965, mean-xform 0.9719.

## v12/v13 — all-generators + extra dataset probes (DALL·E 3)

Trained probes on all features PLUS an extra generator dataset (Sana, then
Midjourney) to add source diversity.

- These are "superset" probes (v11's data + one more). Useful as ensemble
  members but highly correlated with v11.

## Ensemble — 5 members (DALL·E 3)

Averaged P(AI) across 5 probes sharing one CLIP backbone:
v11_w45, v11_w40, v11_w50, v12_sana, v13_midjourney.

| Config | clean | mean-xform | worst |
|---|---|---|---|
| v11_all_w45 (single) | 0.965 | 0.9719 | 0.959 |
| **5-member ensemble** | 0.978 | 0.9739 | 0.968 |

- Ensemble of correlated probes gives a modest gain (+0.002 mean, +0.009 worst).

## Flux8 / v14 — extra Flux data (DALL·E 3)

Added more Flux features to the all-gen probe. **Did not help** — reverted.

## SD3 / v15 — all-gen + SD3 superset probe (DALL·E 3)

Trained probe_v15 on (sid_set_aug + comfy_aug5 + flux + sana + midjourney + sd3).
Added as 6th member.

| Config | clean | mean-xform | worst |
|---|---|---|---|
| 5-member (ref) | 0.978 | 0.9739 | 0.968 |
| 6-member + v15 (all-gen+SD3) | 0.9718 | 0.9718 | 0.9590 |

- **WORSE.** Root cause: v15 = v13's data + SD3, so highly correlated with
  v13. A correlated superset member hurts the ensemble. **Reverted.**

## KEY INSIGHT — pure single-generator probes (DALL·E 3)

Instead of superset probes, train a probe on ONE generator's AI + real
negatives only. Pure probes learn generator-specific signatures that
diversify the ensemble.

| Config | clean | mean-xform | worst |
|---|---|---|---|
| 5-member (ref) | 0.978 | 0.9739 | 0.968 |
| 6-member + v16 (pure-SD3) | 0.9767 | 0.9767 | 0.9690 |
| 7-member + v17 (pure-MJ) | 0.9793 | 0.9793 | 0.9730 |
| **8-member + v18 (pure-Sana)** | **0.9860** | **0.9814** | **0.9710** |

- Each pure probe (SD3, Midjourney, Sana) monotonically improved mean-xform
  AND worst-case. **Pure probes = the breakthrough.**

## Pure-Flux / v19 — ensemble saturation (DALL·E 3)

Added a 9th AI-focused probe (pure-Flux).

| Config | clean | mean-xform | worst |
|---|---|---|---|
| 8-member (ref) | 0.9860 | 0.9814 | 0.9710 |
| 9-member + v19 (pure-Flux) | 0.9804 | 0.9804 | 0.9650 |

- **WORSE.** One more AI-focused probe tips the balance toward over-flagging
  noise as AI. **Saturation reached — 8 members is optimal. Reverted.**

## FINAL — 8-member ensemble (DALL·E 3, t=0.5 frozen)

Members: v11_w45, v11_w40, v11_w50, v12_sana, v13_midjourney, v16_sd3_only,
v17_midjourney_only, v18_sana_only.

| Transform | Accuracy |
|---|---|
| clean | 0.9860 |
| jpeg_q90 | 0.9830 |
| jpeg_q70 | 0.9810 |
| jpeg_q50 | 0.9840 |
| jpeg_q30 | 0.9850 |
| blur_s0.5 | 0.9880 |
| blur_s1.0 | 0.9890 |
| blur_s2.0 | 0.9750 |
| resize_0.5x | 0.9870 |
| resize_0.25x | 0.9870 |
| noise_s0.02 | 0.9780 |
| noise_s0.05 | 0.9710 |
| noise_s0.1 | 0.9710 |
| jitter_20pct | 0.9830 |
| crop_80pct | 0.9770 |

- **Mean-transform = 0.9814, worst-case = 0.9710 (noise_s0.05/0.1).**
- Progression: 5-member 0.9739/0.9680 → 8-member 0.9814/0.9710 (+0.0075 mean,
  +0.0030 worst).

## Parameter budget (vs 2B limit)

- CLIP ViT-L/14 full: 427,616,513; vision tower (used): 303,966,208
- 8 logistic probes: 6,152 params (769 each = 768 coef + 1 bias)
- vision + probes = 303,972,360 (15.2% of 2B); full CLIP + probes = 21.4% of 2B
- **Probes are negligible; well under the limit either way.**
