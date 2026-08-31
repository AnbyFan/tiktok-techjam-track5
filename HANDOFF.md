# HANDOFF — TikTok TechJam 2026 Track 5: Robust AI Image Detection

**Deadline: Monday 2026-08-31.** Two days left as of this handoff (Sat 2026-08-29).

## Task

Iteratively improve an AI-vs-real image classifier's accuracy, especially under
15 real-world transforms, at a **frozen decision threshold t=0.5**. Constraints:
<2B params, no training on eval pools. Maintain `JOURNAL.md` as the running
iteration log (append every experiment with numbers).

## Environment

- **Remote machine (where all work runs)**: SSH `Sparkle@192.168.1.22`, no password.
  Project dir: `C:\Users\Sparkle\Downloads\tiktok techjam 2026 track 5\`
- **GPU**: RTX 3080 20 GB (swapped in 2026-08-29; was 3070 Ti 8 GB — you can
  raise batch sizes, OOM risk is much lower).
- **Local backup** (no datasets): `Y:\qwencode\tiktok techjam 2026 track 5\` —
  code, probes, features, reports, .git. Datasets live only on remote under `data/`.
- Python env on remote already has everything (torch, open_clip, datasets, sklearn).

## Architecture (fixed, do not change the paradigm)

Frozen CLIP ViT-L/14 (openai weights) → L2-normalized 768-d features →
logistic regression probe. Threshold frozen at 0.5; we shape probabilities via
training data choice and class weights instead.

## CRITICAL: two eval pools — do not confuse

- **DALL·E 3 (OFFICIAL, cross-generator, HARD)**:
  `data/wildfake/Images/Diffusion_based/DALLE/DALLE/Advanced/DALLE3`
- **ComfyUI/Flux (own-generator, EASY)**: `data/val/ai`
- Real images (eval only): `data/val/real` (COCO val2017)
- **All model comparisons use the DALL·E 3 pool.**

## Key commands (run on remote, from project dir)

```bat
:: Extract features (streams HF dataset saberzl/SID_Set; ~15-20 min)
python extract_features_aug.py --out features/sid_set_aug --per-class 4000 --aug-copies 3 --batch-size 48

:: Train probe (ai_weight = class weight for AI class)
python train_probe_cw.py --features features/sid_set_aug --out probe_v8_aug_w15 --ai-weight 1.5

:: Robustness eval (15 transforms x 1000 imgs; ~10-15 min each)
python eval_robustness.py --real-dir data\val\real --ai-dir data\wildfake\Images\Diffusion_based\DALLE\DALLE\Advanced\DALLE3 --probe probe_v8_aug_w15 --max-per-class 500 --out reports\dalle3_v8_aug_w15

:: Metrics from a (possibly partial) predictions.csv
python metrics_from_preds.py reports\dalle3_v8_aug_w15\predictions.csv
```

## Current state — results (DALL·E 3 pool, t=0.5 frozen)

| Model | clean | mean-xform | worst | notes |
|---|---|---|---|---|
| v4 baseline | 0.811 | — | — | |
| v5_nocifake | 0.880 | 0.867 | 0.823 | dropped CIFAKE |
| v7 cw5 (no aug) | 0.920 | 0.892 | 0.781 | class-weight only |
| v8_aug_w1 | **0.926** | 0.9281 | 0.914 (blur_s2.0) | best clean |
| **v8_aug_w15 (BEST)** | 0.917 | **0.9329** | **0.919 (resize_0.5x)** | most robust |

Full per-transform tables: `reports/dalle3_v8_aug_w1/robustness_report.md` and
`reports/dalle3_v8_aug_w15/robustness_report.md`. History: `JOURNAL.md`.

**v8 = sid_set features + image-level transform augmentation**: for each AI
training image, `extract_features_aug.py` emits the clean feature plus
`--aug-copies` (3) transformed copies, randomly sampled from ALL 14 scored
transform kinds (see `build_aug_pool()` in that script). This was the
breakthrough: mean-xform 0.892 → 0.9329, worst-case 0.781 → 0.919.

## Next steps (in priority order)

1. **v9 — more/better augmentation** (was about to start this):
   - Raise `--aug-copies` to 5-6 (20 GB GPU handles it; 4000 AI × 6 = 28k rows).
   - Or stratify: guarantee ≥1 copy per transform kind instead of pure random,
     so weak rows (blur_s2.0, resize_0.5x, jpeg_q50) get direct coverage.
   - Retrain at ai_weight 1.0 / 1.5 / 2.0, eval, compare to w15.
2. **Try w2** on existing sid_set_aug features (cheap — 30 s train, then eval).
3. **Ensemble w1 + w15** (average prob_ai): their biases differ (w15
   over-predicts AI on clean, w1 is balanced) so averaging may beat both.
   There is `eval_robustness_ensemble.py` / `predict_ensemble.py` from a
   previous (failed, v5-era) ensemble attempt — check/adapt them.
4. **Lock the final model**, then deliverables (per AGENTS.md):
   - Mirror check: verify submission pipeline end-to-end
   - README + Devpost writeup (tell the story from JOURNAL.md: v4→v8 arc,
     calibration diagnosis, augmentation breakthrough)
   - Demo video, fresh-clone test

## Operational gotchas (learned the hard way)

- **SSH timeout kills remote processes**: if a long `ssh ... python ...`
  command times out locally, the remote python may die mid-run.
  `predictions.csv` is written incrementally-ish; `robustness_report.md` only
  at the end. If a run dies, use `metrics_from_preds.py` on the predictions.csv.
  For >10 min runs, prefer running in background and polling.
- **cmd.exe**: `&` runs commands in PARALLEL (not background); use `&&` for
  sequencing. Paths with spaces need quotes.
- **Don't confuse the two eval pools** (see above) — every number in the
  journal is DALL·E 3 unless stated.
- Probes are tiny (logistic regression on 768-d) — training is seconds;
  feature extraction and robustness evals are the slow parts.
- `features/sid_set_aug/` = 4000 real (clean) + 4000 AI (clean) + 12000 AI
  (transformed) = 20000 rows across 10 shards.
