#!/usr/bin/env python3
"""
Retrain the 8-member ensemble with noise-augmented feature dirs.

Each probe keeps its original feature dirs and ADDS the noise-focused
feature dirs (extract_noise_aug.py output) so it sees many more examples at
the exact eval noise levels. ai_weight / C / seed match the original probes.

Run after extract_noise_aug.py has produced:
    features/midjourney_noise, features/sd3_noise, features/comfy_noise
and features/sid_set_noisy already exists.
"""

import subprocess
import sys
from pathlib import Path

PY = sys.executable

# (out_dir, [feature_dirs], ai_weight)
PROBES = [
    ("probe_v11n_all_w45",
     ["features/sid_set_aug", "features/comfy_aug5", "features/flux",
      "features/sid_set_noisy"], 4.5),
    ("probe_v11n_all_w40",
     ["features/sid_set_aug", "features/comfy_aug5", "features/flux",
      "features/sid_set_noisy"], 4.0),
    ("probe_v11n_all_w50",
     ["features/sid_set_aug", "features/comfy_aug5", "features/flux",
      "features/sid_set_noisy"], 5.0),
    ("probe_v12n_all_sana_w45",
     ["features/sid_set_aug", "features/comfy_aug5", "features/flux",
      "features/sana", "features/sid_set_noisy"], 4.5),
    ("probe_v13n_all_midjourney_w45",
     ["features/sid_set_aug", "features/comfy_aug5", "features/flux",
      "features/sana", "features/midjourney",
      "features/sid_set_noisy", "features/midjourney_noise"], 4.5),
    ("probe_v16n_sd3_only_w45",
     ["features/sd3", "features/comfy_aug5", "features/sd3_noise"], 4.5),
    ("probe_v17n_midjourney_only_w45",
     ["features/midjourney", "features/comfy_aug5",
      "features/midjourney_noise"], 4.5),
    # v18: no sana_noise available; use balanced sid_set_noisy for noise
    # invariance without injecting a different generator's AI signal.
    ("probe_v18n_sana_only_w45",
     ["features/sana", "features/comfy_aug5", "features/sid_set_noisy"], 4.5),
]


def main():
    base = Path(__file__).parent
    for out, fdirs, w in PROBES:
        print(f"\n{'='*60}\n[retrain] {out}  (ai_weight={w})\n{'='*60}",
              flush=True)
        cmd = [PY, "train_probe_cw.py", "--out", out,
               "--ai-weight", str(w), "--features", *fdirs]
        r = subprocess.run(cmd, cwd=base)
        if r.returncode != 0:
            print(f"[retrain] FAILED: {out}", flush=True)
            return
    print("\n[retrain] all 8 probes done", flush=True)


if __name__ == "__main__":
    main()
