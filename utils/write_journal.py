import pathlib

content = """# Iteration Journal - Track 5 AI Image Detection

## Baseline (probe_v4)

- Own-generator (ComfyUI): clean acc=0.875, AUROC=0.960
- Official (DALL-E 3): clean acc=0.811, AUROC=0.901
- Worst rows: noise sigma 0.05/0.1 (~0.73), blur sigma 2.0 (0.746)
- Known weakness: severe degradation flips real to AI (JPEG q50 real_acc=0.718)

"""

path = pathlib.Path(r"C:\Users\Sparkle\Downloads\tiktok techjam 2026 track 5\JOURNAL.md")
path.write_text(content)
print(f"Written {path}")
