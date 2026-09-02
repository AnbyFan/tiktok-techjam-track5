#!/usr/bin/env python3
"""
Scan a single image and report AI/REAL prediction.

Usage:
    python scan_image.py path/to/image.jpg
    python scan_image.py path/to/image.jpg --config ensemble_config.json
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

from ensemble_core import EnsembleScorer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Path to the image file")
    p.add_argument("--config", default="ensemble_config.json",
                   help="JSON config listing member probes + weights")
    return p.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)

    if not image_path.exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    scorer = EnsembleScorer(args.config)
    print(f"Loaded {len(scorer.probes)} probes (threshold={scorer.threshold})\n")

    img = Image.open(image_path).convert("RGB")
    tensor = scorer.preprocess(img)
    prob = scorer.score_tensors(torch.stack([tensor]))[0]
    is_ai = prob >= scorer.threshold

    if is_ai:
        print(f"🤖 AI-GENERATED")
        print(f"   Confidence: {prob:.2%}")
    else:
        print(f"📷 REAL IMAGE")
        print(f"   Confidence: {1 - prob:.2%}")


if __name__ == "__main__":
    main()
