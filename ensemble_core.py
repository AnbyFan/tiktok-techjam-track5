#!/usr/bin/env python3
"""
Shared ensemble scorer for Track 5.

Loads one frozen CLIP backbone plus N linear probes (from a JSON config) and
scores images with a weighted average of their P(AI) probabilities. All members
share the SAME backbone, so a single forward pass serves every probe -- the
ensemble is nearly as fast as one probe.

Config format (ensemble_config.json):
    {
      "model": "ViT-L-14",
      "pretrained": "openai",
      "threshold": 0.5,
      "members": [
        {"probe": "probe_v11_all_w45", "weight": 1.0},
        {"probe": "probe_v12_all_sana_w45", "weight": 1.0}
      ]
    }

Each member's "probe" is a directory containing probe.joblib + probe_config.json
(produced by train_probe.py). Weights are normalized internally.
"""

import json
from pathlib import Path

import joblib
import torch
import open_clip


class EnsembleScorer:
    """Frozen CLIP backbone + N weighted linear probes."""

    def __init__(self, config_path, device=None):
        self.config_path = Path(config_path)
        cfg = json.loads(self.config_path.read_text())
        self.model_name = cfg.get("model", "ViT-L-14")
        self.pretrained = cfg.get("pretrained", "openai")
        self.threshold = float(cfg.get("threshold", 0.5))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Shared backbone (one forward pass serves all members).
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device)
        self.model.eval()
        torch.backends.cudnn.benchmark = True

        # Load each member probe + normalize weights.
        base = self.config_path.parent
        self.probes = []
        raw_weights = []
        for m in cfg["members"]:
            probe_dir = Path(m["probe"])
            if not probe_dir.is_absolute():
                probe_dir = base / probe_dir
            clf = joblib.load(probe_dir / "probe.joblib")
            self.probes.append((probe_dir.name, clf))
            raw_weights.append(float(m.get("weight", 1.0)))
        total = sum(raw_weights)
        if total <= 0:
            raise ValueError("ensemble member weights must sum to > 0")
        self.weights = [w / total for w in raw_weights]

    def encode(self, tensors):
        """Run the shared backbone once; return L2-normalized float features."""
        with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=(self.device == "cuda")):
            t = tensors.to(self.device, non_blocking=True)
            f = self.model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()

    def score(self, features):
        """Weighted average of each probe's P(AI) over precomputed features."""
        probs = None
        for (name, clf), w in zip(self.probes, self.weights):
            p = clf.predict_proba(features)[:, 1]
            probs = p * w if probs is None else probs + p * w
        return probs

    def score_tensors(self, tensors):
        """Encode + score in one call (convenience for batched use)."""
        return self.score(self.encode(tensors))
