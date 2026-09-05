#!/usr/bin/env python3
"""
Simple Telegram bot for AI image detection using attention pooling CLIP model.

Usage:
    1. Get a bot token from @BotFather on Telegram
    2. Create .env file with BOT_TOKEN
    3. Run: .venv\Scripts\python.exe telegram_bot.py

Send an image to the bot and it will respond with "AI" or "REAL".
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import joblib
import numpy as np
import open_clip
from PIL import Image
import io
import requests

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Load .env file if it exists
def load_env():
    for candidate in (Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"):
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ[key.strip()] = value.strip()
            break

load_env()

# Configuration
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Global variables
clip_model = None
preprocess = None
attn_pool = None
classifier = None
device = None


class AttentionPooling(nn.Module):
    """Learnable attention pooling layer."""

    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (batch, n_patches, dim)
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        # Self-attention with query
        attn_out, _ = self.attn(query, x, x)
        attn_out = self.norm(attn_out + query)
        attn_out = self.proj(attn_out)
        return attn_out.squeeze(1)  # (batch, dim)


def get_updates(offset=None):
    """Long-poll for new messages."""
    params = {"timeout": 30}
    if offset:
        params["offset"] = offset
    r = requests.get(f"{API_BASE}/getUpdates", params=params)
    return r.json()


def send_message(chat_id, text):
    """Send a text message."""
    r = requests.post(f"{API_BASE}/sendMessage", json={
        "chat_id": chat_id,
        "text": text
    })
    return r.json()


def download_image(file_id):
    """Download an image from Telegram."""
    # Get file path
    r = requests.get(f"{API_BASE}/getFile", params={"file_id": file_id})
    file_path = r.json()["result"]["file_path"]

    # Download file
    r = requests.get(f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}")
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def extract_clip_features(model, preprocess, image, device):
    """Extract standard CLIP features."""
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.float().cpu().numpy()[0]


def extract_patch_features(model, preprocess, image, device, num_patches=4):
    """Extract CLIP features from multiple patches (returns all patches)."""
    w, h = image.size
    patch_w = w // num_patches
    patch_h = h // num_patches
    features = []
    for i in range(num_patches):
        for j in range(num_patches):
            box = (j * patch_w, i * patch_h, (j + 1) * patch_w, (i + 1) * patch_h)
            patch = image.crop(box)
            feat = extract_clip_features(model, preprocess, patch, device)
            features.append(feat)
    return np.array(features)  # (n_patches, dim)


def predict_image(image):
    """Run image through the attention pooling model and return prediction."""
    feats = extract_patch_features(clip_model, preprocess, image, device)
    # Convert to tensor and add batch dimension
    feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
    # Apply attention pooling
    with torch.inference_mode():
        pooled = attn_pool(feats_t)
        logits = classifier(pooled)
        prob = torch.sigmoid(logits).item()
    is_ai = prob >= 0.5
    return is_ai, prob


def main():
    global clip_model, preprocess, attn_pool, classifier, device

    print("Loading attention pooling CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    clip_model.eval()

    # Load attention pooling model
    model_path = Path(__file__).parent.parent / "models" / "model_attention_pooling_phone.joblib"
    model_state = joblib.load(model_path)
    dim = model_state['dim']
    n_heads = model_state['n_heads']

    attn_pool = AttentionPooling(dim, n_heads=n_heads).to(device)
    attn_pool.load_state_dict(model_state['attn_pool'])
    attn_pool.eval()

    classifier = nn.Linear(dim, 1).to(device)
    classifier.load_state_dict(model_state['classifier'])
    classifier.eval()

    print(f"Loaded on {device}. Bot ready!")
    print("Send an image to the bot to get a prediction.\n")

    offset = None
    while True:
        try:
            updates = get_updates(offset)
            for update in updates.get("result", []):
                offset = update["update_id"] + 1

                # Handle new messages with photos
                if "message" in update and "photo" in update["message"]:
                    chat_id = update["message"]["chat"]["id"]
                    # Get the largest photo
                    photos = update["message"]["photo"]
                    file_id = photos[-1]["file_id"]

                    # Send "analyzing" message
                    send_message(chat_id, "🔍 Analyzing image...")

                    # Download and predict
                    image = download_image(file_id)
                    is_ai, prob = predict_image(image)

                    # Send result
                    if is_ai:
                        result = f"🤖 AI-GENERATED\nConfidence: {prob:.2%}"
                    else:
                        result = f"📷 REAL IMAGE\nConfidence: {1-prob:.2%}"

                    send_message(chat_id, result)

        except KeyboardInterrupt:
            print("\nBot stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    if not BOT_TOKEN:
        print("Error: BOT_TOKEN not found.")
        print("Create a .env file from .env.example and add your token:")
        print("  BOT_TOKEN=your_token_here")
        print("Get a token from @BotFather on Telegram.")
        sys.exit(1)
    main()
