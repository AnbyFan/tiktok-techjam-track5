#!/usr/bin/env python3
"""
Simple Telegram bot for AI image detection.

Usage:
    1. Get a bot token from @BotFather on Telegram
    2. Create .env file with BOT_TOKEN
    3. Run: python telegram_bot.py

Send an image to the bot and it will respond with "AI" or "REAL".
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Auto-activate venv if not already active
VENV_PYTHON = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    print(f"Using venv: {VENV_PYTHON}")
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

import requests

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ensemble_core import EnsembleScorer
from PIL import Image
import io

# Load .env file if it exists
def load_env():
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ[key.strip()] = value.strip()

load_env()

# Configuration
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"


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


def predict_image(image):
    """Run image through the ensemble and return prediction."""
    tensor = scorer.preprocess(image)
    import torch
    prob = scorer.score_tensors(torch.stack([tensor]))[0]
    is_ai = prob >= scorer.threshold
    return is_ai, prob


def main():
    global scorer

    print("Loading ensemble model...")
    scorer = EnsembleScorer("ensemble_config.json")
    print(f"Loaded {len(scorer.probes)} probes. Bot ready!")
    print("Send an image to the bot to get a prediction.\n")

    offset = None
    while True:
        try:
            updates = get_updates(offset)
            for update in updates.get("result", []):
                offset = update["update_id"] + 1

                # Handle new messages with photos
                if "message" in update and "photo" in update["message"]:
                    chat_id = update["message"]["chat_id"]
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
