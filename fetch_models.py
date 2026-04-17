#!/usr/bin/env python3
"""
Download pre-trained models from GitHub Releases.

Usage:
    python3 fetch_models.py
"""
import urllib.request
from pathlib import Path

RELEASE_URL = "https://github.com/mrimek/hn-virality-predictor/releases/download/v1.0.0"
MODELS = [
    "lgbm_show_hn_3y.pkl",
    "lgbm_recent_5y.pkl",
]

Path("models").mkdir(exist_ok=True)

for name in MODELS:
    dest = Path("models") / name
    if dest.exists():
        print(f"  {name} already present, skipping")
        continue
    url = f"{RELEASE_URL}/{name}"
    print(f"  Downloading {name} ...", end=" ", flush=True)
    urllib.request.urlretrieve(url, dest)
    size_kb = dest.stat().st_size // 1024
    print(f"done ({size_kb} KB)")

print("\nModels ready. Run: python3 predict.py --title \"Show HN: ...\"")
