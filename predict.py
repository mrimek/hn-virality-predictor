#!/usr/bin/env python3
"""
Predict virality of a Hacker News post.

Usage:
    python3 predict.py --title "Show HN: I built a fast Rust HTTP server" --url "https://github.com/..."
    python3 predict.py --title "Ask HN: What's the best way to learn Rust?" --model recent_1y
    python3 predict.py --title "OpenAI raises $10B" --model recent_5y

Available models: full, show_hn, show_hn_3y, recent_5y, recent_1y
"""
import argparse
import pickle
import pandas as pd
from pathlib import Path

from features import features_for_prediction, FEATURE_COLS

import os
MODELS_DIR = Path(os.environ.get("HN_MODELS_DIR", "models"))
DEFAULT_MODEL = "show_hn_3y"  # best model for current Show HN posts


def load_model(name: str):
    path = MODELS_DIR / f"lgbm_{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Model '{name}' not found at {path}. Run: python3 train.py --subsets {name}"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(title: str, url: str = "", model_name: str = DEFAULT_MODEL) -> dict:
    payload = load_model(model_name)
    model = payload["model"]

    feats = features_for_prediction(title, url)
    X = pd.DataFrame([feats])[FEATURE_COLS]

    prob = model.predict_proba(X)[0][1]

    return {
        "title": title,
        "url": url,
        "model": model_name,
        "virality_probability": round(prob, 4),
        "prediction": "viral" if prob >= 0.5 else "not viral",
        "model_auc": payload.get("auc"),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict HN post virality")
    parser.add_argument("--title", required=True)
    parser.add_argument("--url", default="")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        choices=["show_hn_3y", "recent_5y", "show_hn"],
        help="Which trained model to use (default: show_hn_3y)"
    )
    args = parser.parse_args()

    result = predict(args.title, args.url, args.model)

    print(f"\nTitle  : {result['title']}")
    if result["url"]:
        print(f"URL    : {result['url']}")
    print(f"Model  : {result['model']}  (AUC: {result['model_auc']:.4f})")
    print(f"Score  : {result['virality_probability']:.1%} chance of going viral")
    print(f"Label  : {result['prediction'].upper()}")


if __name__ == "__main__":
    main()
