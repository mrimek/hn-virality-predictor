import datetime
import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from features import features_for_prediction, FEATURE_COLS

app = FastAPI(title="HN Virality Predictor")

MODELS_DIR = Path("models")
_cache: dict = {}

# Model registry: label, description, which tab it belongs to
MODEL_REGISTRY = {
    "show_hn_3y": {
        "label": "Show HN · Past 3 Years",
        "description": "Recommended — trained on 2023–2026 Show HN posts (AUC 0.757)",
        "tab": "show_hn",
        "auc": 0.757,
        "recommended": True,
    },
    "show_hn": {
        "label": "Show HN · All Time",
        "description": "All 190K Show HN posts since 2007 (AUC 0.696)",
        "tab": "show_hn",
        "auc": 0.696,
        "recommended": False,
    },
    "recent_1y": {
        "label": "General HN · Past Year",
        "description": "Recommended for general posts — 2025–2026 (AUC 0.747)",
        "tab": "general",
        "auc": 0.747,
        "recommended": True,
    },
    "recent_5y": {
        "label": "General HN · Past 5 Years",
        "description": "Broader window — 2021–2026 (AUC 0.735)",
        "tab": "general",
        "auc": 0.735,
        "recommended": False,
    },
}


def load_model(name: str):
    if name not in _cache:
        path = MODELS_DIR / f"lgbm_{name}.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            _cache[name] = pickle.load(f)
    return _cache[name]


def generate_analysis(feats: dict, title: str, url: str, tab: str) -> tuple[list, list]:
    signals, suggestions = [], []
    title_len = int(feats.get("title_len", len(title)))
    hour = int(feats.get("hour", 12))

    # ── Positive signals ──────────────────────────────────────
    if feats.get("is_github") or feats.get("title_mentions_github"):
        signals.append({"label": "GitHub link", "positive": True})

    if feats.get("is_live_url"):
        signals.append({"label": "Live product URL", "positive": True})

    if feats.get("has_demo"):
        signals.append({"label": "Demo mentioned", "positive": True})

    if feats.get("is_solo_build"):
        signals.append({"label": '"I built" narrative', "positive": True})

    if feats.get("built_timeframe"):
        signals.append({"label": "Build timeframe stated", "positive": True})

    if feats.get("has_free_signal"):
        signals.append({"label": "Free / open source", "positive": True})

    if feats.get("title_sweet_spot"):
        signals.append({"label": f"Title length ({title_len} chars)", "positive": True})

    if feats.get("has_parens"):
        signals.append({"label": "Has parenthetical", "positive": True})

    tech_hit = next(
        (t for t in ("tech_rust", "tech_go", "tech_cpp", "tech_wasm", "tech_typescript", "tech_python")
         if feats.get(t)), None
    )
    if tech_hit:
        name_map = {
            "tech_rust": "Rust", "tech_go": "Go", "tech_cpp": "C++",
            "tech_wasm": "WebAssembly", "tech_typescript": "TypeScript", "tech_python": "Python",
        }
        signals.append({"label": f"Tech stack: {name_map[tech_hit]}", "positive": True})

    if 11 <= hour <= 18:
        signals.append({"label": f"Good posting hour ({hour}:00 UTC)", "positive": True})

    # ── Negative signals + suggestions ────────────────────────
    if feats.get("topic_ai_ml"):
        signals.append({"label": "AI/ML topic (4.6% viral rate)", "positive": False})
        suggestions.append(
            "AI/ML is the most crowded Show HN category with the lowest viral rate (4.6% — down from 15% in 2021). "
            "If AI is implementation detail rather than the product itself, lead with the underlying technical problem you're solving."
        )

    if not feats.get("is_github") and not feats.get("title_mentions_github"):
        signals.append({"label": "No GitHub link", "positive": False})
        suggestions.append(
            "Open source projects with a GitHub link perform nearly 3× better on HN. "
            "If the project isn't open source, consider it — even a core library or the data pipeline."
        )

    if not feats.get("title_sweet_spot"):
        if title_len < 60:
            signals.append({"label": f"Title short ({title_len} chars)", "positive": False})
            suggestions.append(
                f"Your title is {title_len} characters — the sweet spot is 60–80. "
                "Add a key technical differentiator or the main constraint: what makes this interesting?"
            )
        else:
            signals.append({"label": f"Title long ({title_len} chars)", "positive": False})
            suggestions.append(
                f"Your title is {title_len} characters — trim to under 80. "
                "Cut adjectives and marketing language first; keep the technical core."
            )

    if not feats.get("has_parens"):
        signals.append({"label": "No parenthetical", "positive": False})
        suggestions.append(
            "Titles with a parenthetical consistently outperform. "
            'Pack in a key constraint or implementation detail: "(written in Zig)", "(open source)", "(sub-millisecond latency)".'
        )

    if not (11 <= hour <= 18):
        signals.append({"label": f"Suboptimal hour ({hour}:00 UTC)", "positive": False})
        suggestions.append(
            f"You're set to post at {hour}:00 UTC. "
            "Best window is 11am–6pm UTC — peak for both US morning and EU afternoon engagement. "
            "Sunday and Saturday are the best days for Show HN."
        )

    if tab == "show_hn":
        if not feats.get("is_show_hn"):
            signals.append({"label": 'Missing "Show HN:" prefix', "positive": False})
            suggestions.append(
                'Add "Show HN: " at the start of your title — it routes your post to the right audience '
                "and signals that you're sharing something you built."
            )

        if not feats.get("is_solo_build") and not feats.get("is_team_build"):
            suggestions.append(
                '"I built…" framing outperforms "we built" and neutral titles on Show HN. '
                "HN rewards individual craft — if it was largely your work, say so."
            )

        if not feats.get("built_timeframe"):
            suggestions.append(
                "If you built this in a specific timeframe, mention it. "
                '"Show HN: I built X in a weekend" consistently outperforms equivalent titles without the constraint — '
                "it makes the achievement concrete and the story compelling."
            )

        if not feats.get("has_demo") and not feats.get("is_live_url"):
            signals.append({"label": "No live demo", "positive": False})
            suggestions.append(
                "Show HN posts with a working demo significantly outperform those without. "
                "Even a minimal hosted version lowers the friction to try your project — "
                "readers are more likely to upvote something they can immediately experience."
            )
    else:
        # General HN tips
        if feats.get("is_show_hn"):
            signals.append({"label": '"Show HN:" on General model', "positive": False})
            suggestions.append(
                'Your title starts with "Show HN:" but you\'re using the General HN model. '
                "Switch to the Show HN tab for a more accurate score."
            )

    return signals, suggestions


class PredictRequest(BaseModel):
    title: str
    url: str = ""
    model: str = "show_hn_3y"
    tab: str = "show_hn"  # "show_hn" or "general"


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/models")
def list_models():
    result = {}
    for name, info in MODEL_REGISTRY.items():
        result[name] = {
            **info,
            "available": (MODELS_DIR / f"lgbm_{name}.pkl").exists(),
        }
    return result


@app.post("/predict")
def predict(req: PredictRequest):
    payload = load_model(req.model)
    if payload is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Model '{req.model}' not trained yet. Run: python3 train.py --subsets {req.model}"},
        )

    model = payload["model"]
    model_cols = payload.get("feature_cols", FEATURE_COLS)

    now = datetime.datetime.now(datetime.timezone.utc)
    feats = features_for_prediction(req.title, req.url, now)

    X = pd.DataFrame([feats])[model_cols]
    prob = float(model.predict_proba(X)[0][1])

    # Top 10%: direct from model (trained on top-10% threshold)
    top_10 = round(prob * 100, 1)

    # Top 1%: rough estimate — only posts in the upper end of the viral set reach top 1%
    # Calibrated so prob=0.9 → ~20%, prob=0.7 → ~12%, prob=0.5 → ~6%
    top_1 = round(min(prob ** 2 * 22, 95.0), 1)

    signals, suggestions = generate_analysis(feats, req.title, req.url, req.tab)

    reg = MODEL_REGISTRY.get(req.model, {})

    return {
        "top_10_pct": top_10,
        "top_1_pct": top_1,
        "prediction": "viral" if prob >= 0.5 else "not viral",
        "model": req.model,
        "model_label": reg.get("label", req.model),
        "model_auc": round(payload.get("auc", reg.get("auc", 0)), 3),
        "signals": signals,
        "suggestions": suggestions,
    }


app.mount("/static", StaticFiles(directory="static"), name="static")
