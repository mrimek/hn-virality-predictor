# HN Virality Predictor

A LightGBM classifier that predicts whether a Hacker News post will reach the top 10% by score — trained on 260,000+ HN posts across five model variants.

**Live tool:** [wannalaunch.com](https://wannalaunch.com)

---

## Models

| Model | Training data | Posts | AUC |
|---|---|---|---|
| `show_hn_3y` ⭐ | Show HN posts, Apr 2023–Apr 2026 | 73,544 | **0.762** |
| `recent_5y` | All HN posts, Apr 2021–Apr 2026 | 1,527,461 | 0.714 |

The `show_hn_3y` model is recommended for any "Show HN:" submission — it reflects current community tastes and has the highest AUC.

"Viral" is defined as landing in the **top 10% by score** among posts submitted in the same window. For Show HN, that's roughly 22+ points.

---

## Quick start

```bash
git clone https://github.com/mrimek/hn-virality-predictor
cd hn-virality-predictor
pip install -r requirements.txt
```

Trained models are stored in `models/`. To predict without retraining:

```bash
python3 predict.py \
  --title "Show HN: I built a Postgres-compatible database in Rust" \
  --url "https://github.com/you/yourproject" \
  --model show_hn_3y
```

Output:
```
Title  : Show HN: I built a Postgres-compatible database in Rust
URL    : https://github.com/you/yourproject
Model  : show_hn_3y  (AUC: 0.7620)
Score  : 12.4% chance of going viral
Label  : NOT VIRAL
```

### Use as a library

```python
from features import features_for_prediction, FEATURE_COLS
from predict import load_model
import pandas as pd

payload = load_model("show_hn_3y")
feats = features_for_prediction("Show HN: I built X in Rust", "https://github.com/...")
prob = payload["model"].predict_proba(pd.DataFrame([feats])[FEATURE_COLS])[0][1]
print(f"{prob:.1%} chance of reaching top 10%")
```

---

## Retrain from scratch

Fetch fresh data from the Hugging Face HN dataset and retrain all models:

```bash
python3 fetch_data.py          # downloads HN dataset → data/hn_posts.parquet
python3 train.py               # trains all model variants → models/lgbm_*.pkl
```

Retrain a specific model only:

```bash
python3 train.py --subsets show_hn_3y
```

### Training details

- **Algorithm:** LightGBM binary classifier
- **Calibration:** Platt scaling (`CalibratedClassifierCV`, `cv="prefit"`) fitted on a separate 15% holdout — not the test set, which would cause the sigmoid to collapse
- **Class imbalance:** `scale_pos_weight` set dynamically per subset (e.g. 17.9× for `show_hn_3y`)
- **Split:** 70% train / 15% calibration / 15% test (stratified)
- **Features:** 46 signals across title style, timing, narrative framing, tech stack, domain, and topic

### Key hyperparameters

```python
n_estimators    = 1000   # early stopping applied
learning_rate   = 0.03
max_depth       = 6
num_leaves      = 50
colsample_bytree = 0.7
reg_alpha       = 0.05
reg_lambda      = 0.5
```

---

## Features (46 total)

The top features by model importance:

| Feature | Description |
|---|---|
| `title_caps_ratio` | Fraction of uppercase letters in title — #1 signal; heavy caps reads as hype |
| `hour` | UTC hour of submission |
| `days_since_domain_show_hn` | Days since this domain last appeared in a Show HN post |
| `title_len` | Character count of the title |
| `day_of_week` | 0=Monday … 6=Sunday |
| `is_show_hn` | Title starts with "Show HN:" |
| `is_github` | URL is a GitHub repo |
| `is_solo_build` | Title uses "I built/made/wrote…" framing |
| `has_parens` | Title contains a parenthetical |
| `title_sweet_spot` | Title is 60–80 characters |
| `topic_ai_ml` | Post is in the AI/ML topic cluster |
| `has_demo` | Title or URL signals a live demo |

---

## What the data says

Key findings from training data (full write-up in [FINDINGS.md](FINDINGS.md)):

- **Caps ratio is the strongest signal.** HN readers associate heavy capitalisation with hype. Standard title or sentence case wins.
- **AI/ML is the most saturated category** — viral rate dropped from ~15% (2021) to ~5% (2025). Open source tools hold steady at ~14%.
- **"I built" outperforms "We built"** — the solo-builder narrative consistently outperforms team launches.
- **Posting window matters.** Best hours are 12:00–18:00 UTC. Sunday performs surprisingly well for Show HN (less competition from news).
- **Parentheticals help.** `Show HN: FastDB (written in Zig, MIT)` outperforms the same title without one.
- **GitHub links are a strong positive signal.** Open source projects perform ~3× better than closed-source equivalents.

---

## Repository structure

```
hn-virality-predictor/
├── features.py        # feature extraction (importable as a library)
├── predict.py         # CLI + load_model() helper
├── train.py           # training pipeline
├── fetch_data.py      # downloads HN dataset from Hugging Face
├── analyze.py         # exploratory analysis scripts
├── api.py             # lightweight FastAPI wrapper (optional)
├── models/            # trained model pickles (lgbm_*.pkl)
├── data/              # parquet cache (gitignored)
├── tests/             # pytest tests for feature extraction
├── FINDINGS.md        # full analysis write-up
└── requirements.txt
```

---

## License

MIT
