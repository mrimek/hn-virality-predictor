# Contributing

The most valuable contributions are **new features** — signals extracted from a post's title, URL, or timing that might predict virality better than what's already there. The loop is fast: add a feature, retrain, check if AUC goes up.

---

## How to add a new feature

All features live in **`features.py`**. Adding one requires exactly three edits to that file:

### 1. Add a pattern or constant (top of file)

```python
# Example: detect posts mentioning a version number like "v2.0" or "2.1.3"
VERSION_PATTERN = re.compile(r"(?i)\bv?\d+\.\d+(\.\d+)?\b")
```

### 2. Compute it in `extract_features()` (around line 165)

```python
df["has_version_number"] = df["title"].str.contains(VERSION_PATTERN).astype(int)
```

The function receives a DataFrame with columns: `id`, `title`, `url`, `score`, `time`, `num_comments`, `author`. It returns a DataFrame of feature columns. All features must be numeric (int or float).

### 3. Add the column name to `FEATURE_COLS` (around line 236)

```python
FEATURE_COLS = [
    ...
    "has_version_number",   # ← add here, in the relevant group
    ...
]
```

That's it. The training pipeline, inference path, and CLI all pick it up automatically.

---

## Retrain and evaluate

```bash
# Install deps
pip install -r requirements.txt

# Download training data (~500MB, cached in data/)
python3 fetch_data.py

# Retrain just the Show HN model (fastest, ~2 min)
python3 train.py --subsets show_hn_3y

# Or retrain everything
python3 train.py
```

The output shows AUC before and after calibration. A good new feature typically lifts AUC by 0.002–0.010. Anything above +0.005 is worth a PR.

```
show_hn_3y:
  Viral posts: 3,904 / 73,544 (5.3%)
  Test AUC (raw): 0.7641          ← compare this to baseline 0.762
  Calibrated prob range: [0.008, 0.187]  mean=0.053
```

---

## Feature ideas worth exploring

These haven't been tried yet — open issues if you want to claim one:

- **`title_has_colon`** — "Show HN: Foo: a bar thing" has two colons; does the subtitle structure help?
- **`title_starts_with_number`** — "10 things..." style titles; positive or negative?
- **`author_karma`** or **`author_age_days`** — does submitter reputation predict virality? (needs enrichment from HN API)
- **`title_has_em_dash`** — "Show HN: FastDB — zero-dependency SQLite" — does the dash signal a cleaner title structure?
- **`is_zig` / `is_gleam` / `is_elixir`** — niche language communities like Lua and Lisp punch above their weight; worth expanding the tech stack features
- **`url_tld`** — `.io` vs `.com` vs `.org` vs `.dev`; does the domain TLD carry signal?
- **`title_ends_with_parens`** — is a parenthetical at the end stronger than one in the middle?
- **`month_of_year`** — already in the model, but seasonal effects haven't been deeply analyzed

---

## Guidelines

- One feature per PR is ideal — easier to measure impact cleanly
- Include the AUC delta in your PR description (before/after)
- Features must be computable at prediction time from title + URL alone — no post-submission signals (score, comments, karma)
- If a feature only applies to Show HN, note that in a comment; it can still be in `FEATURE_COLS` (it'll be 0 for non-Show-HN posts)

---

## Project structure

```
features.py     ← add features here
train.py        ← training pipeline (don't need to touch for most contributions)
predict.py      ← CLI + load_model() helper
fetch_data.py   ← downloads HN dataset from Hugging Face
fetch_models.py ← downloads pre-trained models from GitHub Releases
analyze.py      ← exploratory analysis
tests/          ← pytest tests for feature extraction
models/         ← trained model pickles (gitignored — use fetch_models.py)
```
