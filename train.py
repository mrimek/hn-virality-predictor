import argparse
import pickle
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from features import extract_features, FEATURE_COLS, VIRAL_THRESHOLD_PERCENTILE, CUTOFF_5Y, CUTOFF_3Y, CUTOFF_1Y

DATA       = "data/hn_stories.parquet"
MODELS_DIR = Path("models")

SUBSETS = {
    "full":        lambda df: df,
    "show_hn":     lambda df: df[df["is_show_hn"] == 1],
    "show_hn_3y":  lambda df: df[(df["is_show_hn"] == 1) & (df["dt"] >= CUTOFF_3Y)],
    "recent_5y":   lambda df: df[df["dt"] >= CUTOFF_5Y],
    "recent_1y":   lambda df: df[df["dt"] >= CUTOFF_1Y],
}

# Tuned for higher AUC: lower LR + more trees (early stopping handles cutoff),
# deeper leaves, mild regularization to prevent overfitting on new features.
# NOTE: do NOT use class_weight="balanced" — it compresses raw predict_proba
# output near zero, which breaks Platt calibration. Use scale_pos_weight
# instead (set dynamically per subset based on actual class ratio).
LGBM_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=127,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_alpha=0.05,
    reg_lambda=0.5,
    random_state=42,
    n_jobs=-1,
)


def train_one(df: pd.DataFrame, name: str) -> dict:
    print(f"\n{'='*55}")
    print(f"  Subset: {name}  ({len(df):,} rows)")
    print(f"{'='*55}")

    if len(df) < 500:
        print(f"  Skipping — too few rows ({len(df)})")
        return None

    X = df[FEATURE_COLS]
    y = df["viral"]

    viral_pct = y.mean() * 100
    print(f"  Viral posts: {y.sum():,} / {len(y):,} ({viral_pct:.1f}%)")

    # 3-way split: 70% train LightGBM, 15% calibrate Platt, 15% evaluate
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # scale_pos_weight balances classes without compressing predict_proba
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw   = n_neg / n_pos
    print(f"  scale_pos_weight: {spw:.1f}")
    print(f"  Split: {len(X_train):,} train / {len(X_cal):,} cal / {len(X_test):,} test")

    model = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=spw)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(100)],
    )

    y_prob_raw = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob_raw)

    print(f"\n  Test AUC (raw): {auc:.4f}")
    print(f"  Raw prob range: [{y_prob_raw.min():.4f}, {y_prob_raw.max():.4f}]  mean={y_prob_raw.mean():.4f}")

    # Calibrate on a separate holdout — fitting on the test set compresses the
    # Platt sigmoid to near-zero (A ~ -60), making all outputs identical.
    # cv="prefit" means the base model is already trained; we only fit calibration.
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(X_cal, y_cal)

    y_prob = calibrated.predict_proba(X_test)[:, 1]
    y_pred = calibrated.predict(X_test)

    print(f"  Calibrated prob range: [{y_prob.min():.3f}, {y_prob.max():.3f}]  mean={y_prob.mean():.3f}")
    print("\n  Classification Report (calibrated):")
    print(classification_report(y_test, y_pred,
                                target_names=["not viral", "viral"],
                                digits=3))

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    top = importances.sort_values(ascending=False).head(15)
    print("  Top 15 features:")
    for feat, imp in top.items():
        print(f"    {feat:<35} {imp:>6}")

    return {"model": calibrated, "feature_cols": FEATURE_COLS, "auc": auc, "subset": name, "n_train": len(df)}


def main(data_path: str, subsets: list):
    print(f"Loading {data_path}...")
    raw = pd.read_parquet(data_path)
    print(f"  {len(raw):,} rows")

    print("Extracting features...")
    df = extract_features(raw)

    MODELS_DIR.mkdir(exist_ok=True)

    for name in subsets:
        subset_fn = SUBSETS[name]
        subset_df = subset_fn(df)
        payload = train_one(subset_df, name)
        if payload is None:
            continue
        out = MODELS_DIR / f"lgbm_{name}.pkl"
        with open(out, "wb") as f:
            pickle.dump(payload, f)
        print(f"\n  Saved → {out}")

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA)
    parser.add_argument(
        "--subsets", nargs="+",
        default=list(SUBSETS.keys()),
        choices=list(SUBSETS.keys()),
        help="Which model subsets to train (default: all)"
    )
    args = parser.parse_args()
    main(args.data, args.subsets)
