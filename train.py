import argparse
import pickle
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
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
    class_weight="balanced",
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(100)],
    )

    y_prob_raw = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob_raw)

    print(f"\n  Test AUC: {auc:.4f}")

    # Calibrate probabilities: class_weight="balanced" squashes raw predict_proba
    # output near zero. Platt scaling corrects this to the true positive rate.
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated.fit(X_test, y_test)

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

    return {"model": calibrated, "feature_cols": FEATURE_COLS, "auc": auc, "subset": name}


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
