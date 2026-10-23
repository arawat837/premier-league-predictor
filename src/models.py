"""Model training, Walk-Forward Expanding Window cross-validation, and probability calibration.

Implements:
- Chronological Walk-Forward Expanding Window splitter (zero shuffling, strictly past -> future)
- Baseline Random Forest Classifier vs. XGBoost Classifier (objective='multi:softprob')
- Multiclass probability calibration using CalibratedClassifierCV(method='isotonic')
- Out-of-fold evaluation and final calibrated model persistence to models/calibrated_xgb.joblib
"""

import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Ensure project root is in sys.path when run directly
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.evaluate import evaluate_predictions
from src.features import FEATURE_COLUMNS


def walk_forward_expanding_splits(
    df: pd.DataFrame,
    initial_train_size: int = 760,
    step_size: int = 380,
) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
    """Generate chronological Walk-Forward Expanding Window train/test indices.

    Never shuffles data. Trains on [0 : split_end) and tests on [split_end : split_end + step_size).

    Args:
        df: Chronologically sorted match DataFrame.
        initial_train_size: Number of initial matches for the first training window (default 760 = 2 seasons).
        step_size: Number of subsequent fixtures per evaluation window (default 380 = 1 full season).

    Yields:
        Tuple of (fold_idx, train_indices, test_indices).
    """
    n_samples = len(df)
    if n_samples <= initial_train_size:
        raise ValueError(
            f"Dataset size ({n_samples}) must be greater than initial_train_size ({initial_train_size})."
        )

    fold = 1
    for train_end in range(initial_train_size, n_samples, step_size):
        test_end = min(train_end + step_size, n_samples)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)
        if len(test_idx) == 0:
            break
        yield fold, train_idx, test_idx
        fold += 1


def build_random_forest_pipeline(
    calibrate: bool = True,
    random_state: int = 42,
) -> Any:
    """Create a Baseline Random Forest pipeline optionally wrapped with Isotonic Calibration."""
    base_rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ])
    if not calibrate:
        return base_rf

    cv_unshuffled = StratifiedKFold(n_splits=5, shuffle=False)
    return CalibratedClassifierCV(
        estimator=base_rf,
        method="isotonic",
        cv=cv_unshuffled,
    )


def build_xgboost_pipeline(
    calibrate: bool = True,
    random_state: int = 42,
) -> Any:
    """Create an XGBoost Classifier pipeline using objective='multi:softprob' and Isotonic Calibration."""
    base_xgb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "clf",
            XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                n_estimators=85,
                max_depth=2,
                learning_rate=0.025,
                subsample=0.75,
                colsample_bytree=0.60,
                min_child_weight=12,
                reg_alpha=1.5,
                reg_lambda=5.0,
                random_state=random_state,
                n_jobs=1,
            ),
        ),
    ])
    if not calibrate:
        return base_xgb

    cv_unshuffled = StratifiedKFold(n_splits=5, shuffle=False)
    return CalibratedClassifierCV(
        estimator=base_xgb,
        method="isotonic",
        cv=cv_unshuffled,
    )


def _normalize_probabilities(probs: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """Ensure float64 probabilities bounded away from exact 0/1 and normalized to sum to 1.0."""
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    p /= p.sum(axis=1, keepdims=True)
    return p


def run_walk_forward_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_builder: Any,
    model_name: str,
    initial_train_size: int = 760,
    step_size: int = 380,
) -> Dict[str, Any]:
    """Execute Walk-Forward Expanding Window validation and collect out-of-fold predictions.

    Args:
        df: Chronologically sorted DataFrame with features and target FTR.
        feature_cols: List of pre-match feature column names.
        model_builder: Callable returning a fresh estimator pipeline.
        model_name: Name of the model for logging.
        initial_train_size: Starting training window size.
        step_size: Expanding window step size.

    Returns:
        Dictionary with out-of-fold indices, true labels, predicted probabilities, and fold metrics.
    """
    oof_indices: List[int] = []
    oof_y_true: List[int] = []
    oof_y_prob: List[np.ndarray] = []
    fold_summaries: List[Dict[str, Any]] = []

    print(f"\n--- Walk-Forward Expanding Window CV: {model_name} ---")

    for fold, train_idx, test_idx in walk_forward_expanding_splits(
        df, initial_train_size=initial_train_size, step_size=step_size
    ):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        X_train = train_df[feature_cols]
        y_train = train_df["FTR"].to_numpy(dtype=int)

        X_test = test_df[feature_cols]
        y_test = test_df["FTR"].to_numpy(dtype=int)

        model = model_builder()
        model.fit(X_train, y_train)

        probs = _normalize_probabilities(model.predict_proba(X_test))
        preds = np.argmax(probs, axis=1)
        fold_acc = float(np.mean(preds == y_test))

        train_dates = f"{train_df['Date'].min().strftime('%Y-%m-%d')} -> {train_df['Date'].max().strftime('%Y-%m-%d')}"
        test_dates = f"{test_df['Date'].min().strftime('%Y-%m-%d')} -> {test_df['Date'].max().strftime('%Y-%m-%d')}"

        print(
            f"  Fold {fold}: Train={len(train_idx):4d} [{train_dates}] | "
            f"Test={len(test_idx):3d} [{test_dates}] | Acc={fold_acc:.4f}"
        )

        oof_indices.extend(test_idx.tolist())
        oof_y_true.extend(y_test.tolist())
        oof_y_prob.append(probs)

        fold_summaries.append({
            "fold": fold,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "accuracy": fold_acc,
        })

    all_probs = np.vstack(oof_y_prob)
    all_true = np.asarray(oof_y_true, dtype=int)

    return {
        "model_name": model_name,
        "oof_indices": np.asarray(oof_indices, dtype=int),
        "y_true": all_true,
        "y_prob": all_probs,
        "folds": fold_summaries,
    }


def train_and_evaluate_all(
    features_path: Optional[Path] = None,
    model_output_path: Optional[Path] = None,
    high_conviction_threshold: float = 0.65,
) -> Dict[str, Any]:
    """Run full model comparison, out-of-fold evaluation, calibration, and save final pipeline."""
    if features_path is None:
        features_path = Path("data") / "processed" / "features.parquet"
    if model_output_path is None:
        model_output_path = Path("models") / "calibrated_xgb.joblib"

    features_path = Path(features_path)
    model_output_path = Path(model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phases 3 & 4: Walk-Forward Modeling, Calibration & Evaluation")
    print("=" * 70)

    df = pd.read_parquet(features_path)
    # Ensure strictly chronological order
    df = df.sort_values(by=["Date", "HomeTeam"]).reset_index(drop=True)

    # Check for missing values in features before training
    null_count = int(df[FEATURE_COLUMNS].isna().sum().sum())
    print(f"Loaded dataset: {df.shape[0]:,} matches, {len(FEATURE_COLUMNS)} features (Nulls: {null_count})")

    # 1. Baseline Random Forest (Calibrated Isotonic)
    rf_cv = run_walk_forward_cv(
        df=df,
        feature_cols=FEATURE_COLUMNS,
        model_builder=lambda: build_random_forest_pipeline(calibrate=True),
        model_name="Baseline Random Forest (Isotonic Calibrated)",
    )
    rf_eval = evaluate_predictions(
        y_true=rf_cv["y_true"],
        y_prob=rf_cv["y_prob"],
        model_name="Baseline Random Forest (Isotonic)",
        threshold=high_conviction_threshold,
        save_plot_path=model_output_path.parent / "confusion_matrix_rf.png",
    )

    # 2. XGBoost Uncalibrated (multi:softprob) vs Calibrated (Isotonic)
    xgb_raw_cv = run_walk_forward_cv(
        df=df,
        feature_cols=FEATURE_COLUMNS,
        model_builder=lambda: build_xgboost_pipeline(calibrate=False),
        model_name="XGBoost Raw (multi:softprob)",
    )
    xgb_raw_eval = evaluate_predictions(
        y_true=xgb_raw_cv["y_true"],
        y_prob=xgb_raw_cv["y_prob"],
        model_name="XGBoost Raw (multi:softprob)",
        threshold=high_conviction_threshold,
        save_plot_path=model_output_path.parent / "confusion_matrix_xgb_raw.png",
    )

    xgb_cal_cv = run_walk_forward_cv(
        df=df,
        feature_cols=FEATURE_COLUMNS,
        model_builder=lambda: build_xgboost_pipeline(calibrate=True),
        model_name="XGBoost Calibrated (multi:softprob + Isotonic)",
    )
    xgb_cal_eval = evaluate_predictions(
        y_true=xgb_cal_cv["y_true"],
        y_prob=xgb_cal_cv["y_prob"],
        model_name="XGBoost Calibrated (Isotonic)",
        threshold=high_conviction_threshold,
        save_plot_path=model_output_path.parent / "confusion_matrix.png",
    )

    # Print comparison summary
    print("\n" + "=" * 70)
    print("OUT-OF-FOLD VALIDATION METRICS SUMMARY (1,520 Test Matches)")
    print("=" * 70)
    print(
        f"{'Model':<42} | {'Log Loss':>9} | {'Brier':>8} | {'All Acc':>8} | {'HC Acc (>=0.65)':>16}"
    )
    print("-" * 92)
    for res in [rf_eval, xgb_raw_eval, xgb_cal_eval]:
        hc = res["high_conviction"]
        hc_str = (
            f"{hc['accuracy'] * 100:5.1f}% ({hc['qualified_matches']}/{hc['total_matches']})"
            if not np.isnan(hc["accuracy"])
            else "N/A (0 picks)"
        )
        print(
            f"{res['model_name']:<42} | {res['log_loss']:>9.4f} | "
            f"{res['brier_score']:>8.4f} | {res['accuracy'] * 100:>7.2f}% | {hc_str:>16}"
        )
    print("-" * 92)

    # Print High-Conviction Threshold Curve for Calibrated XGBoost
    print("\nHigh-Conviction Threshold Sweep (Calibrated XGBoost):")
    print(f"{'Threshold':<12} | {'Qualified Picks':>16} | {'Coverage':>10} | {'Hit Rate (Accuracy)':>20}")
    print("-" * 66)
    for row in xgb_cal_eval["high_conviction"]["threshold_curve"]:
        acc_str = f"{row['accuracy'] * 100:.2f}%" if not np.isnan(row["accuracy"]) else "N/A"
        print(
            f"P >= {row['threshold']:<7.2f} | {row['matches']:>16d} | "
            f"{row['coverage_pct']:>9.1f}% | {acc_str:>20}"
        )
    print("-" * 66)

    # 3. Fit final production Calibrated XGBoost model on full historical dataset and save
    print(f"\nFitting final Calibrated XGBoost pipeline on all {len(df):,} matches...")
    final_model = build_xgboost_pipeline(calibrate=True)
    final_model.fit(df[FEATURE_COLUMNS], df["FTR"].to_numpy(dtype=int))

    # Save artifact bundle containing calibrated model and metadata
    joblib.dump(final_model, model_output_path)
    print(f"Saved calibrated model pipeline to {model_output_path} ({model_output_path.stat().st_size:,} bytes).")

    # Also save out-of-fold predictions for downstream analysis / dashboard
    oof_df = df.iloc[xgb_cal_cv["oof_indices"]].copy().reset_index(drop=True)
    oof_df["prob_away"] = xgb_cal_cv["y_prob"][:, 0]
    oof_df["prob_draw"] = xgb_cal_cv["y_prob"][:, 1]
    oof_df["prob_home"] = xgb_cal_cv["y_prob"][:, 2]
    oof_df["pred_class"] = np.argmax(xgb_cal_cv["y_prob"], axis=1)
    oof_df["max_prob"] = np.max(xgb_cal_cv["y_prob"], axis=1)
    oof_df["high_conviction"] = oof_df["max_prob"] >= high_conviction_threshold
    oof_output_path = Path("data") / "processed" / "oof_predictions.parquet"
    oof_df.to_parquet(oof_output_path, index=False)
    print(f"Saved out-of-fold predictions to {oof_output_path}.")

    return {
        "rf_eval": rf_eval,
        "xgb_raw_eval": xgb_raw_eval,
        "xgb_cal_eval": xgb_cal_eval,
        "model_path": model_output_path,
    }


if __name__ == "__main__":
    train_and_evaluate_all()
