"""Training script for Robust Anomaly Detection in Noisy Time-Series Data.

Uses XGBoost with engineered temporal features.
Strictly no data leakage — all validation respects temporal order.

Key design details:
- Anomalies (570 total) form 19 streaks of 30 consecutive timesteps,
  concentrated in the last ~10% of data (indices 124283-136582).
- Validation split is placed *within* the anomaly region so both
  training and validation sets contain anomalies.
- Final model is retrained on ALL labeled data after hyperparameter selection.
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import engineer_features

SEED = 42
np.random.seed(SEED)

DATA_DIR = "data"
MODEL_DIR = "model"
PRED_DIR = "predictions"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


def load_data():
    """Load training data."""
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    print(f"Loaded training data: {df.shape}")
    print(f"  Positive class ratio: {df['y'].mean():.6f} ({df['y'].sum()} anomalies)")
    return df


def temporal_anomaly_aware_split(df):
    """Temporal split that puts some anomaly segments in both train and val.

    Anomalies span indices 124283-136582 (19 segments of 30).
    We split at row 131000 so training gets the first ~10 segments,
    validation gets the last ~9 segments.
    """
    split_idx = 131000
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"\nTemporal split (anomaly-aware):")
    print(f"  Train: rows 0-{split_idx-1} = {len(train_df)} samples")
    print(f"  Val:   rows {split_idx}-{len(df)-1} = {len(val_df)} samples")
    print(f"  Train anomalies: {train_df['y'].sum()} ({train_df['y'].mean():.4f})")
    print(f"  Val anomalies:   {val_df['y'].sum()} ({val_df['y'].mean():.4f})")
    return train_df, val_df


def compute_scale_pos_weight(y):
    """Compute scale_pos_weight for XGBoost to handle class imbalance."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    weight = n_neg / n_pos
    print(f"  scale_pos_weight = {weight:.2f} ({n_neg} neg, {n_pos} pos)")
    return weight


def _aupr_metric(predt, dtrain):
    """Custom metric: AUPR (higher is better)."""
    y = dtrain.get_label()
    try:
        score = average_precision_score(y, predt)
    except ValueError:
        score = 0.0
    return "AUPR", score


def train_xgboost(X_train, y_train, X_val, y_val, scale_pos_weight):
    """Train XGBoost classifier with early stopping and AUPR eval."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": scale_pos_weight,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.5,
        "lambda": 0.5,
        "alpha": 0.0,
        "seed": SEED,
        "verbosity": 0,
    }

    print(f"\nXGBoost params: {params}")

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=100,
        custom_metric=_aupr_metric,
        maximize=True,
        verbose_eval=50,
    )

    # Evaluate on validation set
    val_probs = model.predict(dval)
    val_aupr = average_precision_score(y_val, val_probs)
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"\n=== Validation Results ===")
    print(f"  AUC-PR:  {val_aupr:.6f}")
    print(f"  AUC-ROC: {val_auc:.6f}")

    # Find optimal threshold from validation PR curve (max F1)
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    print(f"  Best threshold (max F1): {best_thresh:.4f} (F1={best_f1:.4f})")

    # Also report F1 at default 0.5
    val_preds_05 = (val_probs >= 0.5).astype(int)
    val_f1_05 = f1_score(y_val, val_preds_05)
    print(f"  F1 at threshold=0.5: {val_f1_05:.4f}")

    return model, best_thresh


def main():
    print("=" * 60)
    print("Robust Anomaly Detection - XGBoost Training")
    print("=" * 60)

    # 1. Load data
    df = load_data()

    # 2. Engineer features for training
    print("\n--- Feature Engineering ---")
    X = engineer_features(df.drop(columns=["y"]))
    y = df["y"].values

    # 3. Temporal train/val split (anomaly-aware)
    full_df = pd.concat([X, df[["y"]]], axis=1)
    train_df, val_df = temporal_anomaly_aware_split(full_df)

    feature_cols = [c for c in train_df.columns if c != "y"]
    X_train = train_df[feature_cols].values
    y_train = train_df["y"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["y"].values

    # 4. Compute class weight (from full training set, not just train fold)
    scale_pos_weight = compute_scale_pos_weight(y)

    # 5. Train XGBoost with early stopping
    val_model, best_thresh = train_xgboost(X_train, y_train, X_val, y_val, scale_pos_weight)

    # 6. Retrain FINAL model on ALL labeled data (best params + full data)
    print("\n" + "=" * 60)
    print("Retraining final model on ALL data...")
    print("=" * 60)

    dfull = xgb.DMatrix(X.values, label=y)
    final_params = {
        "objective": "binary:logistic",
        "scale_pos_weight": scale_pos_weight,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.5,
        "lambda": 0.5,
        "alpha": 0.0,
        "seed": SEED,
        "verbosity": 0,
    }
    best_iter = val_model.best_iteration if hasattr(val_model, "best_iteration") else 500
    final_model = xgb.train(
        final_params,
        dfull,
        num_boost_round=best_iter,
        verbose_eval=100,
    )

    model_path = os.path.join(MODEL_DIR, "xgb_model.json")
    final_model.save_model(model_path)
    print(f"Final model saved to {model_path}")

    meta = {
        "threshold": float(best_thresh),
        "feature_cols": feature_cols,
        "best_iteration": best_iter,
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, "metadata.pkl"))
    print(f"Metadata saved to {MODEL_DIR}/metadata.pkl")

    # 7. Generate predictions for both test sets
    print("\n" + "=" * 60)
    print("Generating Test Predictions")
    print("=" * 60)

    for test_file, out_file, task_name in [
        ("test_simple.csv", "pred_simple.csv", "Task 1"),
        ("test_complex.csv", "pred_complex.csv", "Task 2"),
    ]:
        print(f"\n--- {task_name}: {test_file} ---")
        test_df = pd.read_csv(os.path.join(DATA_DIR, test_file))

        X_test = engineer_features(test_df)
        # Align test columns to match training (reindex handles missing cols)
        X_test_aligned = X_test.reindex(columns=feature_cols, fill_value=0.0)

        dtest = xgb.DMatrix(X_test_aligned.values)
        probs = final_model.predict(dtest)
        preds = (probs >= best_thresh).astype(int)

        out_path = os.path.join(PRED_DIR, out_file)
        pd.DataFrame({"y_pred": preds}).to_csv(out_path, index=False)
        print(f"  Predictions saved to {out_path}")
        print(f"  Predicted anomalies: {preds.sum()} / {len(preds)} ({preds.mean():.4f})")

    print("\nDone!")


if __name__ == "__main__":
    main()
