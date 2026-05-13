"""Training script for Robust Anomaly Detection using LightGBM.

LightGBM uses leaf-wise tree growth and histogram-based splitting,
making it faster than XGBoost while potentially more precise on
sharp anomaly boundaries due to asymmetric leaf expansion.
"""

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
)
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import engineer_features

SEED = 42
np.random.seed(SEED)

DATA_DIR = "../data"
MODEL_DIR = "model"
PRED_DIR = "predictions"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    print(f"Loaded training data: {df.shape}")
    print(f"  Positive class ratio: {df['y'].mean():.6f} ({df['y'].sum()} anomalies)")
    return df


def temporal_anomaly_aware_split(df):
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
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    weight = n_neg / n_pos
    print(f"  scale_pos_weight = {weight:.2f} ({n_neg} neg, {n_pos} pos)")
    return weight


def main():
    print("=" * 60)
    print("Robust Anomaly Detection - LightGBM Training")
    print("=" * 60)

    # 1. Load data
    df = load_data()

    # 2. Engineer features
    print("\n--- Feature Engineering ---")
    X = engineer_features(df.drop(columns=["y"]))
    y = df["y"].values

    # 3. Temporal train/val split
    full_df = pd.concat([X, df[["y"]]], axis=1)
    train_df, val_df = temporal_anomaly_aware_split(full_df)

    feature_cols = [c for c in train_df.columns if c != "y"]
    X_train = train_df[feature_cols]
    y_train = train_df["y"].values
    X_val = val_df[feature_cols]
    y_val = val_df["y"].values

    # 4. Compute class weight
    scale_pos_weight = compute_scale_pos_weight(y)

    # 5. Train LightGBM
    print("\n--- Training LightGBM ---")

    # LightGBM needs is_unbalance for extreme class imbalance
    # (internally scales gradients differently than scale_pos_weight)
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "max_depth": 3,
        "learning_rate": 0.1,
        "is_unbalance": True,           # auto-balance positive/negative weights
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.5,
        "reg_alpha": 0.0,
        "min_child_samples": 10,
        "min_split_gain": 0.1,
        "max_bin": 255,                 # more bins for finer splits
        "seed": SEED,
        "verbosity": -1,
        "force_col_wise": True,
    }

    print(f"  LightGBM params: {params}")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50),
        ],
    )

    # 6. Validation evaluation
    val_probs = model.predict(X_val)
    val_aupr = average_precision_score(y_val, val_probs)
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"\n=== Validation Results ===")
    print(f"  AUC-PR:  {val_aupr:.6f}")
    print(f"  AUC-ROC: {val_auc:.6f}")

    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    print(f"  Best threshold (max F1): {best_thresh:.4f} (F1={best_f1:.4f})")

    val_preds_05 = (val_probs >= 0.5).astype(int)
    val_f1_05 = f1_score(y_val, val_preds_05)
    print(f"  F1 at threshold=0.5: {val_f1_05:.4f}")

    best_iter = model.best_iteration
    print(f"  Best iteration: {best_iter}")

    # 7. Retrain final model on ALL data
    print("\n" + "=" * 60)
    print("Retraining final model on ALL data...")
    print("=" * 60)

    full_data = lgb.Dataset(X.values, label=y)
    final_params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "max_depth": 3,
        "learning_rate": 0.1,
        "is_unbalance": True,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.5,
        "reg_alpha": 0.0,
        "min_child_samples": 10,
        "min_split_gain": 0.1,
        "max_bin": 255,
        "seed": SEED,
        "verbosity": -1,
        "force_col_wise": True,
    }

    final_model = lgb.train(
        final_params,
        full_data,
        num_boost_round=best_iter,
        callbacks=[lgb.log_evaluation(period=100)],
    )

    model_path = os.path.join(MODEL_DIR, "lgb_model.txt")
    final_model.save_model(model_path)
    print(f"Final model saved to {model_path}")

    meta = {
        "threshold": float(best_thresh),
        "feature_cols": feature_cols,
        "best_iteration": best_iter,
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, "metadata.pkl"))
    print(f"Metadata saved to {MODEL_DIR}/metadata.pkl")

    # 8. Generate predictions for both test sets
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
        X_test_aligned = X_test.reindex(columns=feature_cols, fill_value=0.0)

        probs = final_model.predict(X_test_aligned.values)
        preds = (probs >= best_thresh).astype(int)

        out_path = os.path.join(PRED_DIR, out_file)
        pd.DataFrame({"y_pred": preds}).to_csv(out_path, index=False)
        print(f"  Predictions saved to {out_path}")
        print(f"  Predicted anomalies: {preds.sum()} / {len(preds)} ({preds.mean():.4f})")

    # 9. Feature importance
    print("\n" + "=" * 60)
    print("Top 15 Feature Importances (gain)")
    print("=" * 60)
    importance = final_model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    for rank, (feat, imp) in enumerate(feat_imp[:15], 1):
        print(f"  {rank:2d}. {feat:30s} {imp:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
