"""Training script for Robust Anomaly Detection using CatBoost.

Same feature engineering and temporal split as the XGBoost baseline.
CatBoost brings ordered boosting, symmetric trees, and built-in class weights.
"""

import os
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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
    """Same temporal split as XGBoost: index 131000."""
    split_idx = 131000
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"\nTemporal split (anomaly-aware):")
    print(f"  Train: rows 0-{split_idx-1} = {len(train_df)} samples")
    print(f"  Val:   rows {split_idx}-{len(df)-1} = {len(val_df)} samples")
    print(f"  Train anomalies: {train_df['y'].sum()} ({train_df['y'].mean():.4f})")
    print(f"  Val anomalies:   {val_df['y'].sum()} ({val_df['y'].mean():.4f})")
    return train_df, val_df


def main():
    print("=" * 60)
    print("Robust Anomaly Detection - CatBoost Training")
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

    # 4. Compute scale_pos_weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"\n  scale_pos_weight = {scale_pos_weight:.2f} ({n_neg} neg, {n_pos} pos)")

    # 5. Train CatBoost
    print("\n--- Training CatBoost ---")

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.1,
        depth=3,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bylevel=0.8,
        l2_leaf_reg=0.5,
        random_seed=SEED,
        eval_metric="PRAUC",
        early_stopping_rounds=100,
        use_best_model=True,
        verbose=50,
        thread_count=-1,
    )

    model.fit(
        train_pool,
        eval_set=val_pool,
        plot=False,
    )

    # 6. Validation evaluation
    val_probs = model.predict_proba(X_val)[:, 1]
    val_aupr = average_precision_score(y_val, val_probs)
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"\n=== Validation Results ===")
    print(f"  AUC-PR:  {val_aupr:.6f}")
    print(f"  AUC-ROC: {val_auc:.6f}")

    # Find optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    print(f"  Best threshold (max F1): {best_thresh:.4f} (F1={best_f1:.4f})")

    val_preds_05 = (val_probs >= 0.5).astype(int)
    val_f1_05 = f1_score(y_val, val_preds_05)
    print(f"  F1 at threshold=0.5: {val_f1_05:.4f}")

    best_iter = model.get_best_iteration()
    print(f"  Best iteration: {best_iter}")

    # 7. Retrain final model on ALL data
    print("\n" + "=" * 60)
    print("Retraining final model on ALL data...")
    print("=" * 60)

    full_pool = Pool(X, y)
    final_model = CatBoostClassifier(
        iterations=best_iter,
        learning_rate=0.1,
        depth=3,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bylevel=0.8,
        l2_leaf_reg=0.5,
        random_seed=SEED,
        verbose=100,
        thread_count=-1,
    )
    final_model.fit(full_pool, plot=False)

    model_path = os.path.join(MODEL_DIR, "catboost_model.cbm")
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

        probs = final_model.predict_proba(X_test_aligned)[:, 1]
        preds = (probs >= best_thresh).astype(int)

        out_path = os.path.join(PRED_DIR, out_file)
        pd.DataFrame({"y_pred": preds}).to_csv(out_path, index=False)
        print(f"  Predictions saved to {out_path}")
        print(f"  Predicted anomalies: {preds.sum()} / {len(preds)} ({preds.mean():.4f})")

    # 9. Feature importance
    print("\n" + "=" * 60)
    print("Top 15 Feature Importances")
    print("=" * 60)
    importance = final_model.get_feature_importance()
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    for rank, (feat, imp) in enumerate(feat_imp[:15], 1):
        print(f"  {rank:2d}. {feat:30s} {imp:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
