"""XGBoost training for Robust Anomaly Detection in Noisy Time-Series Data.

Produces the final optimized model with gain-based feature selection (297 features).
Deterministic — same seed + same data = identical model every run.
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
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import engineer_features

SEED = 42

DATA_DIR = "../data"
MODEL_DIR = "model"
PRED_DIR = "predictions"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# Optimized hyperparameters (from temporal CV sweep)
OPTIMIZED_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.15,
    "gamma": 0.1,
    "lambda": 0.5,
    "alpha": 0.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
}


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
    print(f"  Train: {len(train_df)} samples, {train_df['y'].sum()} anomalies ({train_df['y'].mean():.4f})")
    print(f"  Val:   {len(val_df)} samples, {val_df['y'].sum()} anomalies ({val_df['y'].mean():.4f})")
    return train_df, val_df


def select_features(X, y, top_n=300):
    """Gain-based feature selection. Deterministic given same data and seed.

    If an existing metadata.pkl is found, loads feature_cols from it
    to guarantee exact reproducibility with the saved model.
    """
    meta_path = os.path.join(MODEL_DIR, "metadata.pkl")
    if os.path.exists(meta_path):
        stored = joblib.load(meta_path)
        if "feature_cols" in stored:
            cols = stored["feature_cols"]
            print(f"\nFeature selection: {len(cols)} features (loaded from metadata.pkl)")
            print(f"  Top 5: {', '.join(cols[:5])}")
            return cols

    spw = (y == 0).sum() / y.sum()
    dtrain = xgb.DMatrix(X.values, label=y, feature_names=X.columns.tolist())
    params = {"objective": "binary:logistic", "max_depth": 3, "learning_rate": 0.1,
              "scale_pos_weight": spw, "seed": SEED, "verbosity": 0}
    model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
    importance = model.get_score(importance_type="gain")
    sorted_feats = sorted(importance.items(), key=lambda x: -x[1])
    selected = [f for f, _ in sorted_feats[:top_n]]
    print(f"\nFeature selection: {len(selected)} features retained (fresh)")
    print(f"  Top 5: {', '.join(f'{f}({v:.0f})' for f, v in sorted_feats[:5])}")
    return selected


def compute_scale_pos_weight(y):
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    weight = n_neg / n_pos
    print(f"  scale_pos_weight = {weight:.2f} ({n_neg} neg, {n_pos} pos)")
    return weight


def main():
    print("=" * 60)
    print("Robust Anomaly Detection - XGBoost Training")
    print("=" * 60)

    # 1. Load data
    df = load_data()

    # 2. Engineer all temporal features
    print("\n--- Feature Engineering ---")
    X_all = engineer_features(df.drop(columns=["y"]))
    y = df["y"].values

    # 3. Feature selection (deterministic)
    feature_cols = select_features(X_all, y)
    X = X_all[feature_cols]

    # 4. Temporal train/val split
    full_df = pd.concat([X, df[["y"]]], axis=1)
    train_df, val_df = temporal_anomaly_aware_split(full_df)

    X_train = train_df[feature_cols].values
    y_train = train_df["y"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["y"].values

    # 5. Class weight
    scale_pos_weight = compute_scale_pos_weight(y)

    # 6. Train XGBoost with early stopping on validation set
    print("\n--- Training XGBoost ---")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    train_params = {
        **OPTIMIZED_PARAMS,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": scale_pos_weight,
        "seed": SEED,
        "verbosity": 0,
    }
    print(f"  Params: {train_params}")

    val_model = xgb.train(
        train_params, dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=50,
    )

    val_probs = val_model.predict(dval)
    val_aupr = average_precision_score(y_val, val_probs)
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"\n=== Validation Results ===")
    print(f"  AUC-PR:  {val_aupr:.6f}")
    print(f"  AUC-ROC: {val_auc:.6f}")

    # Optimal threshold via max F1
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_thresh = thresholds[np.argmax(f1_scores)]
    print(f"  Best threshold (max F1): {best_thresh:.4f} (F1={f1_scores.max():.4f})")
    best_iter = val_model.best_iteration
    print(f"  Best iteration: {best_iter}")

    # 7. Retrain final model on ALL data
    print("\n" + "=" * 60)
    print("Retraining final model on ALL data...")
    print("=" * 60)

    final_params = {
        **OPTIMIZED_PARAMS,
        "objective": "binary:logistic",
        "scale_pos_weight": scale_pos_weight,
        "seed": SEED,
        "verbosity": 0,
    }
    dfull = xgb.DMatrix(X.values, label=y)
    final_model = xgb.train(final_params, dfull, num_boost_round=best_iter, verbose_eval=100)

    model_path = os.path.join(MODEL_DIR, "xgb_model.json")
    final_model.save_model(model_path)
    print(f"Final model saved to {model_path}")

    meta = {
        "threshold": float(best_thresh),
        "feature_cols": feature_cols,
        "best_iteration": best_iter,
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, "metadata.pkl"))
    print(f"Metadata saved to {MODEL_DIR}/metadata.pkl ({len(feature_cols)} features)")

    # 8. Generate predictions
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

        probs = final_model.predict(xgb.DMatrix(X_test_aligned.values))
        preds = (probs >= best_thresh).astype(int)

        out_path = os.path.join(PRED_DIR, out_file)
        pd.DataFrame({"y_pred": preds}).to_csv(out_path, index=False)
        print(f"  Predictions saved to {out_path}")
        print(f"  Predicted anomalies: {preds.sum()} / {len(preds)} ({preds.mean():.4f})")

    print("\nDone!")


if __name__ == "__main__":
    main()
