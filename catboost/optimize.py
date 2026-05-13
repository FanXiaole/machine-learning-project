"""Efficient hyperparameter optimization for CatBoost.
- 3-fold temporal CV (down from 5)
- 30 trials (down from 50, but CatBoost needs more search than XGBoost)
- 1000 max iterations (down from 2000)
- Focus on stability: heavier regularization, lower depth
"""

import os, sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score
import optuna
import joblib, json, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import engineer_features

SEED = 42
DATA_DIR = "../data"
OUT_DIR = "model"
os.makedirs(OUT_DIR, exist_ok=True)


def get_temporal_splits(n_total, anomaly_start, anomaly_end, n_splits=3):
    splits = np.linspace(anomaly_start + 3000, anomaly_end - 1000, n_splits, dtype=int)
    return splits.tolist()


def load_and_engineer():
    print("Loading and engineering features (one-time)...")
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    X = engineer_features(df.drop(columns=["y"]))
    y = df["y"].values
    full = pd.concat([X, df[["y"]]], axis=1)
    print(f"  Done: {full.shape[1]} features, {full.shape[0]} rows")
    return full


def evaluate_params(full_df, params, split_indices):
    feature_cols = [c for c in full_df.columns if c != "y"]
    auprs, aucs = [], []
    for split_idx in split_indices:
        train_df = full_df.iloc[:split_idx]
        val_df = full_df.iloc[split_idx:]
        X_train = train_df[feature_cols].values
        y_train = train_df["y"].values
        X_val = val_df[feature_cols].values
        y_val = val_df["y"].values
        if y_val.sum() == 0:
            continue

        n_neg = (y_train == 0).sum()
        n_pos = y_train.sum()
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        run_params = params.copy()
        run_params["scale_pos_weight"] = spw
        run_params["random_seed"] = SEED

        try:
            model = CatBoostClassifier(
                **run_params,
                iterations=1000,
                eval_metric="PRAUC",
                early_stopping_rounds=50,
                use_best_model=True,
                verbose=False,
                thread_count=-1,
            )
            model.fit(Pool(X_train, y_train), eval_set=Pool(X_val, y_val), plot=False)
            val_probs = model.predict_proba(X_val)[:, 1]
            auprs.append(average_precision_score(y_val, val_probs))
            aucs.append(roc_auc_score(y_val, val_probs))
        except Exception as e:
            return -1.0, 0.0, 0.0
    return np.mean(auprs), np.mean(aucs), np.std(auprs)


def objective(trial, full_df, split_indices):
    params = {
        "depth": trial.suggest_int("depth", 2, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 8.0),
        "random_strength": trial.suggest_float("random_strength", 0.5, 5.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.5),
    }
    mean_aupr, mean_auc, std_aupr = evaluate_params(full_df, params, split_indices)
    if mean_aupr < 0:
        raise optuna.TrialPruned()
    # Heavier penalty on std since CatBoost has stability issues
    score = mean_aupr - 0.3 * std_aupr
    trial.set_user_attr("mean_auc", mean_auc)
    trial.set_user_attr("std_aupr", std_aupr)
    return score


def main():
    print("=" * 60)
    print("CatBoost Efficient Optimization (3-fold CV, 30 trials)")
    print("=" * 60)

    full_df = load_and_engineer()
    split_indices = get_temporal_splits(137192, 124283, 136582, n_splits=3)
    print(f"\nTemporal CV splits: {split_indices}")

    # Baseline
    print("\n--- Baseline ---")
    baseline_params = {
        "depth": 3, "learning_rate": 0.1, "l2_leaf_reg": 0.5,
        "subsample": 0.8, "colsample_bylevel": 0.8,
    }
    bl_aupr, bl_auc, bl_std = evaluate_params(full_df, baseline_params, split_indices)
    print(f"  Mean AUPR: {bl_aupr:.6f} ± {bl_std:.6f}  |  Mean AUC: {bl_auc:.6f}")

    # Run Optuna
    print("\n--- Running Optuna (30 trials) ---")
    study = optuna.create_study(
        direction="maximize",
        study_name="catboost_efficient",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(
        lambda trial: objective(trial, full_df, split_indices),
        n_trials=30,
        show_progress_bar=True,
    )

    # Results
    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)
    best_params = study.best_params
    best_aupr, best_auc, best_std = evaluate_params(full_df, best_params, split_indices)
    print(f"Best trial: #{study.best_trial.number}  |  Value: {study.best_value:.6f}")
    print(f"Best AUPR: {best_aupr:.6f} ± {best_std:.6f}  |  AUC: {best_auc:.6f}")
    print(f"Improvement: {best_aupr - bl_aupr:.6f} AUPR  |  Std reduction: {bl_std - best_std:.6f}")
    print(f"\nBest params:")
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")

    # Train final model with best params
    print("\n" + "=" * 60)
    print("Training final optimized model on ALL data...")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    X = engineer_features(df.drop(columns=["y"]))
    y = df["y"].values
    feature_cols = X.columns.tolist()
    full_spw = (y == 0).sum() / y.sum()

    # Determine best_iter via CV
    val_cut = int(len(y) * 0.95)
    X_cv_train, X_cv_val = X.values[:val_cut], X.values[val_cut:]
    y_cv_train, y_cv_val = y[:val_cut], y[val_cut:]

    iter_params = best_params.copy()
    iter_params["scale_pos_weight"] = (y_cv_train == 0).sum() / y_cv_train.sum()
    iter_params["random_seed"] = SEED
    cv_model = CatBoostClassifier(
        **iter_params, iterations=1000, eval_metric="PRAUC",
        early_stopping_rounds=50, use_best_model=True, verbose=False, thread_count=-1,
    )
    cv_model.fit(Pool(X_cv_train, y_cv_train), eval_set=Pool(X_cv_val, y_cv_val), plot=False)
    best_iter = cv_model.get_best_iteration()

    final_params = best_params.copy()
    final_params["scale_pos_weight"] = full_spw
    final_params["random_seed"] = SEED
    final_model = CatBoostClassifier(
        **final_params, iterations=best_iter, verbose=100, thread_count=-1,
    )
    final_model.fit(Pool(X.values, y), plot=False)

    model_path = os.path.join(OUT_DIR, "catboost_model_optimized.cbm")
    final_model.save_model(model_path)

    # Threshold
    from sklearn.metrics import precision_recall_curve
    cv_probs = cv_model.predict_proba(X_cv_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_cv_val, cv_probs)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_thresh = thresholds[np.argmax(f1_scores)]
    print(f"  Best threshold: {best_thresh:.4f}  |  Best n_estimators: {best_iter}")

    meta = {"threshold": float(best_thresh), "feature_cols": feature_cols, "best_iteration": best_iter, "best_params": best_params}
    joblib.dump(meta, os.path.join(OUT_DIR, "metadata_optimized.pkl"))

    # Predictions
    print("\n--- Generating Test Predictions ---")
    PRED_DIR = "predictions"
    os.makedirs(PRED_DIR, exist_ok=True)
    for test_file, out_file, task_name in [
        ("test_simple.csv", "pred_simple_optimized.csv", "Task 1"),
        ("test_complex.csv", "pred_complex_optimized.csv", "Task 2"),
    ]:
        test_df = pd.read_csv(os.path.join(DATA_DIR, test_file))
        X_test = engineer_features(test_df)
        X_test_aligned = X_test.reindex(columns=feature_cols, fill_value=0.0)
        probs = final_model.predict_proba(X_test_aligned.values)[:, 1]
        preds = (probs >= best_thresh).astype(int)
        out_path = os.path.join(PRED_DIR, out_file)
        pd.DataFrame({"y_pred": preds}).to_csv(out_path, index=False)
        print(f"  {task_name}: {preds.sum()} / {len(preds)} anomalies ({preds.mean():.4f})")

    # Save results
    results = {"baseline": {"aupr": float(bl_aupr), "auc": float(bl_auc), "std": float(bl_std)}, "optimized": {"params": best_params, "aupr": float(best_aupr), "auc": float(best_auc), "std": float(best_std)}}
    with open(os.path.join(OUT_DIR, "optuna_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
