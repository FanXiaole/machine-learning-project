"""Efficient hyperparameter optimization for XGBoost.
- 3-fold temporal CV (down from 5)
- 20 trials (down from 50)
- 1000 max boosting rounds (down from 2000)
- Narrower search around known good region
- Also tests feature pruning
"""

import os, sys
import numpy as np
import pandas as pd
import xgboost as xgb
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

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        run_params = params.copy()
        run_params["scale_pos_weight"] = spw
        run_params["seed"] = SEED
        run_params["verbosity"] = 0
        run_params["objective"] = "binary:logistic"
        run_params["eval_metric"] = "aucpr"

        try:
            model = xgb.train(
                run_params, dtrain,
                num_boost_round=1000,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            val_probs = model.predict(dval)
            auprs.append(average_precision_score(y_val, val_probs))
            aucs.append(roc_auc_score(y_val, val_probs))
        except Exception as e:
            return -1.0, 0.0, 0.0
    return np.mean(auprs), np.mean(aucs), np.std(auprs)


def objective(trial, full_df, split_indices):
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "lambda": trial.suggest_float("lambda", 0.1, 3.0),
        "alpha": trial.suggest_float("alpha", 0.0, 1.5),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 8.0),
    }
    mean_aupr, mean_auc, std_aupr = evaluate_params(full_df, params, split_indices)
    if mean_aupr < 0:
        raise optuna.TrialPruned()
    score = mean_aupr - 0.2 * std_aupr  # heavier penalty on variance
    trial.set_user_attr("mean_auc", mean_auc)
    trial.set_user_attr("std_aupr", std_aupr)
    return score


def main():
    print("=" * 60)
    print("XGBoost Efficient Optimization (3-fold CV, 20 trials)")
    print("=" * 60)

    full_df = load_and_engineer()
    split_indices = get_temporal_splits(137192, 124283, 136582, n_splits=3)
    print(f"\nTemporal CV splits: {split_indices}")

    # Baseline
    print("\n--- Baseline ---")
    baseline_params = {
        "max_depth": 3, "learning_rate": 0.1, "gamma": 0.5,
        "lambda": 0.5, "alpha": 0.0, "subsample": 0.8,
        "colsample_bytree": 0.8, "min_child_weight": 1,
    }
    bl_aupr, bl_auc, bl_std = evaluate_params(full_df, baseline_params, split_indices)
    print(f"  Mean AUPR: {bl_aupr:.6f} ± {bl_std:.6f}  |  Mean AUC: {bl_auc:.6f}")

    # Run Optuna
    print("\n--- Running Optuna (20 trials) ---")
    study = optuna.create_study(
        direction="maximize",
        study_name="xgb_efficient",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(
        lambda trial: objective(trial, full_df, split_indices),
        n_trials=20,
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
    print(f"Improvement: {best_aupr - bl_aupr:.6f} AUPR")
    print(f"\nBest params:")
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")

    # Also test feature pruning
    print("\n--- Testing Feature Pruning ---")
    feature_cols = [c for c in full_df.columns if c != "y"]
    # Train a quick model to get importance
    dtrain = xgb.DMatrix(full_df[feature_cols].values[:100000], label=full_df["y"].values[:100000])
    tmp_params = best_params.copy()
    tmp_params.update({"objective": "binary:logistic", "scale_pos_weight": (full_df["y"].values[:100000]==0).sum() / full_df["y"].values[:100000].sum(), "seed": SEED, "verbosity": 0})
    tmp_model = xgb.train(tmp_params, dtrain, num_boost_round=100, verbose_eval=False)
    importance = tmp_model.get_score(importance_type="gain")
    sorted_feats = sorted(importance.items(), key=lambda x: -x[1])

    for top_n in [200, 300, 500]:
        top_feats = [f for f, _ in sorted_feats[:top_n]]
        if len(top_feats) < top_n:
            continue
        subset_df = full_df[top_feats + ["y"]]
        pr_aupr, pr_auc, pr_std = evaluate_params(subset_df, best_params, split_indices)
        print(f"  Top {top_n} features: AUPR={pr_aupr:.6f} ± {pr_std:.6f}")

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
    final_params_base = best_params.copy()
    final_params_base.update({"objective": "binary:logistic", "scale_pos_weight": (y[:val_cut]==0).sum() / y[:val_cut].sum(), "seed": SEED, "verbosity": 0})
    dtrain_cv = xgb.DMatrix(X.values[:val_cut], label=y[:val_cut])
    dval_cv = xgb.DMatrix(X.values[val_cut:], label=y[val_cut:])
    cv_model = xgb.train(final_params_base, dtrain_cv, num_boost_round=1000, evals=[(dval_cv, "val")], early_stopping_rounds=50, verbose_eval=False)
    best_iter = cv_model.best_iteration

    final_params = best_params.copy()
    final_params.update({"objective": "binary:logistic", "scale_pos_weight": full_spw, "seed": SEED, "verbosity": 1})
    dfull = xgb.DMatrix(X.values, label=y)
    final_model = xgb.train(final_params, dfull, num_boost_round=best_iter, verbose_eval=100)

    model_path = os.path.join(OUT_DIR, "xgb_model_optimized.json")
    final_model.save_model(model_path)

    # Threshold
    from sklearn.metrics import precision_recall_curve
    val_probs = cv_model.predict(dval_cv)
    precisions, recalls, thresholds = precision_recall_curve(y[val_cut:], val_probs)
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
        dtest = xgb.DMatrix(X_test_aligned.values)
        probs = final_model.predict(dtest)
        preds = (probs >= best_thresh).astype(int)
        out_path = os.path.join(PRED_DIR, out_file)
        pd.DataFrame({"y_pred": preds}).to_csv(out_path, index=False)
        print(f"  {task_name}: {preds.sum()} / {len(preds)} anomalies ({preds.mean():.4f})")

    # Save results
    results = {"baseline": {"aupr": float(bl_aupr), "auc": float(bl_auc)}, "optimized": {"params": best_params, "aupr": float(best_aupr), "auc": float(best_auc)}}
    with open(os.path.join(OUT_DIR, "optuna_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
