"""Adversarial feature pruning for robust anomaly detection.

Strategy:
  1. Train an adversary model to distinguish train vs test_complex (hardest shift).
  2. Compute each feature's: (a) anomaly detection importance, (b) adversarial importance.
  3. Remove features with HIGH adversarial importance but LOW anomaly importance.
  4. Retrain both XGBoost and CatBoost on pruned feature set.
  5. Evaluate via temporal CV + generate test predictions.

Core idea: features that discriminate train-vs-test but don't help detect
anomalies are "spurious correlations" that hurt Task 2 generalization.
"""

import os, sys, json
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import joblib, warnings
warnings.filterwarnings("ignore")

SEED = 42
DATA_DIR = "data"
np.random.seed(SEED)

sys.path.insert(0, "xgboost")
from features import engineer_features


def load_and_engineer_all():
    """Engineer features for all three datasets."""
    print("Loading and engineering features...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_simple = pd.read_csv(os.path.join(DATA_DIR, "test_simple.csv"))
    df_complex = pd.read_csv(os.path.join(DATA_DIR, "test_complex.csv"))

    X_train = engineer_features(df_train.drop(columns=["y"]))
    y_train = df_train["y"].values
    X_simple = engineer_features(df_simple)
    X_complex = engineer_features(df_complex)

    return X_train, y_train, X_simple, X_complex


def compute_adversarial_importance(X_train, X_complex):
    """Train adversary to distinguish train vs complex. Return per-feature gain."""
    print("\n=== Computing Adversarial Feature Importance ===")
    n_train = min(30000, len(X_train))
    n_complex = min(30000, len(X_complex))

    train_sample = X_train.sample(n_train, random_state=SEED)
    complex_sample = X_complex.sample(n_complex, random_state=SEED)

    # Label: 0 = train, 1 = complex
    X_adv = pd.concat([train_sample, complex_sample], axis=0)
    y_adv = np.array([0] * n_train + [1] * n_complex)

    # Shuffle
    idx = np.random.permutation(len(X_adv))
    X_adv, y_adv = X_adv.iloc[idx], y_adv[idx]

    dtrain = xgb.DMatrix(X_adv.values, label=y_adv, feature_names=X_adv.columns.tolist())
    params = {"objective": "binary:logistic", "max_depth": 3, "learning_rate": 0.1,
              "seed": SEED, "verbosity": 0, "eval_metric": "auc"}
    model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)

    # Cross-val AUC
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    aucs = []
    for tr, vl in cv.split(X_adv, y_adv):
        dt = xgb.DMatrix(X_adv.values[tr], label=y_adv[tr])
        dv = xgb.DMatrix(X_adv.values[vl], label=y_adv[vl])
        m = xgb.train(params, dt, num_boost_round=200, verbose_eval=False)
        aucs.append(roc_auc_score(y_adv[vl], m.predict(dv)))
    print(f"  Adversary CV AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    importance = model.get_score(importance_type="gain")
    return importance


def compute_anomaly_importance(X_train, y_train):
    """Compute anomaly detection feature importance."""
    print("\n=== Computing Anomaly Detection Feature Importance ===")
    dtrain = xgb.DMatrix(X_train.values, label=y_train, feature_names=X_train.columns.tolist())
    spw = (y_train == 0).sum() / y_train.sum()
    params = {"objective": "binary:logistic", "max_depth": 3, "learning_rate": 0.1,
              "scale_pos_weight": spw, "seed": SEED, "verbosity": 0, "eval_metric": "aucpr"}
    model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
    importance = model.get_score(importance_type="gain")
    return importance


def select_robust_features(anomaly_imp, adversary_imp, X_train, top_n=250, adv_thresh_quantile=0.85):
    """Select features good for anomaly detection but NOT spurious for domain shift.

    Strategy:
      - Keep features with anomaly importance > 0
      - Remove features where adversarial_importance / anomaly_importance is high
        (i.e., feature is more useful for distinguishing domain than detecting anomalies)
    """
    all_feats = set(list(anomaly_imp.keys()) + list(adversary_imp.keys()))

    # Convert to arrays for scoring
    feat_list = []
    anomaly_scores = []
    adversary_scores = []
    for f in all_feats:
        a_imp = anomaly_imp.get(f, 0)
        adv_imp = adversary_imp.get(f, 0)
        feat_list.append(f)
        anomaly_scores.append(a_imp)
        adversary_scores.append(adv_imp)

    anomaly_arr = np.array(anomaly_scores)
    adversary_arr = np.array(adversary_scores)

    # Normalize to [0, 1]
    anomaly_norm = anomaly_arr / (anomaly_arr.max() + 1e-10)
    adversary_norm = adversary_arr / (adversary_arr.max() + 1e-10)

    # Domain-invariance score: high anomaly importance - penalty for high adversarial importance
    # If a feature has HIGH adversarial importance relative to its anomaly importance, penalize it
    adv_penalty = np.where(anomaly_norm > 0.01, adversary_norm / (anomaly_norm + 0.01), adversary_norm * 10)
    robustness_score = anomaly_norm - 0.5 * adv_penalty

    # Sort by robustness score
    ranked = sorted(zip(feat_list, robustness_score, anomaly_norm, adversary_norm),
                    key=lambda x: -x[1])

    # Select top N
    selected = [f for f, _, _, _ in ranked[:top_n]]

    # Report on removed features
    print(f"\n=== Feature Selection Results ===")
    print(f"  Selected: {len(selected)} features")
    print(f"  Removed:  {len(feat_list) - len(selected)} features")

    # Show top removed features (high adversarial, low anomaly)
    removed = [(f, a, adv) for f, _, a, adv in ranked[top_n:]]
    print(f"\n  Top 10 REMOVED features (high adversarial, low anomaly):")
    for f, a_imp, adv_imp in removed[:10]:
        print(f"    {f:30s}  anomaly={a_imp:.4f}  adversary={adv_imp:.4f}")

    # Show top kept features
    print(f"\n  Top 10 KEPT features (robust):")
    for f, score, a_imp, adv_imp in ranked[:10]:
        print(f"    {f:30s}  score={score:.4f}  anomaly={a_imp:.4f}  adversary={adv_imp:.4f}")

    return selected


def temporal_cv_splits():
    return [127283, 131432, 135582]


def cv_eval_xgb(full_df, feature_cols, params, splits):
    auprs = []
    for sp in splits:
        train = full_df.iloc[:sp]
        val = full_df.iloc[sp:]
        Xt, yt = train[feature_cols].values, train["y"].values
        Xv, yv = val[feature_cols].values, val["y"].values
        spw = (yt == 0).sum() / max(yt.sum(), 1)
        p = params.copy()
        p.update({"objective": "binary:logistic", "scale_pos_weight": spw, "seed": SEED,
                   "verbosity": 0, "eval_metric": "aucpr"})
        dtrain = xgb.DMatrix(Xt, label=yt)
        dval = xgb.DMatrix(Xv, label=yv)
        m = xgb.train(p, dtrain, num_boost_round=1000, evals=[(dval, "val")],
                       early_stopping_rounds=50, verbose_eval=False)
        auprs.append(average_precision_score(yv, m.predict(dval)))
    return np.mean(auprs), np.std(auprs)


def cv_eval_cb(full_df, feature_cols, params, splits):
    auprs = []
    for sp in splits:
        train = full_df.iloc[:sp]
        val = full_df.iloc[sp:]
        Xt, yt = train[feature_cols].values, train["y"].values
        Xv, yv = val[feature_cols].values, val["y"].values
        spw = (yt == 0).sum() / max(yt.sum(), 1)
        model = CatBoostClassifier(**params, scale_pos_weight=spw, random_seed=SEED,
                                    iterations=800, eval_metric="PRAUC",
                                    early_stopping_rounds=50, use_best_model=True,
                                    verbose=False, thread_count=-1,
                                    train_dir="catboost/catboost_info")
        model.fit(Pool(Xt, yt), eval_set=Pool(Xv, yv), plot=False)
        auprs.append(average_precision_score(yv, model.predict_proba(Xv)[:, 1]))
    return np.mean(auprs), np.std(auprs)


def main():
    print("=" * 60)
    print("Adversarial Feature Pruning for Robust Anomaly Detection")
    print("=" * 60)

    # 1. Load data
    X_train, y_train, X_simple, X_complex = load_and_engineer_all()

    # 2. Compute importances
    adversary_imp = compute_adversarial_importance(X_train, X_complex)
    anomaly_imp = compute_anomaly_importance(X_train, y_train)

    # 3. Select robust features
    robust_features = select_robust_features(anomaly_imp, adversary_imp, X_train, top_n=250)

    # 4. Baseline CV evaluation with original 297 features
    original_features = list(anomaly_imp.keys())
    print(f"\n=== Baseline CV (original {len(original_features)} features) ===")
    full_df = pd.concat([X_train, pd.Series(y_train, name="y")], axis=1)
    splits = temporal_cv_splits()

    xgb_base_params = {"max_depth": 3, "learning_rate": 0.15, "gamma": 0.1, "lambda": 0.5,
                        "alpha": 0.0, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1}
    cb_base_params = {"depth": 3, "learning_rate": 0.03, "l2_leaf_reg": 10.0,
                       "random_strength": 2.0, "subsample": 1.0, "colsample_bylevel": 0.8,
                       "min_data_in_leaf": 30}

    xgb_orig_aupr, xgb_orig_std = cv_eval_xgb(full_df, original_features, xgb_base_params, splits)
    cb_orig_aupr, cb_orig_std = cv_eval_cb(full_df, original_features, cb_base_params, splits)
    print(f"  XGBoost:  AUPR={xgb_orig_aupr:.4f}±{xgb_orig_std:.4f}")
    print(f"  CatBoost: AUPR={cb_orig_aupr:.4f}±{cb_orig_std:.4f}")

    # 5. Retrain with robust features
    print(f"\n=== Robust CV (pruned {len(robust_features)} features) ===")
    xgb_rob_aupr, xgb_rob_std = cv_eval_xgb(full_df, robust_features, xgb_base_params, splits)
    cb_rob_aupr, cb_rob_std = cv_eval_cb(full_df, robust_features, cb_base_params, splits)
    print(f"  XGBoost:  AUPR={xgb_rob_aupr:.4f}±{xgb_rob_std:.4f}")
    print(f"  CatBoost: AUPR={cb_rob_aupr:.4f}±{cb_rob_std:.4f}")

    # 6. Final models on all data with robust features
    print(f"\n=== Training Final Models (Robust Features) ===")
    X_rob = X_train[robust_features]
    full_spw = (y_train == 0).sum() / y_train.sum()

    # XGBoost final
    val_cut = int(len(y_train) * 0.95)
    xgb_p = xgb_base_params.copy()
    xgb_p.update({"objective": "binary:logistic",
                   "scale_pos_weight": (y_train[:val_cut]==0).sum()/y_train[:val_cut].sum(),
                   "seed": SEED, "verbosity": 0, "eval_metric": "aucpr"})
    dt_cv = xgb.DMatrix(X_rob.values[:val_cut], label=y_train[:val_cut])
    dv_cv = xgb.DMatrix(X_rob.values[val_cut:], label=y_train[val_cut:])
    xgb_cv = xgb.train(xgb_p, dt_cv, num_boost_round=1000, evals=[(dv_cv, "val")],
                        early_stopping_rounds=50, verbose_eval=False)
    best_iter_xgb = xgb_cv.best_iteration
    xgb_final = xgb.train({**xgb_p, "scale_pos_weight": full_spw},
                           xgb.DMatrix(X_rob.values, label=y_train),
                           num_boost_round=best_iter_xgb, verbose_eval=100)

    val_probs = xgb_cv.predict(dv_cv)
    from sklearn.metrics import precision_recall_curve as prc
    p, r, t = prc(y_train[val_cut:], val_probs)
    f1s = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-10)
    xgb_thresh = t[np.argmax(f1s)]
    print(f"  XGBoost: iter={best_iter_xgb}, thresh={xgb_thresh:.4f}, Val AUPR={average_precision_score(y_train[val_cut:], val_probs):.4f}")

    # CatBoost final
    cb_p = cb_base_params.copy()
    cb_cv_train_pool = Pool(X_rob.values[:val_cut], y_train[:val_cut])
    cb_cv_val_pool = Pool(X_rob.values[val_cut:], y_train[val_cut:])
    cb_cv = CatBoostClassifier(**cb_p,
                                scale_pos_weight=(y_train[:val_cut]==0).sum()/y_train[:val_cut].sum(),
                                random_seed=SEED, iterations=800, eval_metric="PRAUC",
                                early_stopping_rounds=50, use_best_model=True,
                                verbose=False, thread_count=-1,
                                train_dir="catboost/catboost_info")
    cb_cv.fit(cb_cv_train_pool, eval_set=cb_cv_val_pool, plot=False)
    best_iter_cb = cb_cv.get_best_iteration()
    cb_final = CatBoostClassifier(**cb_p, scale_pos_weight=full_spw, random_seed=SEED,
                                   iterations=best_iter_cb, verbose=100, thread_count=-1,
                                   train_dir="catboost/catboost_info")
    cb_final.fit(Pool(X_rob.values, y_train), plot=False)

    cb_probs = cb_cv.predict_proba(X_rob.values[val_cut:])[:, 1]
    p, r, t = prc(y_train[val_cut:], cb_probs)
    f1s = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-10)
    cb_thresh = t[np.argmax(f1s)]
    print(f"  CatBoost: iter={best_iter_cb}, thresh={cb_thresh:.4f}, Val AUPR={average_precision_score(y_train[val_cut:], cb_probs):.4f}")

    # 7. Generate test predictions
    print(f"\n=== Test Predictions (Robust Features) ===")
    for test_df, name in [(X_simple, "Task 1"), (X_complex, "Task 2")]:
        X_test_rob = test_df[robust_features]
        xgb_preds = (xgb_final.predict(xgb.DMatrix(X_test_rob.values)) >= xgb_thresh).astype(int)
        cb_preds = (cb_final.predict_proba(X_test_rob.values)[:, 1] >= cb_thresh).astype(int)
        print(f"  {name}: XGB={xgb_preds.sum()}/{len(xgb_preds)}  CB={cb_preds.sum()}/{len(cb_preds)}")

    # 8. Save models
    xgb_final.save_model("xgboost/model/xgb_model_robust.json")
    cb_final.save_model("catboost/model/catboost_model_robust.cbm")
    joblib.dump({"threshold": float(xgb_thresh), "feature_cols": robust_features,
                 "best_iteration": best_iter_xgb}, "xgboost/model/metadata_robust.pkl")
    joblib.dump({"threshold": float(cb_thresh), "feature_cols": robust_features,
                 "best_iteration": best_iter_cb}, "catboost/model/metadata_robust.pkl")

    # 9. Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY: Adversarial Pruning")
    print("=" * 60)
    print(f"  Features: {len(original_features)} → {len(robust_features)}")
    print(f"\n  XGBoost:")
    print(f"    Original:  AUPR={xgb_orig_aupr:.4f}±{xgb_orig_std:.4f}")
    print(f"    Robust:    AUPR={xgb_rob_aupr:.4f}±{xgb_rob_std:.4f}")
    print(f"    Δ:         {xgb_rob_aupr - xgb_orig_aupr:+.4f}")
    print(f"\n  CatBoost:")
    print(f"    Original:  AUPR={cb_orig_aupr:.4f}±{cb_orig_std:.4f}")
    print(f"    Robust:    AUPR={cb_rob_aupr:.4f}±{cb_rob_std:.4f}")
    print(f"    Δ:         {cb_rob_aupr - cb_orig_aupr:+.4f}")
    print("\nDone!")


if __name__ == "__main__":
    main()
