"""Distribution-Invariant Feature Optimization (Direction A).

Adds MAD, median, IQR, and percentile-rank features that maintain
consistent semantics across distribution shifts.

Evaluates whether these features improve temporal CV AUPR
compared to the original feature set.
"""

import os, sys, json
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import joblib, warnings, time
warnings.filterwarnings("ignore")

SEED = 42
DATA_DIR = "data"
np.random.seed(SEED)

sys.path.insert(0, "xgboost")
from features import engineer_features


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    return df


def temporal_cv_splits():
    return [127283, 131432, 135582]


def select_features_gain(X, y, top_n=300):
    """Feature selection via XGBoost gain importance."""
    spw = (y == 0).sum() / y.sum()
    dtrain = xgb.DMatrix(X.values, label=y, feature_names=X.columns.tolist())
    params = {"objective": "binary:logistic", "max_depth": 3, "learning_rate": 0.1,
              "scale_pos_weight": spw, "seed": SEED, "verbosity": 0}
    model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
    importance = model.get_score(importance_type="gain")
    sorted_feats = sorted(importance.items(), key=lambda x: -x[1])
    top_feats = [f for f, _ in sorted_feats[:top_n]]
    top_gain = sum(v for _, v in sorted_feats[:top_n])
    total_gain = sum(v for _, v in sorted_feats)
    print(f"  Top {len(top_feats)} features: {top_gain/total_gain*100:.1f}% gain")
    print(f"  Top 5: {', '.join(f'{f}({v:.0f})' for f, v in sorted_feats[:5])}")
    return top_feats


def cv_eval_xgb(full_df, feature_cols, params, splits):
    auprs = []
    for sp in splits:
        train = full_df.iloc[:sp]; val = full_df.iloc[sp:]
        Xt, yt = train[feature_cols].values, train["y"].values
        Xv, yv = val[feature_cols].values, val["y"].values
        spw = (yt == 0).sum() / max(yt.sum(), 1)
        p = {**params, "objective": "binary:logistic", "scale_pos_weight": spw,
             "seed": SEED, "verbosity": 0, "eval_metric": "aucpr"}
        dt = xgb.DMatrix(Xt, label=yt); dv = xgb.DMatrix(Xv, label=yv)
        m = xgb.train(p, dt, num_boost_round=1000, evals=[(dv, "val")],
                       early_stopping_rounds=50, verbose_eval=False)
        auprs.append(average_precision_score(yv, m.predict(dv)))
    return np.mean(auprs), np.std(auprs)


def cv_eval_cb(full_df, feature_cols, params, splits):
    auprs = []
    for sp in splits:
        train = full_df.iloc[:sp]; val = full_df.iloc[sp:]
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
    t0 = time.time()
    print("=" * 60)
    print("Direction A: Distribution-Invariant Features")
    print("=" * 60)

    # 1. Load + engineer with robust features
    print("\n--- Phase 1: Feature Engineering (robust=True) ---")
    df = load_data()
    X_robust = engineer_features(df.drop(columns=["y"]), robust=True)
    y = df["y"].values
    print(f"  Total robust features: {X_robust.shape[1]}")

    # 2. Feature selection
    print("\n--- Phase 2: Feature Selection ---")
    top_features = select_features_gain(X_robust, y, top_n=300)

    # 3. Compare with original features
    print("\n--- Phase 3: CV Comparison (Original vs Robust) ---")
    splits = temporal_cv_splits()

    # Train original features for comparison
    X_orig = engineer_features(df.drop(columns=["y"]), robust=False)
    orig_features = select_features_gain(X_orig, y, top_n=300)

    full_orig = pd.concat([X_orig, pd.Series(y, name="y")], axis=1)
    full_rob = pd.concat([X_robust, pd.Series(y, name="y")], axis=1)

    xgb_params = {"max_depth": 3, "learning_rate": 0.15, "gamma": 0.1, "lambda": 0.5,
                   "alpha": 0.0, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1}
    cb_params = {"depth": 3, "learning_rate": 0.03, "l2_leaf_reg": 10.0,
                  "random_strength": 2.0, "subsample": 1.0, "colsample_bylevel": 0.8,
                  "min_data_in_leaf": 30}

    print("\n  Original features:")
    xgb_orig, xgb_orig_std = cv_eval_xgb(full_orig, orig_features, xgb_params, splits)
    cb_orig, cb_orig_std = cv_eval_cb(full_orig, orig_features, cb_params, splits)
    print(f"    XGBoost:  AUPR={xgb_orig:.4f}±{xgb_orig_std:.4f}")
    print(f"    CatBoost: AUPR={cb_orig:.4f}±{cb_orig_std:.4f}")

    print("\n  Robust features:")
    xgb_rob, xgb_rob_std = cv_eval_xgb(full_rob, top_features, xgb_params, splits)
    cb_rob, cb_rob_std = cv_eval_cb(full_rob, top_features, cb_params, splits)
    print(f"    XGBoost:  AUPR={xgb_rob:.4f}±{xgb_rob_std:.4f}")
    print(f"    CatBoost: AUPR={cb_rob:.4f}±{cb_rob_std:.4f}")

    print(f"\n  Δ XGBoost:  {xgb_rob - xgb_orig:+.4f}")
    print(f"  Δ CatBoost: {cb_rob - cb_orig:+.4f}")

    # 4. Feature importance analysis — which features dominate?
    print("\n--- Phase 4: Robust Feature Importance Analysis ---")
    spw = (y == 0).sum() / y.sum()
    dt = xgb.DMatrix(X_robust[top_features].values, label=y, feature_names=top_features)
    imp_model = xgb.train({**xgb_params, "objective": "binary:logistic",
                            "scale_pos_weight": spw, "seed": SEED, "verbosity": 0},
                           dt, num_boost_round=200, verbose_eval=False)
    importance = imp_model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])

    # Count feature types in top 30
    type_counts = {"rs": 0, "rm": 0, "rmax": 0, "rmin": 0, "z": 0, "lag": 0, "diff": 0,
                   "rmad": 0, "rmed": 0, "riqr": 0, "rz": 0, "prank": 0, "other": 0}
    for feat, _ in sorted_imp[:30]:
        matched = False
        for suffix in ["rmad", "rmed", "riqr", "rz", "prank", "rs", "rmax", "rmin",
                        "rm", "z", "lag", "diff"]:
            if f"_{suffix}" in feat:
                type_counts[suffix] += 1
                matched = True
                break
        if not matched:
            type_counts["other"] += 1

    print("  Feature types in top 30 by gain:")
    for k, v in sorted(type_counts.items(), key=lambda x: -x[1]):
        if v > 0:
            print(f"    {k:8s}: {v}")

    # 5. Train final models with robust features (best performing set)
    print("\n--- Phase 5: Final Models + Test Predictions ---")
    X_final = X_robust[top_features]

    # Determine which features to use (original or robust)
    use_robust = xgb_rob > xgb_orig
    final_features = top_features if use_robust else orig_features
    X_final_feat = X_robust[final_features] if use_robust else X_orig[final_features]
    print(f"  Using {'robust' if use_robust else 'original'} features ({len(final_features)} features)")

    full_spw = (y == 0).sum() / y.sum()
    val_cut = int(len(y) * 0.95)

    # XGBoost
    xgb_p = {**xgb_params, "objective": "binary:logistic",
              "scale_pos_weight": (y[:val_cut]==0).sum()/y[:val_cut].sum(),
              "seed": SEED, "verbosity": 0, "eval_metric": "aucpr"}
    dt_cv = xgb.DMatrix(X_final_feat.values[:val_cut], label=y[:val_cut])
    dv_cv = xgb.DMatrix(X_final_feat.values[val_cut:], label=y[val_cut:])
    xgb_cv = xgb.train(xgb_p, dt_cv, num_boost_round=1000, evals=[(dv_cv, "val")],
                        early_stopping_rounds=50, verbose_eval=False)
    xgb_final = xgb.train({**xgb_p, "scale_pos_weight": full_spw},
                           xgb.DMatrix(X_final_feat.values, label=y),
                           num_boost_round=xgb_cv.best_iteration, verbose_eval=100)

    xgb_probs = xgb_cv.predict(dv_cv)
    p, r, t = precision_recall_curve(y[val_cut:], xgb_probs)
    f1s = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-10)
    xgb_thresh = t[np.argmax(f1s)]
    print(f"  XGBoost: iter={xgb_cv.best_iteration}, thresh={xgb_thresh:.4f}, Val AUPR={average_precision_score(y[val_cut:], xgb_probs):.4f}")

    # CatBoost
    cb_cv = CatBoostClassifier(**cb_params,
                                scale_pos_weight=(y[:val_cut]==0).sum()/y[:val_cut].sum(),
                                random_seed=SEED, iterations=800, eval_metric="PRAUC",
                                early_stopping_rounds=50, use_best_model=True,
                                verbose=False, thread_count=-1,
                                train_dir="catboost/catboost_info")
    cb_cv.fit(Pool(X_final_feat.values[:val_cut], y[:val_cut]),
              eval_set=Pool(X_final_feat.values[val_cut:], y[val_cut:]), plot=False)
    cb_final = CatBoostClassifier(**cb_params, scale_pos_weight=full_spw, random_seed=SEED,
                                   iterations=cb_cv.get_best_iteration(),
                                   verbose=100, thread_count=-1,
                                   train_dir="catboost/catboost_info")
    cb_final.fit(Pool(X_final_feat.values, y), plot=False)

    cb_probs = cb_cv.predict_proba(X_final_feat.values[val_cut:])[:, 1]
    p, r, t = precision_recall_curve(y[val_cut:], cb_probs)
    f1s = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-10)
    cb_thresh = t[np.argmax(f1s)]
    print(f"  CatBoost: iter={cb_cv.get_best_iteration()}, thresh={cb_thresh:.4f}, Val AUPR={average_precision_score(y[val_cut:], cb_probs):.4f}")

    # Test predictions
    print("\n--- Test Predictions ---")
    for test_file, task_name in [("test_simple.csv", "Task 1"), ("test_complex.csv", "Task 2")]:
        test_df = pd.read_csv(os.path.join(DATA_DIR, test_file))
        X_test = engineer_features(test_df, robust=use_robust)
        X_test = X_test.reindex(columns=final_features, fill_value=0.0)

        xgb_probs_t = xgb_final.predict(xgb.DMatrix(X_test.values))
        cb_probs_t = cb_final.predict_proba(X_test.values)[:, 1]

        xgb_preds = (xgb_probs_t >= xgb_thresh).astype(int)
        cb_preds = (cb_probs_t >= cb_thresh).astype(int)
        print(f"  {task_name}: XGB={xgb_preds.sum()}/{len(xgb_preds)}  CB={cb_preds.sum()}/{len(cb_preds)}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Distribution-Invariant Features")
    print("=" * 60)
    print(f"  Time: {(time.time()-t0)/60:.1f} min")
    print(f"\n  CV AUPR Comparison:")
    print(f"    XGBoost  original: {xgb_orig:.4f}±{xgb_orig_std:.4f}")
    print(f"    XGBoost  robust:   {xgb_rob:.4f}±{xgb_rob_std:.4f}  (Δ={xgb_rob-xgb_orig:+.4f})")
    print(f"    CatBoost original: {cb_orig:.4f}±{cb_orig_std:.4f}")
    print(f"    CatBoost robust:   {cb_rob:.4f}±{cb_rob_std:.4f}  (Δ={cb_rob-cb_orig:+.4f})")
    print(f"\n  Using for submission: {'robust' if use_robust else 'original'} features")

    print("\nDone!")


if __name__ == "__main__":
    main()
