"""Efficient targeted optimization for both XGBoost and CatBoost.

Strategy (knowledge-driven, not random search):
  Phase 1 — Feature selection via XGBoost gain importance (top 300)
  Phase 2 — CatBoost targeted grid search: 3 rounds × ~10 combos each
            Focus: reduce variance via l2_leaf_reg, random_strength, min_data_in_leaf
  Phase 3 — XGBoost: validate baseline on reduced features, quick lr/depth check
  Phase 4 — Train final models + generate predictions
"""

import os, sys, json, itertools
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import joblib, warnings, time
warnings.filterwarnings("ignore")

SEED = 42
DATA_DIR = "data"
FULL_DATA_PATH = os.path.join(DATA_DIR, "train.csv")

# ---- Phase 0: Feature engineering (shared) ----
sys.path.insert(0, "xgboost")
from features import engineer_features


def load_full():
    df = pd.read_csv(FULL_DATA_PATH)
    X = engineer_features(df.drop(columns=["y"]))
    y = df["y"].values
    cols = X.columns.tolist()
    print(f"  Features: {len(cols)}  |  Samples: {len(y)}  |  Anomaly rate: {y.mean():.4f}")
    return X, y, cols


def temporal_cv_splits(n_total=137192, anomaly_start=124283, anomaly_end=136582, n_splits=3):
    return np.linspace(anomaly_start + 3000, anomaly_end - 1000, n_splits, dtype=int).tolist()


# ---- Phase 1: Feature selection ----
def select_features(X, y, top_n=300):
    """Quick XGBoost training to get gain-based feature importance, keep top N."""
    print(f"\n{'='*60}")
    print(f"Phase 1: Feature Selection (top {top_n} by gain)")
    print("=" * 60)

    spw = (y == 0).sum() / y.sum()
    dtrain = xgb.DMatrix(X.values, label=y, feature_names=X.columns.tolist())

    params = {
        "objective": "binary:logistic", "max_depth": 3, "learning_rate": 0.1,
        "scale_pos_weight": spw, "subsample": 0.8, "colsample_bytree": 0.8,
        "gamma": 0.5, "lambda": 0.5, "seed": SEED, "verbosity": 0,
    }
    model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)

    importance = model.get_score(importance_type="gain")
    sorted_feats = sorted(importance.items(), key=lambda x: -x[1])
    top_feats = [f for f, _ in sorted_feats[:top_n]]

    total_gain = sum(v for _, v in importance.items())
    top_gain = sum(v for _, v in sorted_feats[:top_n])
    print(f"  Top {top_n} features capture {top_gain/total_gain*100:.1f}% of total gain")
    print(f"  Top 5: {', '.join(f'{f}({v:.1f})' for f, v in sorted_feats[:5])}")

    return top_feats, importance


# ---- Phase 2: CatBoost targeted grid search ----
def eval_catboost(X_train, y_train, X_val, y_val, params):
    """Single CatBoost train + eval. Returns (aupr, auc)."""
    spw = (y_train == 0).sum() / max(y_train.sum(), 1)
    model = CatBoostClassifier(
        **params, scale_pos_weight=spw, random_seed=SEED,
        iterations=800, eval_metric="PRAUC", early_stopping_rounds=50,
        use_best_model=True, verbose=False, thread_count=-1,
        train_dir="catboost/catboost_info",
    )
    model.fit(Pool(X_train, y_train), eval_set=Pool(X_val, y_val), plot=False)
    probs = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, probs), roc_auc_score(y_val, probs)


def catboost_grid_search(full_df, feature_cols, splits):
    """3-round targeted grid search for CatBoost."""
    print(f"\n{'='*60}")
    print("Phase 2: CatBoost Targeted Grid Search")
    print("=" * 60)
    print(f"  Features: {len(feature_cols)}  |  CV splits: {splits}")

    def cv_eval(params):
        """3-fold CV evaluation."""
        auprs, aucs = [], []
        for sp in splits:
            train = full_df.iloc[:sp]
            val = full_df.iloc[sp:]
            Xt, yt = train[feature_cols].values, train["y"].values
            Xv, yv = val[feature_cols].values, val["y"].values
            aupr, auc = eval_catboost(Xt, yt, Xv, yv, params)
            auprs.append(aupr)
            aucs.append(auc)
        return np.mean(auprs), np.std(auprs), np.mean(aucs)

    # Round 1: Fix depth=3, lr=0.1, search l2_leaf_reg × random_strength
    print("\n  Round 1: Tuning l2_leaf_reg × random_strength (depth=3 fixed)")
    best_score, best_params = -1, {}
    for l2 in [1.0, 3.0, 5.0, 7.0, 10.0]:
        for rs in [0.5, 2.0, 4.0, 6.0]:
            p = {"depth": 3, "learning_rate": 0.1, "l2_leaf_reg": l2,
                 "random_strength": rs, "subsample": 0.8, "colsample_bylevel": 0.8,
                 "min_data_in_leaf": 30}
            aupr, std, auc = cv_eval(p)
            score = aupr - 0.3 * std
            print(f"    l2={l2:.1f}  rs={rs:.1f}  →  AUPR={aupr:.4f}±{std:.4f}  AUC={auc:.4f}  score={score:.4f}")
            if score > best_score:
                best_score, best_params = score, p
    print(f"  Round 1 best: score={best_score:.4f}  l2={best_params.get('l2_leaf_reg'):.1f}  rs={best_params.get('random_strength'):.1f}")

    # Round 2: Vary depth and min_data_in_leaf around best from R1
    print("\n  Round 2: Tuning depth × min_data_in_leaf")
    r1_l2 = best_params["l2_leaf_reg"]
    r1_rs = best_params["random_strength"]
    for depth in [2, 3, 4]:
        for mdal in [10, 30, 60, 100]:
            p = {"depth": depth, "learning_rate": 0.1, "l2_leaf_reg": r1_l2,
                 "random_strength": r1_rs, "subsample": 0.8, "colsample_bylevel": 0.8,
                 "min_data_in_leaf": mdal}
            aupr, std, auc = cv_eval(p)
            score = aupr - 0.3 * std
            print(f"    depth={depth}  min_leaf={mdal:3d}  →  AUPR={aupr:.4f}±{std:.4f}  score={score:.4f}")
            if score > best_score:
                best_score, best_params = score, p
    print(f"  Round 2 best: score={best_score:.4f}  depth={best_params['depth']}  min_leaf={best_params['min_data_in_leaf']}")

    # Round 3: Fine-tune learning_rate and subsample
    print("\n  Round 3: Fine-tuning learning_rate × subsample")
    r3_depth = best_params["depth"]
    r3_l2 = best_params["l2_leaf_reg"]
    r3_rs = best_params["random_strength"]
    r3_mdal = best_params["min_data_in_leaf"]
    for lr in [0.03, 0.05, 0.08, 0.12, 0.15]:
        for ss in [0.7, 0.85, 1.0]:
            p = {"depth": r3_depth, "learning_rate": lr, "l2_leaf_reg": r3_l2,
                 "random_strength": r3_rs, "subsample": ss, "colsample_bylevel": 0.8,
                 "min_data_in_leaf": r3_mdal}
            aupr, std, auc = cv_eval(p)
            score = aupr - 0.3 * std
            print(f"    lr={lr:.3f}  ss={ss:.2f}  →  AUPR={aupr:.4f}±{std:.4f}  score={score:.4f}")
            if score > best_score:
                best_score, best_params = score, p

    # Final evaluation
    final_aupr, final_std, final_auc = cv_eval(best_params)
    print(f"\n  CatBoost Best: AUPR={final_aupr:.4f}±{final_std:.4f}  AUC={final_auc:.4f}")
    print(f"  Best params: {best_params}")
    return best_params, final_aupr, final_std


# ---- Phase 3: XGBoost quick validation ----
def xgboost_quick_check(full_df, feature_cols, splits):
    """Validate XGBoost baseline on reduced features, quick lr/depth sweep."""
    print(f"\n{'='*60}")
    print("Phase 3: XGBoost Validation (baseline + quick sweep)")
    print("=" * 60)

    def cv_eval_xgb(params):
        auprs = []
        for sp in splits:
            train = full_df.iloc[:sp]
            val = full_df.iloc[sp:]
            Xt, yt = train[feature_cols].values, train["y"].values
            Xv, yv = val[feature_cols].values, val["y"].values
            spw = (yt == 0).sum() / max(yt.sum(), 1)
            p = params.copy()
            p.update({"objective": "binary:logistic", "scale_pos_weight": spw,
                       "seed": SEED, "verbosity": 0, "eval_metric": "aucpr"})
            dtrain = xgb.DMatrix(Xt, label=yt)
            dval = xgb.DMatrix(Xv, label=yv)
            model = xgb.train(p, dtrain, num_boost_round=1000, evals=[(dval, "val")],
                              early_stopping_rounds=50, verbose_eval=False)
            probs = model.predict(dval)
            auprs.append(average_precision_score(yv, probs))
        return np.mean(auprs), np.std(auprs)

    # Baseline
    base = {"max_depth": 3, "learning_rate": 0.1, "gamma": 0.5, "lambda": 0.5,
            "alpha": 0.0, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1}
    bl_aupr, bl_std = cv_eval_xgb(base)
    print(f"  Baseline:         AUPR={bl_aupr:.4f}±{bl_std:.4f}")

    # Quick sweep
    best_score, best_params = bl_aupr - 0.2 * bl_std, base.copy()
    for lr in [0.05, 0.08, 0.12, 0.15]:
        for depth in [2, 3, 4]:
            for gamma in [0.1, 0.5, 1.0]:
                p = base.copy()
                p.update({"learning_rate": lr, "max_depth": depth, "gamma": gamma})
                aupr, std = cv_eval_xgb(p)
                score = aupr - 0.2 * std
                if score > best_score:
                    best_score, best_params = score, p.copy()

    best_aupr, best_std = cv_eval_xgb(best_params)
    print(f"  Best (sweep):     AUPR={best_aupr:.4f}±{best_std:.4f}  Δ={best_aupr-bl_aupr:.4f}")
    print(f"  Best params: {best_params}")

    # Use baseline if sweep didn't meaningfully improve
    if best_aupr - bl_aupr < 0.002:
        print("  → Sweep improvement negligible, keeping baseline params")
        best_params = base
        best_aupr, best_std = bl_aupr, bl_std

    return best_params, best_aupr, best_std


# ---- Phase 4: Final models + predictions ----
def train_final_and_predict(xgb_params, cb_params, feature_cols):
    print(f"\n{'='*60}")
    print("Phase 4: Final Models + Predictions")
    print("=" * 60)

    df = pd.read_csv(FULL_DATA_PATH)
    X = engineer_features(df.drop(columns=["y"]))
    y = df["y"].values
    X_sel = X[feature_cols]
    full_spw = (y == 0).sum() / y.sum()

    # --- XGBoost final ---
    print("\n  Training final XGBoost...")
    val_cut = int(len(y) * 0.95)
    xgb_p = xgb_params.copy()
    xgb_p.update({"objective": "binary:logistic", "scale_pos_weight": (y[:val_cut]==0).sum()/y[:val_cut].sum(),
                   "seed": SEED, "verbosity": 0, "eval_metric": "aucpr"})
    dt_cv = xgb.DMatrix(X_sel.values[:val_cut], label=y[:val_cut])
    dv_cv = xgb.DMatrix(X_sel.values[val_cut:], label=y[val_cut:])
    xgb_cv = xgb.train(xgb_p, dt_cv, num_boost_round=1000,
                        evals=[(dv_cv, "val")], early_stopping_rounds=50, verbose_eval=False)
    best_iter_xgb = xgb_cv.best_iteration

    xgb_final = xgb.train({**xgb_p, "scale_pos_weight": full_spw}, xgb.DMatrix(X_sel.values, label=y),
                           num_boost_round=best_iter_xgb, verbose_eval=100)

    # XGBoost threshold
    from sklearn.metrics import precision_recall_curve as prc
    xgb_probs = xgb_cv.predict(dv_cv)
    p, r, t = prc(y[val_cut:], xgb_probs)
    f1s = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-10)
    xgb_thresh = t[np.argmax(f1s)]
    print(f"    Best iter: {best_iter_xgb}  |  Threshold: {xgb_thresh:.4f}")
    print(f"    Val AUPR: {average_precision_score(y[val_cut:], xgb_probs):.4f}")

    xgb_final.save_model("xgboost/model/xgb_model_optimized.json")
    joblib.dump({"threshold": float(xgb_thresh), "feature_cols": feature_cols,
                 "best_iteration": best_iter_xgb, "best_params": xgb_params},
                "xgboost/model/metadata_optimized.pkl")

    # --- CatBoost final ---
    print("\n  Training final CatBoost...")
    X_cv_tr, X_cv_v = X_sel.values[:val_cut], X_sel.values[val_cut:]
    y_cv_tr, y_cv_v = y[:val_cut], y[val_cut:]
    cb_p = cb_params.copy()
    cb_p.update({"scale_pos_weight": (y_cv_tr==0).sum()/y_cv_tr.sum(), "random_seed": SEED})
    cb_cv = CatBoostClassifier(**cb_p, iterations=800, eval_metric="PRAUC",
                                early_stopping_rounds=50, use_best_model=True,
                                verbose=False, thread_count=-1, train_dir="catboost/catboost_info")
    cb_cv.fit(Pool(X_cv_tr, y_cv_tr), eval_set=Pool(X_cv_v, y_cv_v), plot=False)
    best_iter_cb = cb_cv.get_best_iteration()

    cb_final = CatBoostClassifier(**{**cb_params, "scale_pos_weight": full_spw, "random_seed": SEED},
                                   iterations=best_iter_cb, verbose=100, thread_count=-1,
                                   train_dir="catboost/catboost_info")
    cb_final.fit(Pool(X_sel.values, y), plot=False)

    # CatBoost threshold
    cb_probs = cb_cv.predict_proba(X_cv_v)[:, 1]
    p, r, t = prc(y_cv_v, cb_probs)
    f1s = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-10)
    cb_thresh = t[np.argmax(f1s)]
    print(f"    Best iter: {best_iter_cb}  |  Threshold: {cb_thresh:.4f}")
    print(f"    Val AUPR: {average_precision_score(y_cv_v, cb_probs):.4f}")

    cb_final.save_model("catboost/model/catboost_model_optimized.cbm")
    joblib.dump({"threshold": float(cb_thresh), "feature_cols": feature_cols,
                 "best_iteration": best_iter_cb, "best_params": cb_params},
                "catboost/model/metadata_optimized.pkl")

    # --- Predictions ---
    print(f"\n  Generating Test Predictions...")
    for test_file, task_name in [("test_simple.csv", "Task 1"), ("test_complex.csv", "Task 2")]:
        test_df = pd.read_csv(os.path.join(DATA_DIR, test_file))
        X_test = engineer_features(test_df)
        X_test = X_test.reindex(columns=feature_cols, fill_value=0.0)

        # XGBoost
        xgb_preds = (xgb_final.predict(xgb.DMatrix(X_test.values)) >= xgb_thresh).astype(int)
        pd.DataFrame({"y_pred": xgb_preds}).to_csv(
            f"xgboost/predictions/pred_{test_file.replace('.csv','')}_optimized.csv", index=False)

        # CatBoost
        cb_preds = (cb_final.predict_proba(X_test.values)[:, 1] >= cb_thresh).astype(int)
        pd.DataFrame({"y_pred": cb_preds}).to_csv(
            f"catboost/predictions/pred_{test_file.replace('.csv','')}_optimized.csv", index=False)

        print(f"    {task_name} ({test_file}): XGB={xgb_preds.sum()}/{len(xgb_preds)}  CB={cb_preds.sum()}/{len(cb_preds)}")

    return xgb_thresh, cb_thresh


# ---- Main ----
def main():
    t0 = time.time()
    print("=" * 60)
    print("Efficient Knowledge-Driven Optimization")
    print("=" * 60)
    print("Feature selection → CatBoost targeted grid → XGBoost quick sweep → Final models")

    # Phase 0
    print(f"\n--- Phase 0: Load & Engineer Features ---")
    X, y, all_cols = load_full()
    full_df = pd.concat([X, pd.Series(y, name="y")], axis=1)
    splits = temporal_cv_splits()
    print(f"  CV splits: {splits}")

    # Phase 1: Feature selection
    top_features, importance = select_features(X, y, top_n=300)
    print(f"  Selected {len(top_features)} features")

    # Phase 2: CatBoost targeted grid search
    cb_params, cb_aupr, cb_std = catboost_grid_search(full_df, top_features, splits)

    # Phase 3: XGBoost quick validation
    xgb_params, xgb_aupr, xgb_std = xgboost_quick_check(full_df, top_features, splits)

    # Phase 4: Final models + predictions
    train_final_and_predict(xgb_params, cb_params, top_features)

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Total time: {elapsed/60:.1f} min")
    print(f"  Features used: {len(top_features)}")
    print(f"\n  XGBoost optimized:")
    print(f"    AUPR={xgb_aupr:.4f}±{xgb_std:.4f}")
    print(f"    Params: {xgb_params}")
    print(f"\n  CatBoost optimized:")
    print(f"    AUPR={cb_aupr:.4f}±{cb_std:.4f}")
    print(f"    Params: {cb_params}")
    print("\nDone!")


if __name__ == "__main__":
    main()
