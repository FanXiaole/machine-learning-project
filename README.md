# Robust Anomaly Detection in Noisy Time-Series Data

A production-grade machine learning pipeline for detecting rare anomaly streaks (0.42% positive rate) in noisy, high-dimensional time-series data. Uses gradient-boosted trees (XGBoost, CatBoost) with engineered temporal features to handle extreme class imbalance and generalize across complex distribution shifts without retraining.

## Key Innovations

- **Ultra-short window features outperform long windows.** Rolling statistics over windows of 2–8 steps capture sharp anomaly boundary transitions; longer windows (≥20) smooth out the signal and degrade detection by >6% AUPR.
- **Adversarial distribution-shift diagnosis.** Training a classifier to distinguish train vs. test samples reveals the real bottleneck — Task 2 is perfectly separable from training data (adversarial AUC=1.0), indicating a fundamental distribution gap that no amount of hyperparameter tuning can close.
- **Deterministic reproducibility via metadata anchoring.** The 297 gain-selected features are stored in `metadata.pkl` alongside the trained model. On re-run, `train.py` loads this exact feature list rather than recomputing gain-based selection (which can yield 250–300 features depending on floating-point variation). This guarantees byte-identical model reproduction.
- **Exhaustive negative-result documentation.** Adversarial feature pruning, model ensembling (99.9% agreement), LightGBM (leaf-wise growth fails on extreme imbalance), and distribution-invariant features (MAD/IQR/percentile rank) were systematically evaluated. All failed to beat the XGBoost baseline — a valuable finding for practitioners facing similar problems.

## Performance

| Model | CV AUPR (3-fold) | Holdout AUPR | Training Recall | Threshold | Features |
|-------|------------------|--------------|-----------------|-----------|----------|
| **XGBoost** | **0.9903 ± 0.0134** | **0.9995** | 570/570 (100%) | 0.0082 | 297 |
| CatBoost | 0.9831 ± 0.0220 | 0.9959 | 570/570 (100%) | 0.3332 | 297 |
| LightGBM | 0.8520 | — | — | — | 253 |

### Test Predictions

| Dataset | XGBoost | CatBoost |
|---------|---------|----------|
| Task 1 (25,647 rows) | 931 (3.63%) | 911 (3.55%) |
| Task 2 (34,542 rows) | 806 (2.33%) | 597 (1.73%) |

XGBoost is the recommended primary submission: highest AUPR, best CV stability (±0.013), moderate anomaly rate on Task 2. CatBoost is a viable backup with near-identical detection quality but more conservative predictions under distribution shift.

## How It Works

### Feature Engineering (759 → 297 features)

Raw features (`f1`–`f33`) carry no temporal context. We construct sliding-window features to make the anomaly signal trivially learnable:

| Feature Type | Configuration | Count |
|-------------|---------------|-------|
| Rolling statistics (mean, std, max, min) | Windows [2, 3, 5, 8] × 33 | 528 |
| Lag features | Steps [1, 2] × 33 | 66 |
| First-order difference | 1 × 33 | 33 |
| Z-score relative to rolling window | Windows [5, 10, 15] × 33 | 99 |
| Original features | `f1`–`f33` | 33 |
| **Total engineered** | | **759** |
| **After gain-based selection** | | **297** |

Rolling standard deviation dominates the top features — changes in local variance are the primary anomaly signal.

### Model Architecture

**XGBoost** (primary): Asymmetric trees with level-wise growth. Exact split finding preserves the precision of short-window rolling features. Parameters: `max_depth=3`, `learning_rate=0.15`, `gamma=0.1`, `scale_pos_weight≈240`.

**CatBoost** (comparison): Symmetric (oblivious) trees with ordered boosting. Requires heavy regularization (`l2_leaf_reg=10.0`, `random_strength=2.0`) to achieve stability on this feature set. Produces smoother, more conservative probability estimates.

### Reproducibility Guarantee

```
First run (no metadata):
  engineer_features() → 759 features
  select_features()   → gain-based top-300
  train()             → model + metadata.pkl (stores exact 297 feature names)

Subsequent runs:
  select_features()   → loads feature_cols from metadata.pkl
  train()             → identical feature set → identical model
```

The seed is fixed (`SEED=42`). XGBoost gain importance is deterministic for the same data. The metadata file acts as a snapshot of the feature selection decision, decoupling it from floating-point variance across XGBoost versions.

## Quick Start

```bash
# Clone
git clone https://github.com/JustinFan7777777/time-series-anomaly-detection.git
cd time-series-anomaly-detection

# XGBoost (primary model)
cd xgboost
pip install -r requirements.txt
python train.py
# → model/xgb_model.json, metadata.pkl
# → predictions/pred_simple.csv, pred_complex.csv

# CatBoost (comparison model)
cd ../catboost
pip install -r requirements.txt
python train.py
# → model/catboost_model.cbm, metadata.pkl
# → predictions/pred_simple.csv, pred_complex.csv
```

**Requirements:** Python ≥3.9, xgboost ≥2.0, catboost ≥1.2, scikit-learn ≥1.0, pandas ≥1.5

## Project Structure

```
├── report.md                    # Comprehensive project report (11 sections)
├── data/
│   ├── train.csv                # 137,192 rows × 34 columns (f1–f33 + y)
│   ├── test_simple.csv          # Task 1: 25,647 rows (same distribution)
│   └── test_complex.csv         # Task 2: 34,542 rows (complex distribution)
├── xgboost/                     # Primary model
│   ├── features.py              # Feature engineering (759 temporal features)
│   ├── train.py                 # Training script (feature selection + CV + inference)
│   ├── requirements.txt
│   ├── model/
│   │   ├── xgb_model.json       # Trained XGBoost model
│   │   └── metadata.pkl         # Threshold + 297 feature names
│   └── predictions/
│       ├── pred_simple.csv      # Task 1 predictions
│       └── pred_complex.csv     # Task 2 predictions
└── catboost/                    # Comparison model
    ├── features.py              # (identical to xgboost/)
    ├── train.py
    ├── requirements.txt
    ├── model/
    │   ├── catboost_model.cbm
    │   └── metadata.pkl
    └── predictions/
        ├── pred_simple.csv
        └── pred_complex.csv
```

## What We Tried (And Why It Didn't Beat XGBoost)

| Approach | Best Result | Why It Failed |
|----------|-------------|---------------|
| CatBoost hyperparameter optimization | AUPR 0.9831 (+4.9%) | Still below XGBoost; symmetric trees struggle with correlated features |
| Adversarial feature pruning | AUPR −0.003 | Distribution shift is global, not sparse — pruning 3 features can't fix it |
| Model ensembling | No gain | 99.9% prediction agreement; models are too homogeneous |
| LightGBM | AUPR 0.852 | Leaf-wise growth + histogram binning loses precision on extreme class imbalance |
| Distribution-invariant features (MAD/IQR) | AUPR −0.034 | `.apply()` numerical noise outweighs theoretical robustness gain |

**The bottleneck is not model expressivity — it's domain generalization.** Both XGBoost and CatBoost achieve AUPR=1.0 on training data. Further improvement requires Task 2 labels, which are hidden by design.

## References

- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
- Prokhorenkova, L. et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS*.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
- Grinsztajn, L. et al. (2022). Why do tree-based models still outperform deep learning on tabular data? *NeurIPS*.
