# CatBoost for Robust Anomaly Detection in Noisy Time-Series Data

## 1. Introduction

This report documents the application of CatBoost (Categorical Boosting) to the anomaly detection task, performed in parallel with the XGBoost baseline. CatBoost introduces two key algorithmic innovations — **ordered boosting** and **symmetric (oblivious) trees** — that differentiate it from traditional gradient boosting frameworks. The goals are to (1) evaluate whether CatBoost can match XGBoost's detection performance, and (2) systematically optimize its hyperparameters for this highly imbalanced time-series setting.

## 2. Method

### 2.1 CatBoost Algorithm Overview

CatBoost differs from XGBoost in two fundamental ways:

**Ordered Boosting.** Traditional gradient boosting computes residuals using a model that has already seen all training labels, introducing a subtle overfitting bias known as *prediction shift*. CatBoost mitigates this by training a separate model for each data point using only the observations that precede it in a random permutation. The residuals used for fitting each new tree are thus unbiased estimates of the true gradient. In highly imbalanced settings (0.42% positive rate), this is particularly important: the gradient signal for the minority class is already weak, and biased gradient estimates can further degrade rare-class detection.

**Symmetric (Oblivious) Trees.** Unlike XGBoost's asymmetric trees where each leaf can split on a different feature, CatBoost enforces that all nodes at the same depth use the same split condition. This constraint acts as a structural regularizer: it reduces model capacity per tree, making the model less prone to overfitting on noise, and produces smoother probability estimates. The trade-off is that symmetric trees may require more iterations to match the expressive power of asymmetric trees, and they can struggle when features are highly correlated (since one split must serve all nodes at a level).

### 2.2 Feature Engineering

The same feature engineering pipeline from the XGBoost baseline is used: 759 temporal features derived from the 33 raw features via rolling statistics (windows 2, 3, 5, 8), lag features (steps 1, 2), first-order differencing, and z-score normalization (windows 5, 10, 15). 

A key finding during optimization is that CatBoost's symmetric tree structure benefits significantly from **feature pruning**. Using XGBoost gain-based importance, we select the top 297 features (which capture ~100% of total gain). This reduces feature correlation noise that symmetric trees are particularly sensitive to, and speeds up training by ~2.5×.

Top 5 features by gain importance:
| Rank | Feature | Type | Gain |
|------|---------|------|------|
| 1 | `f32_rs5` | Rolling std (window=5) | 30,980 |
| 2 | `f32_rmax8` | Rolling max (window=8) | 12,849 |
| 3 | `f24_rmin2` | Rolling min (window=2) | 9,656 |
| 4 | `f33_rs2` | Rolling std (window=2) | 9,240 |
| 5 | `f6_z10` | Z-score (window=10) | 9,204 |

### 2.3 Hyperparameter Optimization Strategy

A knowledge-driven three-round targeted grid search is conducted using 3-fold temporal cross-validation (split points at indices 127,283, 131,432, and 135,582, all within the anomaly region). The optimization objective is `AUPR - 0.3 × std(AUPR)`, which penalizes unstable models and favors consistent performance across temporal splits.

**Round 1: l2_leaf_reg × random_strength (depth=3 fixed, 20 combinations).** These two parameters are CatBoost's primary regularization mechanisms. L2 regularization directly penalizes large leaf weights, while `random_strength` adds Gaussian noise to the split scores, acting as an implicit regularizer.

**Round 2: depth × min_data_in_leaf (12 combinations).** Tree depth controls model capacity; `min_data_in_leaf` prevents splits on too few samples. With symmetric trees, depth has a more predictable effect on capacity than in asymmetric models.

**Round 3: learning_rate × subsample (15 combinations).** Learning rate controls the contribution of each tree; subsample (row sampling) adds stochasticity. Lower learning rates with more iterations typically produce smoother decision boundaries.

### 2.4 Handling Class Imbalance

The primary mechanism is `scale_pos_weight = N_neg / N_pos ≈ 435` (varies per fold depending on the exact train split). This adjusts the loss function such that misclassifying a positive sample incurs a penalty proportional to the class imbalance ratio. CatBoost applies this weight consistently across its ordered boosting procedure.

AUC-PR (average precision) is used as the evaluation metric for early stopping and threshold selection, as it is more informative than AUC-ROC for highly imbalanced data (0.42% positive rate).

## 3. Validation Strategy

### 3.1 Time-Aware Cross-Validation

Standard K-fold cross-validation would leak temporal information. Instead, three temporal holdout splits are used, all within the anomaly region (anomalies span indices 124,283–136,582, organized as 19 streaks of 30 consecutive time steps):

| Split | Training Range | Validation Range | Train Anomalies | Val Anomalies |
|-------|---------------|------------------|-----------------|---------------|
| 1 | 0–127,282 | 127,283–137,191 | ~180 | ~390 |
| 2 | 0–131,431 | 131,432–137,191 | ~300 | ~270 |
| 3 | 0–135,581 | 135,582–137,191 | ~420 | ~150 |

Each split preserves strict temporal order and ensures both training and validation sets contain anomaly segments.

### 3.2 Model Selection

Early stopping with `patience=50` rounds is applied on validation AUC-PR. The best model configuration is selected based on the composite score `AUPR - 0.3 × std(AUPR)` across all three splits. The final model is then retrained on all 137,192 labeled samples using the selected hyperparameters.

### 3.3 Metrics

AUC-PR is the primary validation metric. AUC-ROC is also reported for completeness. The prediction threshold is selected to maximize F1 score on a held-out validation portion (last 5% of training data).

## 4. Optimization Results

### 4.1 Round 1: Regularization Tuning

| l2_leaf_reg | random_strength | AUPR | Std | Score |
|-------------|-----------------|------|-----|-------|
| 1.0 | 0.5 | 0.9725 | ±0.0389 | 0.9608 |
| 1.0 | 2.0 | 0.9738 | ±0.0358 | 0.9630 |
| 1.0 | 4.0 | 0.9355 | ±0.0835 | 0.9105 |
| 3.0 | 4.0 | 0.9810 | ±0.0265 | 0.9730 |
| 5.0 | 2.0 | 0.9778 | ±0.0313 | 0.9684 |
| 7.0 | 4.0 | 0.9807 | ±0.0271 | 0.9725 |
| **10.0** | **2.0** | **0.9831** | **±0.0237** | **0.9760** |

**Findings:** Higher L2 regularization consistently reduces variance. The combination of `l2_leaf_reg=10.0` with moderate `random_strength=2.0` achieves the best balance between accuracy and stability. Low L2 values combined with high random_strength lead to degraded performance (AUPR drops to ~0.93).

### 4.2 Round 2: Tree Structure Tuning

| depth | min_data_in_leaf | AUPR | Std | Score |
|-------|-----------------|------|-----|-------|
| 2 | 10–100 | 0.9676 | ±0.0457 | 0.9539 |
| **3** | **10–100** | **0.9831** | **±0.0237** | **0.9760** |
| 4 | 10–100 | 0.9774 | ±0.0318 | 0.9678 |

**Findings:** `depth=3` is the clear optimal. Depth 2 underfits (symmetric trees at depth 2 have very limited capacity). Depth 4 shows signs of overfitting (higher variance, lower mean). Notably, `min_data_in_leaf` has virtually no effect (identical scores for values 10–100), confirming that the symmetric tree structure already provides sufficient leaf-level regularization.

### 4.3 Round 3: Learning Rate and Sampling

| learning_rate | subsample | AUPR | Std | Score |
|---------------|-----------|------|-----|-------|
| 0.030 | 0.70 | 0.8742 | ±0.0752 | 0.8517 |
| **0.030** | **1.00** | **0.9831** | **±0.0220** | **0.9765** |
| 0.050 | 1.00 | 0.9771 | ±0.0317 | 0.9676 |
| 0.080 | 0.85 | 0.9814 | ±0.0259 | 0.9737 |
| 0.120 | 0.85 | 0.9778 | ±0.0309 | 0.9685 |
| 0.150 | 0.85 | 0.9799 | ±0.0274 | 0.9716 |

**Findings:** Lower learning rates with no subsampling produce the most stable models. The combination `lr=0.03, subsample=1.0` achieves the lowest variance. Importantly, low learning rate combined with row subsampling (`lr=0.03, subsample=0.7`) is the worst configuration — when trees are already weak (low learning rate), reducing training data further amplifies instability on the rare class.

### 4.4 Final Optimized Parameters

| Parameter | Baseline | Optimized | Rationale |
|-----------|----------|-----------|-----------|
| `depth` | 3 | **3** | Optimal for symmetric tree capacity |
| `learning_rate` | 0.1 | **0.03** | Lower rate → smoother convergence |
| `l2_leaf_reg` | 0.5 | **10.0** | Heavy regularization suppresses variance |
| `random_strength` | — | **2.0** | Noise injection for additional robustness |
| `subsample` | 0.8 | **1.0** | Full data per tree at low learning rate |
| `colsample_bylevel` | 0.8 | **0.8** | Maintained from baseline |
| `min_data_in_leaf` | — | **30** | Moderate leaf size constraint |
| `scale_pos_weight` | 240 | **~435** (per-fold) | Higher weight for extreme imbalance |

### 4.5 Performance Improvement

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **AUPR** | 0.9373 | **0.9831** | +4.9% |
| **AUC** | 0.9878 | **0.9972** | +1.0% |
| **Std(AUPR)** | 0.0887 | **0.0220** | **−75.2%** |

The 75% reduction in cross-validation variance is the most significant achievement. The baseline CatBoost model was highly unstable across different temporal splits, making it unreliable for deployment. The optimized model achieves consistent performance regardless of where the temporal split falls within the anomaly region.

## 5. Discussion

### 5.1 Why CatBoost Required Heavy Regularization

CatBoost's symmetric tree structure, while providing built-in regularization, proved insufficient on its own for this dataset. The 759 engineered features contain many correlated rolling statistics (e.g., `f1_rm2` and `f1_rm3` differ only by window size). Symmetric trees, which force all nodes at a level to use the same split, are particularly vulnerable to correlated features: they cannot "route" different samples to different splits at the same depth. This explains why the baseline (`l2_leaf_reg=0.5`) showed such high variance (std=0.089) — without strong L2 regularization, the model allocated large weights to a few features, making predictions brittle to small changes in the temporal split.

The solution was twofold: (1) **feature pruning** to 297 features, removing low-gain correlated features, and (2) **heavy L2 regularization** (`l2_leaf_reg=10.0`) combined with `random_strength=2.0` to further dampen leaf weight magnitudes.

### 5.2 Comparison with XGBoost

| Aspect | CatBoost | XGBoost |
|--------|----------|---------|
| Tree structure | Symmetric (oblivious) | Asymmetric |
| Gradient estimation | Ordered (unbiased) | Standard |
| Optimal depth | 3 | 3 |
| Optimal l2 | 10.0 | 0.5 |
| Feature sensitivity | Higher (needs pruning) | Lower |
| CV stability | Good (after tuning) | Excellent |
| Training speed | Slower | Faster |

XGBoost's asymmetric trees are inherently more suited to this feature set: they can make fine-grained splits on different features for different subsets of data, naturally handling the redundancy among rolling window features. CatBoost compensates through heavier regularization and feature pruning, achieving comparable performance after optimization.

### 5.3 Strengths

1. **Ordered boosting** provides unbiased gradient estimates, which is theoretically appealing for extreme class imbalance.
2. **Highly regularizable**: the combination of `l2_leaf_reg`, `random_strength`, and symmetric tree constraints offers multiple complementary regularization paths.
3. **Smooth probability estimates**: symmetric trees produce more conservative probability outputs, which may generalize better under distribution shift (Task 2).
4. **Feature pruning synergy**: reducing from 759 to 297 features improved both speed and stability.

### 5.4 Limitations

1. **Training overhead**: ordered boosting requires maintaining per-sample models, making CatBoost 2–3× slower than XGBoost on this dataset.
2. **Feature correlation sensitivity**: symmetric trees struggle when many features carry redundant information.
3. **Hyperparameter sensitivity**: performance degrades sharply with suboptimal regularization (AUPR drops from 0.983 to 0.874 with wrong lr/subsample combination).

## 6. Division of Work

Solo project — all work (feature selection, hyperparameter optimization, model training, report writing) was completed by the team member alone.

## References

- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems (NeurIPS)*.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD*.
