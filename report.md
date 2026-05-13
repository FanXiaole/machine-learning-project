# Robust Anomaly Detection in Noisy Time-Series Data — Comprehensive Report

## 1. Introduction

This project addresses anomaly detection in noisy time-series data, motivated by applications such as financial market monitoring. The dataset consists of 137,192 sequential observations with 33 numerical features, where anomalies are rare (0.42% positive rate) and occur exclusively in contiguous streaks of 30 consecutive time steps (19 streaks total, all concentrated in the last ~10% of data at indices 124,283–136,582).

Two models are developed and compared: **XGBoost** (Chen & Guestrin, 2016) and **CatBoost** (Prokhorenkova et al., 2018). Both are gradient-boosted tree ensembles, but they differ fundamentally in tree structure and gradient estimation, offering complementary perspectives on the problem.

### 1.1 Tasks

| Task | Dataset | Description |
|------|---------|-------------|
| Task 1 | `test_simple.csv` (25,647 rows) | Anomaly detection under the training distribution |
| Task 2 | `test_complex.csv` (34,542 rows) | Robust detection under a more complex, unseen distribution |

**Critical constraint**: The model used for Task 2 must be identical to the one used for Task 1 — no retraining, fine-tuning, or adaptation is allowed.

---

## 2. Feature Engineering

The raw features (`f1`–`f33`) capture instantaneous values but contain limited temporal context. Since anomalies manifest as 30-step streaks with sharp boundary transitions, sliding-window temporal features are constructed to capture local dynamics.

An extensive search over window configurations revealed that **ultra-short windows (2–8 steps) dramatically outperform longer windows** (5–100). The reason: anomaly streaks have sharp boundaries where variance changes abruptly; rolling statistics over short windows capture these transitions, while longer windows smooth out the signal.

| Feature Type | Configuration | Features Generated |
|-------------|---------------|-------------------|
| Rolling statistics (mean, std, max, min) | Windows [2, 3, 5, 8] × 33 features | 4×4×33 = 528 |
| Lag features | Steps [1, 2] × 33 features | 2×33 = 66 |
| Difference features | First-order × 33 features | 1×33 = 33 |
| Z-score (relative to rolling window) | Windows [5, 10, 15] × 33 features | 3×33 = 99 |
| Original features | `f1`–`f33` | 33 |
| **Total** | | **759** |

Feature pruning via XGBoost gain-based importance reduces this to **297 features** (~39% of total) while retaining ~100% of the gain. The top features are dominated by rolling standard deviation, confirming that changes in local variance are the primary anomaly signal.

---

## 3. XGBoost

### 3.1 Model Design

XGBoost uses asymmetric (non-oblivious) trees where each leaf can split on a different feature. This flexibility is well-suited to the feature set, where many rolling statistics are correlated — the model can route different subsets of samples to different feature splits.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_depth` | 3 | Shallow trees prevent overfitting; deeper trees (≥6) showed severe degradation |
| `learning_rate` | 0.15 | Higher rate with shallow trees enables fast convergence |
| `gamma` | 0.1 | Low minimum loss reduction — allows more splits given shallow depth |
| `lambda` (L2) | 0.5 | Light L2 regularization on leaf weights |
| `subsample` | 0.8 | Row subsampling for variance reduction |
| `colsample_bytree` | 0.8 | Column subsampling per tree |
| `scale_pos_weight` | ~240 | Counteracts 0.42% positive rate |

### 3.2 Validation Strategy

A single temporal holdout split at index 131,000 (within the anomaly region) is used for model selection, ensuring both training and validation sets contain anomaly streaks while strictly preserving temporal order. Early stopping with `patience=100` rounds is applied on validation AUC-PR. The final model is retrained on all 137,192 labeled samples.

### 3.3 Results

| Metric | Value |
|--------|-------|
| Temporal CV AUPR (3-fold) | 0.9903 ± 0.0134 |
| Val AUPR (holdout) | 0.9995 |
| Best iteration | 184 |
| Optimal threshold | 0.0082 |

---

## 4. CatBoost

### 4.1 Algorithm Design

CatBoost introduces two key innovations not present in XGBoost:

**Ordered Boosting**: Traditional gradient boosting computes residuals using a model that has seen all training labels, introducing prediction shift bias. CatBoost trains per-sample "historical" models using only preceding observations in a random permutation, producing unbiased gradient estimates. This is theoretically beneficial for extreme class imbalance where the minority gradient signal is already weak.

**Symmetric (Oblivious) Trees**: All nodes at the same depth use identical split conditions. This structural constraint acts as a regularizer, producing smoother probability estimates and reducing overfitting. However, symmetric trees are more sensitive to feature correlations — when many features carry redundant information, forcing a single split per level wastes model capacity.

### 4.2 Hyperparameter Optimization

CatBoost's initial baseline (default parameters) achieved AUPR=0.9373 with alarming instability (std=0.0887 across temporal splits). A three-round targeted grid search was conducted using 3-fold temporal cross-validation to address this.

**Optimization Objective**: `AUPR − 0.3 × std(AUPR)` — heavily penalizes unstable models.

**Round 1 — Regularization (`l2_leaf_reg` × `random_strength`, 20 combinations)**:

| l2_leaf_reg | random_strength | AUPR | Std | Score |
|-------------|-----------------|------|-----|-------|
| 1.0 | 4.0 | 0.9355 | ±0.0835 | 0.9105 |
| 3.0 | 4.0 | 0.9810 | ±0.0265 | 0.9730 |
| 7.0 | 2.0 | 0.9786 | ±0.0302 | 0.9695 |
| **10.0** | **2.0** | **0.9831** | **±0.0237** | **0.9760** |

Heavy L2 regularization (`l2_leaf_reg=10.0`) combined with moderate random strength (`2.0`) provides the best accuracy-stability balance. Low L2 values lead to catastrophic variance.

**Round 2 — Tree Structure (`depth` × `min_data_in_leaf`, 12 combinations)**:

| depth | AUPR | Std |
|-------|------|-----|
| 2 | 0.9676 | ±0.0457 |
| **3** | **0.9831** | **±0.0237** |
| 4 | 0.9774 | ±0.0318 |

`depth=3` is optimal. Depth 2 underfits; depth 4 overfits. Notably, `min_data_in_leaf` has virtually no effect (identical scores for values 10–100), confirming symmetric trees provide sufficient inherent leaf regularization.

**Round 3 — Learning Rate and Sampling (15 combinations)**:

| learning_rate | subsample | AUPR | Std |
|---------------|-----------|------|-----|
| 0.030 | 0.70 | 0.8742 | ±0.0752 |
| **0.030** | **1.00** | **0.9831** | **±0.0220** |
| 0.080 | 0.85 | 0.9814 | ±0.0259 |
| 0.150 | 0.85 | 0.9799 | ±0.0274 |

Lower learning rates with full data per tree yield the most stable models. Critically, low learning rate combined with subsampling (`lr=0.03, subsample=0.7`) is the worst configuration — when trees are already weak, reducing training data amplifies rare-class instability.

**Final Optimized Parameters**:

| Parameter | Baseline | Optimized |
|-----------|----------|-----------|
| `depth` | 3 | 3 |
| `learning_rate` | 0.1 | **0.03** |
| `l2_leaf_reg` | 0.5 | **10.0** |
| `random_strength` | — | **2.0** |
| `subsample` | 0.8 | **1.0** |
| `colsample_bylevel` | 0.8 | 0.8 |
| `min_data_in_leaf` | — | 30 |

### 4.3 Optimization Results

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| AUPR | 0.9373 | **0.9831** | +4.9% |
| AUC | 0.9878 | **0.9972** | +1.0% |
| Std(AUPR) | 0.0887 | **0.0220** | **−75.2%** |

| Final Model Property | Value |
|----------------------|-------|
| Best iteration | 571 |
| Optimal threshold | 0.3332 |
| Val AUPR (holdout) | 0.9959 |

---

## 5. Adversarial Feature Pruning

### 5.1 Motivation

The primary risk for Task 2 is distribution shift: the model may rely on features that are predictive of anomalies in the training distribution but behave differently in the complex test distribution. To investigate this, we conduct **adversarial validation**: training a classifier to distinguish training samples from test samples. A high adversarial AUC indicates significant distribution shift.

### 5.2 Adversarial Validation Results

| Comparison | Adversarial CV AUC | Interpretation |
|------------|-------------------|----------------|
| Train vs Task 1 (simple) | **0.85** | Significant shift — joint distribution differs, though marginal feature means/stds are close |
| Train vs Task 2 (complex) | **1.00** | Complete separability — the complex test set is trivially distinguishable from training data |

For Task 2, a single feature (`f26`) achieves gain=1166 in discriminating train from test, indicating its behavior is completely different in the two distributions.

### 5.3 Feature Pruning Approach

The strategy: for each feature, compute both its **anomaly detection importance** (how well it helps detect anomalies) and its **adversarial importance** (how well it distinguishes train vs test). Remove features where the ratio of adversarial-to-anomaly importance is high — i.e., features that are more useful for domain discrimination than anomaly detection.

A robustness score is defined as:
```
robustness_score = anomaly_norm − 0.5 × (adversary_norm / anomaly_norm)
```

Features are ranked by this score and the top 250 (out of 253 gain-positive features) are retained.

### 5.4 Results

| Model | Original (253 features) | Pruned (250 features) | Δ |
|-------|------------------------|----------------------|---|
| XGBoost | 0.9898 ± 0.0133 | 0.9866 ± 0.0186 | −0.0032 |
| CatBoost | 0.9798 ± 0.0281 | 0.9792 ± 0.0295 | −0.0007 |

Only 3 features were removed, with negligible impact on CV performance. Test predictions shifted modestly:

| Dataset | Model | Original Predictions | Pruned Predictions |
|---------|-------|---------------------|--------------------|
| Task 1 | XGB | 931 (3.63%) | 923 (3.60%) |
| Task 1 | CB | 911 (3.55%) | 929 (3.62%) |
| Task 2 | XGB | 806 (2.33%) | 724 (2.10%) |
| Task 2 | CB | 597 (1.73%) | 658 (1.90%) |

### 5.5 Reflection: Why Adversarial Pruning Failed

The adversarial pruning approach yielded no meaningful improvement. Several factors explain this:

1. **Global distribution shift, not sparse spurious features.** The adversary achieves AUC=1.0 because the *entire joint distribution* differs between train and test, not because a few "bad" features cause the shift. The adversarial signal is dispersed across hundreds of features — no single feature has anomalously high adversarial importance that can be pruned in isolation.

2. **Coupled importance.** The features that matter for anomaly detection (rolling standard deviations, z-scores) are largely the same features that carry domain information. Removing any feature tends to hurt anomaly detection slightly, without meaningfully reducing domain sensitivity.

3. **The shift is fundamental.** Task 2 is designed to be a "more complex scenario" where the underlying data characteristics differ. Adversarial pruning assumes the shift can be mitigated by removing spurious features, but when the shift is baked into the data generation process itself, no amount of feature selection can close the gap.

4. **Unknown true labels.** Without access to Task 2's true labels, we cannot verify whether changes in predictions represent improved generalization or degraded performance. The 82-observation reduction in XGBoost's Task 2 predictions (806→724) could equally represent fewer false positives (better) or fewer true positives (worse).

**Key takeaway**: Adversarial validation is a powerful diagnostic tool for detecting distribution shift *before* it causes problems, but adversarial feature pruning only helps when the shift is caused by a small set of identifiable spurious features. When the shift is systemic, the only effective remedies are domain adaptation techniques (which are prohibited here) or more fundamentally distribution-invariant feature engineering. Candidates include:
- Robust scaling (e.g., per-feature standardization)
- Ratio-based features that are scale-invariant
- Exponentially weighted moving averages that adapt to changing distributions
- Frequency-domain features less sensitive to mean/variance shifts

---

## 6. Final Model Comparison

### 6.1 Cross-Validation Performance (3-fold Temporal)

| Model | AUPR | Std | AUC |
|-------|------|-----|-----|
| **XGBoost** | **0.9903** | ±0.0134 | — |
| CatBoost | 0.9831 | ±0.0220 | 0.9972 |

### 6.2 Holdout Validation

| Model | Val AUPR | Best Iteration | Threshold |
|-------|----------|---------------|-----------|
| **XGBoost** | **0.9995** | 184 | 0.0082 |
| CatBoost | 0.9959 | 571 | 0.3332 |

### 6.3 Test Predictions

| Dataset | XGBoost | CatBoost |
|---------|---------|----------|
| Task 1 (simple, 25,647 rows) | 931 (3.63%) | 911 (3.55%) |
| Task 2 (complex, 34,542 rows) | 806 (2.33%) | 597 (1.73%) |

### 6.4 Structural Comparison

| Aspect | XGBoost | CatBoost |
|--------|---------|----------|
| Tree structure | Asymmetric | Symmetric (oblivious) |
| Gradient estimation | Standard | Ordered (unbiased) |
| Optimal depth | 3 | 3 |
| Optimal L2 regularization | 0.5 | 10.0 |
| CV stability | Excellent (±0.013) | Good (±0.022) |
| Training speed | Faster (~3 min) | Slower (~8 min) |
| Feature sensitivity | Lower | Higher (needs pruning) |
| Probability calibration | Extreme (threshold=0.008) | Conservative (threshold=0.333) |
| Task 2 predictions | 2.33% anomaly rate | 1.73% anomaly rate |

### 6.5 Which Model to Submit?

**XGBoost** is the recommended primary submission: higher AUPR, better CV stability, and a more moderate anomaly rate on Task 2 that avoids excessive false negatives. CatBoost's conservative threshold (0.333 vs 0.008) and lower Task 2 anomaly rate (1.73%) suggest it may miss anomalies under distribution shift.

Both models detect all 570 training anomalies (perfect recall). The choice between them hinges on Task 2 generalization, which cannot be evaluated without hidden labels.

---

## 7. Discussion

### 7.1 Why XGBoost Outperforms CatBoost on This Dataset

XGBoost's asymmetric trees are better suited to this feature set. The 297 engineered features contain many correlated rolling statistics (e.g., `f1_rm2` and `f1_rm3` differ only by window size). Asymmetric trees can route different samples to different features at each leaf, naturally handling redundancy. CatBoost's symmetric trees force all nodes at a level to use the same split, wasting capacity on correlated features.

CatBoost's ordered boosting, while theoretically appealing, provides marginal benefit when 137K samples are available and the anomaly signal (sharp variance changes) is strong enough to be captured by standard gradient estimation.

### 7.2 Task 2 Generalization

The adversarial validation (AUC=1.0 for train vs Task 2) reveals a fundamental distribution gap. Without the ability to retrain or adapt, model generalization depends entirely on whether the engineered features capture distribution-invariant anomaly patterns. The rolling standard deviation features that dominate both models are likely more robust than raw features, but their reliance on fixed window sizes is an implicit assumption about the temporal scale of anomalies.

### 7.3 Limitations

1. **No Task 2 labels**: All generalization analysis is necessarily indirect. The true Task 2 performance remains unknown.
2. **Fixed window assumption**: Rolling windows of 2–8 steps implicitly assume anomaly signatures operate on these time scales. A scenario with fundamentally different temporal dynamics would degrade performance.
3. **Single model family**: Both models are tree-based ensembles. Neural approaches (e.g., temporal convolutional networks) might capture different aspects of the temporal structure.
4. **Adversarial pruning limitation**: As demonstrated, when distribution shift is global rather than sparse, feature pruning cannot meaningfully improve robustness.

---

## 8. Division of Work

Solo project — all work (data analysis, feature engineering, XGBoost implementation, CatBoost implementation, hyperparameter optimization, adversarial validation, report writing) was completed by the team member alone.

---

## References

- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD*.
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems (NeurIPS)*.
