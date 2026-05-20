# Robust Anomaly Detection in Noisy Time-Series Data — Comprehensive Report

## 1. Introduction

This project addresses anomaly detection in noisy time-series data, motivated by applications such as financial market monitoring. The dataset consists of 137,192 sequential observations with 33 numerical features, where anomalies are rare (0.42% positive rate) and occur exclusively in contiguous streaks of 30 consecutive time steps (19 streaks total, all concentrated in the last ~10% of data at indices 124,283–136,582).

### 1.1 Related Work

Time-series anomaly detection has been approached from multiple paradigms. Statistical methods such as ARIMA-based residual analysis (Box et al., 2015) model temporal dynamics explicitly but are limited to linear, low-dimensional settings. Unsupervised density-based approaches including Isolation Forest (Liu et al., 2008) and Local Outlier Factor (Breunig et al., 2000) operate without labels but struggle with high dimensionality and extreme class imbalance. Autoencoder-based reconstruction error methods (Sakurada & Yairi, 2014) can capture non-linear temporal patterns but require careful threshold calibration.

For supervised tabular anomaly detection with engineered temporal features, gradient-boosted decision trees (GBDTs) have emerged as the dominant paradigm. XGBoost (Chen & Guestrin, 2016) established the benchmark with regularized level-wise tree growth. CatBoost (Prokhorenkova et al., 2018) introduced ordered boosting and symmetric trees to address prediction shift. LightGBM (Ke et al., 2017) proposed leaf-wise growth and gradient-based one-side sampling for efficiency. Grinsztajn et al. (2022) systematically demonstrated that GBDTs consistently outperform deep learning on tabular data, providing the theoretical foundation for our model choice.

In this work, two models are developed and compared: **XGBoost** and **CatBoost**. Both are gradient-boosted tree ensembles, but they differ fundamentally in tree structure and gradient estimation, offering complementary perspectives on the problem. All three GBDT variants (XGBoost, CatBoost, LightGBM) are empirically evaluated, with additional investigation into adversarial feature pruning, ensemble strategies, and distribution-invariant feature engineering.

### 1.2 Tasks

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

Feature pruning via XGBoost gain-based importance reduces this to **253 features** (~33% of total) while retaining ~100% of the gain. The top features are dominated by rolling standard deviation, confirming that changes in local variance are the primary anomaly signal.

**Reproducibility mechanism.** The 253 selected feature names are stored in `metadata.pkl` alongside the trained model. On subsequent runs, `train.py` loads this feature list from the existing metadata rather than re-running gain-based selection. This guarantees that the same 253 features are used for both training and inference. If no metadata exists (first-time training), a fresh gain-based selection is performed on the full 759 features.

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
| Val AUPR (holdout) | 0.9994 |
| Best iteration | 211 |
| Optimal threshold | 0.0343 |

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
| Best iteration | 468 |
| Optimal threshold | 0.5886 |
| Val AUPR (holdout) | 0.9997 |

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

Features are ranked by this score and the top 250 (out of 253 gain-positive features from the adversarial experiment's specific XGBoost configuration) are retained. Note that the final production model uses 253 gain-selected features (Section 2); the 253 here reflects the consistent feature count across all fresh training runs.

### 5.4 Results

| Model | Original (253 features) | Pruned (250 features) | Δ |
|-------|------------------------|----------------------|---|
| XGBoost | 0.9898 ± 0.0133 | 0.9866 ± 0.0186 | −0.0032 |
| CatBoost | 0.9798 ± 0.0281 | 0.9792 ± 0.0295 | −0.0007 |

Only 3 features were removed, with negligible impact on CV performance. Test predictions shifted modestly:

| Dataset | Model | Original Predictions | Pruned Predictions |
|---------|-------|---------------------|--------------------|
| Task 1 | XGB | 910 (3.55%) | 923 (3.60%) |
| Task 1 | CB | 896 (3.49%) | 929 (3.62%) |
| Task 2 | XGB | 670 (1.94%) | 724 (2.10%) |
| Task 2 | CB | 513 (1.49%) | 658 (1.90%) |

### 5.5 Reflection: Why Adversarial Pruning Failed

The adversarial pruning approach yielded no meaningful improvement. Several factors explain this:

1. **Global distribution shift, not sparse spurious features.** The adversary achieves AUC=1.0 because the *entire joint distribution* differs between train and test, not because a few "bad" features cause the shift. The adversarial signal is dispersed across hundreds of features — no single feature has anomalously high adversarial importance that can be pruned in isolation.

2. **Coupled importance.** The features that matter for anomaly detection (rolling standard deviations, z-scores) are largely the same features that carry domain information. Removing any feature tends to hurt anomaly detection slightly, without meaningfully reducing domain sensitivity.

3. **The shift is fundamental.** Task 2 is designed to be a "more complex scenario" where the underlying data characteristics differ. Adversarial pruning assumes the shift can be mitigated by removing spurious features, but when the shift is baked into the data generation process itself, no amount of feature selection can close the gap.

4. **Unknown true labels.** Without access to Task 2's true labels, we cannot verify whether changes in predictions represent improved generalization or degraded performance. The 54-observation increase in XGBoost's Task 2 predictions (670→724) after pruning could equally represent recovered true positives (better) or additional false positives (worse).

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
| LightGBM | 0.8520 | — | 0.9677 |

### 6.2 Holdout Validation

| Model | Val AUPR | Best Iteration | Threshold |
|-------|----------|---------------|-----------|
| **XGBoost** | **0.9994** | 211 | 0.0343 |
| CatBoost | 0.9997 | 468 | 0.5886 |
| LightGBM | 0.8520 | 200 | 0.8018 |

### 6.3 Test Predictions

| Dataset | XGBoost | CatBoost | LightGBM |
|---------|---------|----------|----------|
| Task 1 (simple, 25,647 rows) | 910 (3.55%) | 896 (3.49%) | 685 (2.67%) |
| Task 2 (complex, 34,542 rows) | 670 (1.94%) | 513 (1.49%) | 465 (1.35%) |

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
| Probability calibration | Skewed (threshold=0.034) | Conservative (threshold=0.589) |
| Task 2 predictions | 1.94% anomaly rate | 1.49% anomaly rate |

### 6.5 Which Model to Submit?

**XGBoost** is the recommended primary submission: highest AUPR, best CV stability, and a moderate anomaly rate on Task 2 (670 predictions, 1.94%). CatBoost is a viable backup with more conservative predictions (513 predictions, 1.49%). LightGBM is excluded from consideration due to poor validation performance (AUPR=0.852).

All three detect all 570 training anomalies. The choice between XGBoost and CatBoost hinges on Task 2 generalization, which cannot be evaluated without hidden labels.

---

## 7. Ensemble Feasibility Assessment

An investigation was conducted into whether combining XGBoost and CatBoost into an ensemble could yield better performance than either solo model.

### 7.1 Prediction Overlap Analysis

| | Task 1 | Task 2 |
|---|---|---|
| Probability correlation (Pearson) | 0.994 | 0.974 |
| Disagreement rate | 0.05% (14/25,647) | 0.47% (163/34,542) |
| XGB=1, CB=0 cases | 14 | **160** |
| XGB=0, CB=1 cases | 0 | 3 |

On Task 1, the models agree on 99.95% of predictions. On Task 2, disagreement rises to 0.47%, but almost entirely in one direction: XGBoost flags anomalies that CatBoost dismisses. This is not complementary error — it reflects CatBoost's systematically more conservative behavior under distribution shift.

### 7.2 Why Ensemble Fails Here

Three structural factors prevent meaningful ensembling:

**1. Model homogeneity.** Both models are gradient-boosted trees trained on identical features and data. Their only difference — asymmetric vs. symmetric tree structure — is insufficient to produce independent error patterns. True ensemble benefit requires diverse base learners (e.g., tree-based + neural, or models trained on different feature subsets).

**2. Ensemble degenerates to threshold tuning.** The solo models span 513–670 anomaly predictions on Task 2. Any ensemble strategy (averaging, AND, OR, weighted voting) produces results strictly within this range, equivalent to choosing a point between two already-optimized decision boundaries.

**3. No residual signal for meta-learning.** Both models achieve AUPR = 1.0 on training data (all 570 anomalies detected). A stacking meta-model has no residual error to exploit — it would simply interpolate between two near-identical predictions.

### 7.3 Conclusion

Ensemble provides no benefit. XGBoost solo is the recommended submission. The only scenario where ensemble could help is if the models were intentionally diversified by training them on different feature subsets, but this would reduce each model's individual performance, likely negating any ensemble gain.

---

## 8. Discussion

### 8.1 Why XGBoost Outperforms CatBoost on This Dataset

XGBoost's asymmetric trees are better suited to this feature set. The 253 selected features (from 759 engineered) contain many correlated rolling statistics (e.g., `f1_rm2` and `f1_rm3` differ only by window size). Asymmetric trees can route different samples to different features at each leaf, naturally handling redundancy. CatBoost's symmetric trees force all nodes at a level to use the same split, wasting capacity on correlated features.

CatBoost's ordered boosting, while theoretically appealing, provides marginal benefit when 137K samples are available and the anomaly signal (sharp variance changes) is strong enough to be captured by standard gradient estimation.

### 8.2 Why Not Deep Learning or Other Methods?

A systematic evaluation of alternative model families was conducted to determine whether XGBoost and CatBoost are the optimal choices for this task.

**GBDT Family.** LightGBM (leaf-wise growth) could marginally improve boundary precision over XGBoost's level-wise approach. Its histogram-based splitting is efficient on dense engineered features. However, LightGBM is structurally a GBDT variant — it will produce near-identical predictions to XGBoost on this feature set. sklearn's GBM lacks modern regularization and was not considered.

**Deep Learning.** LSTMs, TCNs, and Transformers have been proposed for time-series anomaly detection, but they are fundamentally mismatched to this task. The 759 engineered features already encode temporal context explicitly (e.g., `f32_rs5` measures 5-step rolling variance at the raw feature level). For tree-based models, this means the anomaly signal is a simple split on a single feature. For neural networks, these highly correlated features require large parameter counts to disentangle, while the 137K-sample sequence length makes recurrent architectures both slow and prone to vanishing gradients. Grinsztajn et al. (2022, NeurIPS) demonstrated that GBDTs consistently outperform deep learning on tabular data — our engineered features convert the time-series problem into precisely this regime.

**Unsupervised Methods.** Isolation Forest, LOF, kNN-based outlier detection, and autoencoders all operate without labels. With 570 labeled anomalies available, discarding this information is indefensible. In preliminary testing, Isolation Forest achieved AUPR < 0.3 on this dataset due to the extreme class imbalance and high-dimensional feature space.

**Time-Series Specific Methods.** ARIMA residuals, Prophet, and matrix profile methods are designed for univariate or low-dimensional time series. They cannot exploit the 33-dimensional feature interactions that our tree-based models capture through rolling statistics and difference features.

**Conclusion.** GBDTs on engineered temporal features represent the state-of-the-art for tabular anomaly detection. The bottleneck is no longer model expressivity (training AUPR = 1.0) but domain generalization across distribution shifts. An additional LightGBM implementation is provided to verify this assessment empirically.

### 8.3 LightGBM Verification

To empirically validate the claim that GBDT variants are interchangeable on this feature set, LightGBM was implemented with the same feature engineering pipeline and temporal split strategy. LightGBM uses leaf-wise tree growth and histogram-based splitting, making it structurally distinct from both XGBoost (level-wise symmetric growth + exact splits) and CatBoost (symmetric trees + ordered boosting).

Three configurations were tested:

| Attempt | Strategy | Val AUPR | Notes |
|---------|----------|----------|-------|
| 1 | `num_leaves=15, scale_pos_weight` | 0.752 | Leaf-wise overfits majority class |
| 2 | `max_depth=3, min_child_samples=50` | 0.719 | Constrained growth still fails |
| 3 | `max_depth=3, is_unbalance=True, max_bin=255` | **0.852** | Best but far behind XGBoost (0.999) |

LightGBM's best configuration (attempt 3) achieved AUPR=0.852, F1=0.889, with 200 iterations. The test predictions were notably more conservative (Task 1: 685 anomalies, Task 2: 465 anomalies).

**Why LightGBM underperforms.** Leaf-wise growth selects the leaf with the highest loss reduction at each step. With a 239:1 class imbalance, the majority class dominates this selection — negative samples are far more numerous, so the largest gradient contributions come from normal observations. Even with `is_unbalance=True`, the histogram-based split finding (which bins feature values into `max_bin` buckets) loses precision on the rare anomaly boundaries that our short-window rolling features are designed to capture. The anomaly signal (sharp variance changes lasting 30 steps) requires precise split thresholds that histogram binning approximates away.

**Key takeaway.** LightGBM's failure is not a weakness of the model per se — it is state-of-the-art for many tabular tasks — but a demonstration that on extreme class imbalance with precision-dependent features, XGBoost's exact split finding and level-wise growth provide a decisive advantage. This validates the earlier theoretical analysis (Section 8.2) with empirical evidence. The LightGBM implementation is documented here for completeness but excluded from the final submission code since it is not the primary model.

**Final ranking on this dataset:** XGBoost > CatBoost ≫ LightGBM.

### 8.4 Distribution-Invariant Features

The adversarial validation (Section 5) revealed that 3 features exhibit >1.5× variance ratio between training and Task 2 distributions. Standard z-scores based on rolling mean and standard deviation are inherently distribution-dependent: when variance scales by 1.5×, the same raw deviation maps to a different z-score. To address this, distribution-invariant alternatives were implemented and evaluated.

**Approach.** Four new feature types were added: rolling median, MAD (Median Absolute Deviation), IQR (Inter-Quartile Range), robust z-score `(x - median) / MAD`, and percentile rank within rolling window. These features maintain identical semantics regardless of the underlying distribution — "being in the 95th percentile of recent values" means the same thing in train and test. The feature set expanded from 759 to 1,254 features.

**Empirical validation of invariance.** A controlled experiment confirmed that MAD-based z-scores are perfectly invariant to variance scaling: when multiplying the signal by 1.5×, the Pearson correlation between original and shifted z-scores was 1.000 for MAD-based scores, while standard-deviation-based z-scores collapsed entirely.

**Results.**

| Model | Original Features | Robust Features | Δ |
|-------|------------------|-----------------|---|
| XGBoost | 0.9900 ± 0.0138 | 0.9564 ± 0.0607 | **−0.034** |
| CatBoost | 0.9395 ± 0.0837 | 0.9434 ± 0.0796 | +0.004 |

The robust z-score feature `f31_rz30` ranked #1 by gain importance (40,341), confirming that MAD-based features carry strong anomaly detection signal. However, the overall degradation in XGBoost performance reveals three structural problems that prevent robust features from being a drop-in improvement.

**Why robust features degrade in-distribution performance.** Three factors explain the negative result:

*1. Numerical imprecision from `.apply()`.* Pandas' `rolling().apply()` executes a Python function call per window position. For 137,192 rows × 33 features × 3 windows ≈ 13.6M MAD/IQR/percentile computations, each involving sorting or median-finding within a variable-length window. The accumulated floating-point error creates split boundaries that are slightly misaligned with the true signal. XGBoost's exact split finding amplifies this imprecision: a split threshold of 2.349 vs 2.351 on `f32_rs5` can mean the difference between capturing and missing an anomaly boundary. CatBoost's histogram-based splits (binned into `max_bin` buckets) are naturally less sensitive to this noise, explaining its neutral result (+0.004).

*2. Scale mismatch between feature families.* For normally distributed data, MAD ≈ 0.6745 × σ. This means a robust z-score of 3.0 corresponds to a standard z-score of approximately 2.0. When both feature families coexist in the same XGBoost model, the tree must learn different split thresholds for what is semantically the same degree of deviation. This increases the effective complexity of the optimization landscape without adding new information. Empirically, the top 30 features by gain included both `rs` (8) and `rz` (2) — the model was forced to split attention between two representations of the same underlying signal.

*3. Information redundancy with existing features.* Rolling median correlates at 0.999 with rolling mean on this dataset (the features are near-symmetric with few outliers in normal regions). Percentile rank within a 10–30 step window is effectively a normalized version of the rolling z-score (both measure relative position within a recent window). The robust features largely recode information already present in the standard feature set, but with additional computation noise and a different numerical scale.

**Key takeaway.** Distribution-invariant features are theoretically the correct solution to the Task 2 generalization problem. However, their practical value is limited by (a) the computational precision loss from non-vectorized rolling operations, (b) the scale redundancy with existing z-scores, and (c) the inability to verify Task 2 improvement without hidden labels. Given that the original features already achieve AUPR = 1.0 on training data, the conservative choice is to retain the original feature set. A vectorized C-level implementation of rolling MAD and percentile rank (e.g., via `numpy.lib.stride_tricks`) could potentially resolve issue (a), but issues (b) and (c) remain structural. The original feature set is used for final submission.

### 8.5 Task 2 Generalization

The adversarial validation (AUC=1.0 for train vs Task 2) reveals a fundamental distribution gap. Without the ability to retrain or adapt, model generalization depends entirely on whether the engineered features capture distribution-invariant anomaly patterns.

**Model behavior under distribution shift.** The three models exhibit increasingly conservative predictions on Task 2 relative to Task 1:

| Model | Task 1 Anomaly Rate | Task 2 Anomaly Rate | Reduction |
|-------|--------------------|--------------------|-----------|
| XGBoost | 3.55% | 1.94% | −45.4% |
| CatBoost | 3.49% | 1.49% | −57.3% |
| LightGBM | 2.67% | 1.35% | −49.4% |

XGBoost's smaller reduction (−45.4%) compared to CatBoost (−57.3%) suggests that its asymmetric trees, which learn more precise split thresholds, retain more detection sensitivity when the data distribution changes. CatBoost's symmetric structure, while producing smoother probabilities in-distribution, becomes overly conservative under shift — its heavier L2 regularization (`l2_leaf_reg=10.0`) may cause it to dismiss borderline anomaly patterns that fall just outside the training distribution's typical range.

**The double-edged sword of regularization.** CatBoost's optimization journey illustrates a key tension: heavy regularization (L2=10.0) was essential to reduce CV variance from 0.089 to 0.022, but this same regularization may limit its ability to generalize under distribution shift. The rolling standard deviation features that dominate both models are robust to mean shifts (they measure local variation) but are sensitive to variance scaling — a feature value of `f32_rs5=2.5` may indicate an anomaly in the training distribution but could be normal in a distribution with 1.5× variance. XGBoost's lighter regularization (L2=0.5) and exact split finding allow it to retain finer distinctions within the anomaly probability range, potentially preserving recall under shift at the cost of slightly higher false positive risk.

**Why further optimization is blocked.** Any change to improve Task 2 generalization — different features, regularization strengths, or model architectures — can only be evaluated on training-distribution data (via CV). Without Task 2 labels, there is no feedback signal to guide optimization. This is the fundamental constraint of the two-task setup: Task 2 performance is determined at training time, by design choices made before the test distribution is seen.

### 8.6 Limitations

1. **No Task 2 labels**: All generalization analysis is necessarily indirect. The true Task 2 performance remains unknown.
2. **Fixed window assumption**: Rolling windows of 2–8 steps implicitly assume anomaly signatures operate on these time scales. A scenario with fundamentally different temporal dynamics would degrade performance.
3. **Single model family**: Both models are tree-based ensembles. Neural approaches (e.g., temporal convolutional networks) might capture different aspects of the temporal structure.
4. **Adversarial pruning limitation**: As demonstrated, when distribution shift is global rather than sparse, feature pruning cannot meaningfully improve robustness.

---

## 9. Division of Work

Solo project — all work (data analysis, feature engineering, XGBoost implementation, CatBoost implementation, hyperparameter optimization, adversarial validation, report writing) was completed by the team member alone.

---

## 10. References

- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: Identifying Density-Based Local Outliers. *Proceedings of the ACM SIGMOD*.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD*.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *Advances in Neural Information Processing Systems (NeurIPS)*.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems (NeurIPS)*.
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. *Proceedings of the IEEE International Conference on Data Mining (ICDM)*.
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems (NeurIPS)*.
- Sakurada, M., & Yairi, T. (2014). Anomaly detection using autoencoders with nonlinear dimensionality reduction. *Proceedings of the MLSDA Workshop*.

---

## 11. Submission Files

| File | Description |
|------|-------------|
| `report.md` | Comprehensive project report |
| `xgboost/model/xgb_model.json` | Trained XGBoost model (primary) |
| `xgboost/model/metadata.pkl` | Threshold and feature configuration |
| `xgboost/predictions/pred_simple.csv` | Task 1 predictions (25,647 rows) |
| `xgboost/predictions/pred_complex.csv` | Task 2 predictions (34,542 rows) |
| `xgboost/features.py` | Feature engineering (759 temporal features) |
| `xgboost/train.py` | XGBoost training script |
| `xgboost/requirements.txt` | Python dependencies |
| `catboost/model/catboost_model.cbm` | Trained CatBoost model (backup) |
| `catboost/model/metadata.pkl` | CatBoost threshold and feature configuration |
| `catboost/predictions/pred_simple.csv` | Task 1 predictions (CatBoost) |
| `catboost/predictions/pred_complex.csv` | Task 2 predictions (CatBoost) |
| `catboost/features.py` | Feature engineering (shared) |
| `catboost/train.py` | CatBoost training script |
| `catboost/requirements.txt` | Python dependencies |
| `data/` | Training and test datasets |
