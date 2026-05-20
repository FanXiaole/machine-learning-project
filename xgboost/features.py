"""Feature engineering for time-series anomaly detection with XGBoost.

Creates sliding-window temporal features (lags, rolling statistics, differencing)
from raw time-ordered feature vectors. Designed for the Robust Anomaly Detection project.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def compute_rolling_features(
    series: pd.Series,
    windows: List[int],
    min_periods: int = 1,
) -> pd.DataFrame:
    """Compute rolling mean and std for a single series over multiple window sizes."""
    result = {}
    name = series.name
    for w in windows:
        result[f"{name}_rm{w}"] = series.rolling(w, min_periods=min_periods).mean()
        result[f"{name}_rs{w}"] = series.rolling(w, min_periods=min_periods).std()
        result[f"{name}_rmax{w}"] = series.rolling(w, min_periods=min_periods).max()
        result[f"{name}_rmin{w}"] = series.rolling(w, min_periods=min_periods).min()
    return pd.DataFrame(result)


def compute_lag_features(series: pd.Series, lags: List[int]) -> pd.DataFrame:
    """Compute lag features for a single series."""
    result = {}
    name = series.name
    for lag in lags:
        result[f"{name}_lag{lag}"] = series.shift(lag)
    return pd.DataFrame(result)


def compute_diff_features(series: pd.Series, diffs: List[int]) -> pd.DataFrame:
    """Compute difference features for a single series."""
    result = {}
    name = series.name
    for d in diffs:
        result[f"{name}_diff{d}"] = series.diff(d)
    return pd.DataFrame(result)


def compute_zscore_features(series: pd.Series, windows: List[int]) -> pd.DataFrame:
    """Compute z-score relative to a rolling window (how many stds from rolling mean)."""
    result = {}
    name = series.name
    for w in windows:
        rm = series.rolling(w, min_periods=1).mean()
        rs = series.rolling(w, min_periods=1).std().replace(0, np.nan)
        result[f"{name}_z{w}"] = (series - rm) / rs
    return pd.DataFrame(result)


def compute_robust_features(series: pd.Series, windows: List[int]) -> pd.DataFrame:
    """Distribution-invariant alternatives to mean/std.

    - Rolling median: robust to outliers, scale-invariant
    - MAD (Median Absolute Deviation): robust std, unchanged under variance scaling
    - IQR (Inter-Quartile Range): robust spread, unchanged under shift+scale
    - Robust z-score: (x - median) / MAD, equivalent semantics across distributions
    """
    result = {}
    name = series.name
    for w in windows:
        rmed = series.rolling(w, min_periods=1).median()
        # MAD = median(|x_i - median(x)|) over the window
        rmad = series.rolling(w, min_periods=1).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )
        # IQR = Q75 - Q25
        riqr = series.rolling(w, min_periods=1).apply(
            lambda x: np.percentile(x, 75) - np.percentile(x, 25), raw=True
        )
        result[f"{name}_rmed{w}"] = rmed
        result[f"{name}_rmad{w}"] = rmad
        result[f"{name}_riqr{w}"] = riqr
        # Robust z-score: (x - median) / MAD
        result[f"{name}_rz{w}"] = (series - rmed) / rmad.replace(0, np.nan)
    return pd.DataFrame(result)


def compute_percentile_rank(series: pd.Series, windows: List[int]) -> pd.DataFrame:
    """Percentile rank within rolling window.

    Fully distribution-invariant: 'being in the 95th percentile of recent values'
    means the same thing regardless of the underlying distribution.
    """
    result = {}
    name = series.name
    for w in windows:
        # Fraction of values in window that are <= current value
        prank = series.rolling(w, min_periods=1).apply(
            lambda x: np.searchsorted(np.sort(x), x[-1]) / len(x), raw=True
        )
        result[f"{name}_prank{w}"] = prank
    return pd.DataFrame(result)


def engineer_features(
    df: pd.DataFrame,
    rolling_windows: Optional[List[int]] = None,
    lag_steps: Optional[List[int]] = None,
    diff_steps: Optional[List[int]] = None,
    zscore_windows: Optional[List[int]] = None,
    features_for_rolling: Optional[List[str]] = None,
    features_for_lag: Optional[List[str]] = None,
    features_for_diff: Optional[List[str]] = None,
    robust: bool = False,
    robust_windows: Optional[List[int]] = None,
    prank_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Create comprehensive temporal features from raw time-series data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw feature dataframe with columns f1..f33.
    rolling_windows : list of int
        Window sizes for rolling statistics. Default [2, 3, 5, 8].
    lag_steps : list of int
        Lag steps. Default [1, 2].
    diff_steps : list of int
        Difference steps. Default [1].
    zscore_windows : list of int
        Windows for z-score. Default [5, 10, 15].
    robust : bool
        If True, add distribution-invariant features (MAD, median, IQR,
        percentile rank). These maintain consistent semantics under
        distribution shift (e.g., Task 1 → Task 2).
    robust_windows : list of int
        Windows for robust statistics. Default [8, 15, 30].
    prank_windows : list of int
        Windows for percentile rank. Default [10, 20, 30].

    Returns
    -------
    pd.DataFrame with original + engineered features.
    """
    if rolling_windows is None:
        rolling_windows = [2, 3, 5, 8]
    if lag_steps is None:
        lag_steps = [1, 2]
    if diff_steps is None:
        diff_steps = [1]
    if zscore_windows is None:
        zscore_windows = [5, 10, 15]
    if robust_windows is None:
        robust_windows = [8, 15, 30]
    if prank_windows is None:
        prank_windows = [10, 20, 30]

    feature_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]

    if features_for_rolling is None:
        features_for_rolling = feature_cols
    if features_for_lag is None:
        features_for_lag = feature_cols
    if features_for_diff is None:
        features_for_diff = feature_cols

    result_parts = [df]

    print(f"Engineering features on {len(df)} rows...")

    # Rolling statistics
    print("  Computing rolling statistics...")
    rolling_frames = []
    for col in features_for_rolling:
        rolling_frames.append(compute_rolling_features(df[col], rolling_windows))
    if rolling_frames:
        result_parts.append(pd.concat(rolling_frames, axis=1))

    # Lag features
    print("  Computing lag features...")
    lag_frames = []
    for col in features_for_lag:
        lag_frames.append(compute_lag_features(df[col], lag_steps))
    if lag_frames:
        result_parts.append(pd.concat(lag_frames, axis=1))

    # Difference features
    print("  Computing difference features...")
    diff_frames = []
    for col in features_for_diff:
        diff_frames.append(compute_diff_features(df[col], diff_steps))
    if diff_frames:
        result_parts.append(pd.concat(diff_frames, axis=1))

    # Z-score features
    print("  Computing z-score features...")
    zscore_frames = []
    for col in features_for_diff:
        zscore_frames.append(compute_zscore_features(df[col], zscore_windows))
    if zscore_frames:
        result_parts.append(pd.concat(zscore_frames, axis=1))

    # Distribution-invariant robust features
    if robust:
        print("  Computing robust features (MAD/IQR/median)...")
        robust_frames = []
        for col in feature_cols:  # all features for robust stats
            robust_frames.append(compute_robust_features(df[col], robust_windows))
        if robust_frames:
            result_parts.append(pd.concat(robust_frames, axis=1))

        print("  Computing percentile rank features...")
        prank_frames = []
        for col in feature_cols:
            prank_frames.append(compute_percentile_rank(df[col], prank_windows))
        if prank_frames:
            result_parts.append(pd.concat(prank_frames, axis=1))

    # Combine
    result = pd.concat(result_parts, axis=1)

    result = result.ffill().bfill().fillna(0)

    print(f"  Total features after engineering: {result.shape[1]}")
    return result
