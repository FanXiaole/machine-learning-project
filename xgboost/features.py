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


def engineer_features(
    df: pd.DataFrame,
    rolling_windows: Optional[List[int]] = None,
    lag_steps: Optional[List[int]] = None,
    diff_steps: Optional[List[int]] = None,
    zscore_windows: Optional[List[int]] = None,
    features_for_rolling: Optional[List[str]] = None,
    features_for_lag: Optional[List[str]] = None,
    features_for_diff: Optional[List[str]] = None,
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """Create comprehensive temporal features from raw time-series data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw feature dataframe with columns f1..f33.
    rolling_windows : list of int
        Window sizes for rolling statistics. Default [5, 10, 20, 30, 50, 100].
    lag_steps : list of int
        Lag steps. Default [1, 2, 3, 5, 10, 20, 30].
    diff_steps : list of int
        Difference steps. Default [1].
    zscore_windows : list of int
        Windows for z-score. Default [30, 50, 100].
    features_for_rolling : list of str
        Which features get rolling stats. Default all f1..f33.
    features_for_lag : list of str
        Which features get lag features. Default all f1..f33.
    fill_method : str
        How to fill NaN values after rolling operations.

    Returns
    -------
    pd.DataFrame with original + engineered features.
    """
    if rolling_windows is None:
        rolling_windows = [2, 3, 5, 8]  # ultra-short: captures anomaly boundary transitions
    if lag_steps is None:
        lag_steps = [1, 2]  # minimal lags for immediate context
    if diff_steps is None:
        diff_steps = [1]
    if zscore_windows is None:
        zscore_windows = [5, 10, 15]  # short-term z-scores for relative deviation

    feature_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]

    if features_for_rolling is None:
        features_for_rolling = feature_cols
    if features_for_lag is None:
        features_for_lag = feature_cols
    if features_for_diff is None:
        features_for_diff = feature_cols

    result_parts = [df.copy()]

    print(f"Engineering features on {len(df)} rows...")

    # Rolling statistics for ALL features
    print("  Computing rolling statistics...")
    rolling_frames = []
    for col in features_for_rolling:
        rolling_frames.append(compute_rolling_features(df[col], rolling_windows))
    if rolling_frames:
        result_parts.append(pd.concat(rolling_frames, axis=1))

    # Lag features for all features
    print("  Computing lag features...")
    lag_frames = []
    for col in features_for_lag:
        lag_frames.append(compute_lag_features(df[col], lag_steps))
    if lag_frames:
        result_parts.append(pd.concat(lag_frames, axis=1))

    # Difference features for all features
    print("  Computing difference features...")
    diff_frames = []
    for col in features_for_diff:
        diff_frames.append(compute_diff_features(df[col], diff_steps))
    if diff_frames:
        result_parts.append(pd.concat(diff_frames, axis=1))

    # Z-score features for all features
    print("  Computing z-score features...")
    zscore_frames = []
    for col in features_for_diff:
        zscore_frames.append(compute_zscore_features(df[col], zscore_windows))
    if zscore_frames:
        result_parts.append(pd.concat(zscore_frames, axis=1))

    # Combine all features
    result = pd.concat(result_parts, axis=1)

    # Fill NaN from rolling/lag/diff operations
    # Forward fill then backward fill for remaining leading NaNs
    result = result.ffill().bfill()

    # Any remaining NaN -> 0
    result = result.fillna(0)

    print(f"  Total features after engineering: {result.shape[1]}")
    return result
