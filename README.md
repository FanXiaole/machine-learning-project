# Robust Time-Series Anomaly Detection

A lightweight, highly optimized machine learning pipeline designed to detect rare, contiguous anomaly streaks (0.42% positive rate) in noisy, high-dimensional time-series data. 

This project evaluates GBDT architectures (XGBoost, CatBoost) on engineered temporal features to handle extreme class imbalance and complex distribution shifts without retraining.

## 🚀 Quick Start

```bash
# Clone the repository
git clone [https://github.com/JustinFan7777777/time-series-anomaly-detection.git](https://github.com/JustinFan7777777/time-series-anomaly-detection.git)
cd time-series-anomaly-detection/xgboost

# Install dependencies
pip install -r requirements.txt

# Run feature engineering and model training
python train.py
