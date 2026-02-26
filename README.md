
Project Overview

Course: LUBS5565M Applied AI in Business 
Institution: Leeds University Business School
Author: Aakrati Pancholi
Date: February 2026

Objective: Predict whether weekly business sales will increase or decrease compared to the previous week using machine learning algorithms.

Result: Achieved 64.23% accuracy, exceeding the 56% target by 8.23% and outperforming competitor predictions by **12.5%**.

 Key Results

Metric | Value
Best Model | Random Forest (Weekly)
Accuracy | 64.23%
Target | 56.00% ✅
Improvement over Baseline | +14.23%
Improvement over Daily | +11.08%
Training Accuracy | 71.45%
Test Accuracy | 64.23%
Overfitting Gap | 7.22% (Controlled)

Key Insight

Weekly prediction beats daily prediction by 11.08%

Why? Weekly predictions filter out daily volatility, capture meaningful business trends, and align better with business decision cycles.

Project Structure

```
business-sales-forecasting-ai/
├── data/
│   ├── raw/                    # Original datasets
│   │   ├── sales_daily_2023.csv
│   │   ├── sales_daily_2024.csv
│   │   ├── sales_daily_2025.csv
│   │   └── external_factors.csv
│   └── processed/              # Cleaned features
│       ├── weekly_features.csv
│       ├── training_data.csv
│       └── testing_data.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_building.ipynb
│   └── 04_results_visualization.ipynb
├── models/
│   ├── best_weekly_model.pkl   # Final 64.23% model
│   ├── scaler_weekly.pkl
│   ├── daily_model.pkl
│   └── model_metadata.json
├── src/
│   ├── data_processor.py
│   ├── feature_engineer.py
│   ├── model_trainer.py
│   ├── predict.py
│   └── utils.py
├── visualizations/
│   ├── accuracy_comparison.png
│   ├── feature_importance.png
│   ├── sales_forecast.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── trend_analysis.png
├── results/
│   ├── figures/
│   ├── tables/
│   └── performance_report.txt
├── tests/
│   ├── test_data_processor.py
│   ├── test_model.py
│   └── test_predictions.py
├── requirements.txt
├── config.yaml
├── MODEL_DOCUMENTATION.md
└── README.md
```
 Quick Start
 1. Load Model

```python
import joblib
model = joblib.load('models/best_weekly_model.pkl')
scaler = joblib.load('models/scaler_weekly.pkl')
```
2. Make Prediction

```python
import pandas as pd

features = pd.DataFrame([{
    'MA7': 45000,
    'MA30': 44500,
    'Lag1': 45200,
    'Lag4': 44800,
    'Sales_Return': 0.015
}])

prediction = model.predict(scaler.transform(features))[0]
# 0 = Decrease, 1 = Increase
```

 Model Development Journey

Tested 7 Approaches:

Baseline (Random): 50.00%
Daily Logistic Regression: 52.15%
Daily k-NN: 54.08%
Daily Decision Tree: 51.90%
Daily Random Forest: 53.15% (best daily)
Daily XGBoost (28 features): 48.75% (overfitted)
→ Weekly Random Forest (7 features): 64.23%

The Breakthrough

Switched from daily to weekly prediction (+11.08% accuracy)
Reduced from 28 features to 7 core features
Used shallow trees (max_depth=4) to prevent overfitting
Added business domain features
Optimized hyperparameters for business cycles

Final Model Details

Features Used (7):

MA7 – 7-day moving average
MA30 – 30-day moving average
Lag1 – Sales from 1 week prior
Lag4 – Sales from 4 weeks prior
Sales_Return – Weekly percentage change
Day_of_Week – Day of week encoding
Seasonality – Month/quarter indicator

Model Parameters:

Algorithm: Random Forest
n_estimators: 120
max_depth: 4
min_samples_split: 25
min_samples_leaf: 12

Performance:

Training: 71.45%
Testing: 64.23%
Precision: 0.6512
Recall: 0.6289
F1 Score: 0.6398
ROC-AUC: 0.7145
Overfitting: 7.22% (controlled)

Business Application

This model can be used for:

Weekly sales forecasting
Inventory planning
Revenue prediction
Promotion timing
Resource allocation
Marketing decision support
Risk management
Budget planning

Key Learnings

More features ≠ better accuracy – 7 features beat 28 features
Time horizon matters – Weekly prediction beats daily by ~11%
Simplicity wins – Simple Random Forest beat complex XGBoost
Domain knowledge is essential – Business features performed best
Regularization is critical – Shallow trees prevented overfitting
Business context matters – Weekly predictions align with planning and operations

Technologies Used

Python 3.9+
pandas, numpy – Data processing
scikit-learn – Machine learning
matplotlib, seaborn – Visualization
pytest – Testing

Contact

Aakrati Pancholi
LUBS5565M Applied AI in Business
pancholiaakrati@gmail.com  February 2026

License

This project is for academic and commercial purposes.

