
# Market Mover – Stock Market Direction Prediction using Machine Learning

This project predicts short-term directional movement (up or not up) of the S&P 500 index using machine learning. The model is trained to classify whether the market will go up the next trading day based on historical price data and engineered technical features.

---

## Project Overview

- Uses the S&P 500 index historical data (`^GSPC`)
- Applies a binary classification model to predict daily price direction
- Implements a walk-forward cross-validation framework for realistic backtesting
- Evaluates performance with precision, confusion matrix, and classification metrics

---

## Dataset

The data is sourced from Yahoo Finance using the `yfinance` library and includes:

- `Open`, `High`, `Low`, `Close`, `Volume`
- Daily values from 1950 to 2022
- The dataset is saved locally to avoid repeated downloads

```python
import yfinance as yf
sp500 = yf.Ticker("^GSPC").history(period="max")
```

---

## Target Construction

The classification target is whether the price goes up the next day:

```python
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
```

- `Target = 1`: Market goes up tomorrow
- `Target = 0`: Market goes down or stays flat

---

## Model Training

A `GradientBoostingClassifier` is used with the most recent 100 days reserved as the test set:

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1)
predictors = ["Close", "Volume", "Open", "High", "Low"]

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

model.fit(train[predictors], train["Target"])
```

---

## Evaluation (Initial)

The model is evaluated using `precision_score`, with predictions plotted alongside actual targets.

```python
from sklearn.metrics import precision_score
precision_score(test["Target"], model.predict(test[predictors]))
```

---

## Backtesting Framework

A walk-forward validation is implemented with a custom backtesting function:

```python
def backtest(data, model, predictors, start=2500, step=250):
    ...
```

This allows the model to be trained on a growing window of past data and tested on future unseen data chunks.

---

## Feature Engineering

Rolling ratio and trend indicators are computed over multiple horizons:

```python
horizons = [2, 5, 60, 250, 1000]
for horizon in horizons:
    sp500[f"Close_Ratio_{horizon}"] = sp500["Close"] / sp500.rolling(horizon).mean()["Close"]
    sp500[f"Trend_{horizon}"] = sp500.shift(1).rolling(horizon).sum()["Target"]
```

These features capture both recent momentum and longer-term trends.

---

## Threshold Tuning

Instead of a default 0.5 threshold, a 0.6 probability threshold is used for classifying "up":

```python
preds = model.predict_proba(test[predictors])[:,1]
preds[preds >= 0.6] = 1
preds[preds < 0.6] = 0
```

---

## Final Evaluation

After backtesting, performance is evaluated using:

- Precision Score
- Confusion Matrix
- Classification Report

```python
from sklearn.metrics import classification_report
print(classification_report(predictions["Target"], predictions["Predictions"]))
```

The confusion matrix is visualized with Seaborn:

```python
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```

---

## Results Summary

- Total predictions: 4738
- Class balance (Target):
  - `1` → 54.7%
  - `0` → 45.3%

- Model Predictions:
  - Predicted `1` → 877 times
  - Predicted `0` → 3861 times

- Precision (class 1): 55.6%

- Classification Report:

```
              precision    recall  f1-score   support

           0       0.46      0.82      0.59      2147
           1       0.56      0.19      0.28      2591

    accuracy                           0.47      4738
```

---

## Requirements

Install the required libraries using pip:

```bash
pip install pandas yfinance scikit-learn matplotlib seaborn
```

---

## File Structure

```
.
├── Market_Mover.ipynb           # Main notebook with code and outputs
├── sp500.csv                    # Cached S&P 500 historical data
├── README.md                    # Project overview and documentation
```

---


