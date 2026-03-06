# DACON Trade Forecasting Competition

Machine learning project developed for a DACON data science competition.

This project predicts future trade volume by identifying **lead–lag relationships between items in time-series data** and using them as predictive features.

---

# Overview

In many forecasting tasks, items are predicted independently.
However, in real-world trade data, some items tend to **lead or follow the trends of other items**.

This project detects these **lead–lag relationships** and incorporates them into a machine learning model to improve prediction performance.

---

# Dataset

The dataset consists of trade records including:

* `item_id`
* `year`
* `month`
* `value` (trade volume)

The data was transformed into **monthly time-series format** for each item.

---

# Method

## 1. Data Aggregation

Trade data was aggregated by **item and month**.

```python
monthly = (
    train
    .groupby(["item_id", "year", "month"], as_index=False)["value"]
    .sum()
)
```

Then the dataset was converted into a **pivot table**:

* rows → item_id
* columns → month
* values → trade volume

---

## 2. Lead–Lag Relationship Detection

For each pair of items:

* Lagged correlations were calculated
* Optimal lag was identified
* Pairs with strong correlation were selected

```python
corr = np.corrcoef(x[:-lag], y[lag:])
```

This allows the model to capture **temporal dependencies between items**.

---

## 3. Feature Engineering

The following features were used:

| Feature  | Description                       |
| -------- | --------------------------------- |
| b_t      | current value of the target item  |
| b_t_1    | previous value of the target item |
| a_t_lag  | lagged value of the leading item  |
| max_corr | correlation strength              |
| best_lag | optimal lag between items         |

Target variable:

```
b_t_plus_1
```

Next month's trade volume.

---

## 4. Model

AutoML was used to automatically search for the best model.

Library:

FLAML AutoML

Candidate models:

* Random Forest
* LightGBM
* XGBoost

Evaluation metric:

Mean Squared Error (MSE)

```python
automl.fit(
    X_train=train_X,
    y_train=train_y,
    task="regression",
    metric="mse"
)
```

---

# Result

DACON Trade Forecasting Competition

* Rank: **74 / 960**
* Top **7.7%**
* Total participants: **1,768**

---

# Key Idea

Instead of predicting each item independently, this project introduces **lead–lag relationships between items as predictive features**.

This approach allows the model to capture **inter-item temporal dependencies** in trade data.

---

# Tech Stack

Python
Pandas
NumPy
Scikit-learn
FLAML AutoML

---

# Future Improvements

Possible improvements include:

* Time-series models (LSTM, Transformer)
* More advanced feature engineering
* Cross-validation for time-series data
* Graph-based relationship modeling between items
