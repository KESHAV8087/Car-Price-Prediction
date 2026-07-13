# Car Price Prediction 🚗

A machine learning model that predicts the resale value of a used car from its specifications, helping owners and buyers estimate a fair market price instead of relying on potentially lowball dealer offers.

Built in 2022 as part of my B.Tech AI coursework (3rd year).

---

## Problem

When selling a used car, owners are often at an information disadvantage — dealers and traders quote prices that are hard to sanity-check. This project trains a regression model on real used-car listings so a seller can plug in their car's details and get a data-driven price estimate as a negotiation baseline.

---

## Dataset

The [Vehicle Dataset (CarDekho)](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) — 301 used-car listings with the following features:

| Feature | Description |
|---|---|
| `Year` | Year of purchase (2003–2018) |
| `Present_Price` | Current ex-showroom price (lakhs) |
| `Kms_Driven` | Distance driven |
| `Fuel_Type` | Petrol / Diesel / CNG |
| `Seller_Type` | Dealer / Individual |
| `Transmission` | Manual / Automatic |
| `Owner` | Number of previous owners |
| `Selling_Price` | **Target** — resale price (lakhs) |

---

## Approach

1. **Data inspection** — checked shape, dtypes, and confirmed no missing values.
2. **Categorical encoding** — label-encoded `Fuel_Type`, `Seller_Type`, and `Transmission` into numeric form.
3. **Feature/target split** — dropped `Car_Name` and `Selling_Price` from features; `Selling_Price` as the target.
4. **Train/test split** — 90/10 split.
5. **Modeling** — trained and compared two linear models:
   - **Linear Regression**
   - **Lasso Regression** (L1 regularization)
6. **Evaluation** — R² score on both train and test sets, plus actual-vs-predicted scatter plots.

---

## Results

| Model | Train R² | Test R² |
|---|---|---|
| Linear Regression | 0.88 | 0.84 |
| Lasso Regression | 0.84 | 0.87 |

Both models explain the majority of variance in resale price. Lasso generalized slightly better on the held-out test set, while plain Linear Regression fit the training data marginally more closely — a small, interpretable illustration of the bias–variance tradeoff.

---

## Tech Stack

- **Python**
- **pandas** — data loading & preprocessing
- **scikit-learn** — Linear Regression, Lasso, train/test split, R² metric
- **Matplotlib / Seaborn** — visualization

---

## What I'd Do Differently Today

This was an early, foundational ML project. With what I know now, I would add: proper cross-validation instead of a single split, feature scaling for the Lasso model, a "car age" feature engineered from `Year`, and tree-based baselines (Random Forest / XGBoost) for comparison.

---

## Running It

```bash
pip install pandas scikit-learn matplotlib seaborn
jupyter notebook Car_Price_Prediction.ipynb
```
