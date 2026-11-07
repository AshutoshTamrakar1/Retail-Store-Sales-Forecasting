# WOMart Retail Store Sales Forecasting (Job-a-Thon September)

## Problem Statement
**WOMart** is a leading nutrition and supplement retail chain operating **365 retail stores** across **4 regions**, **4 store types**, and **5 location types**. Accurate daily store-level sales forecasting provides essential visibility into upcoming cash flow and inventory planning.

Given **17 months** of historical daily sales data (`2018-01-01` to `2019-05-31`, `188,340` records) including store attributes, holiday flags, promotional discounts, and daily order counts (`#Order`), the objective is to forecast daily store sales for each of the 365 stores over the **next two months** (`2019-06-01` to `2019-07-31`, `22,265` records).

* **Evaluation Metric:** $\text{MSLE} \times 1000$

$$\text{Score} = 1000 \times \frac{1}{N} \sum_{i=1}^{N} \left( \log(1 + y_i) - \log(1 + \hat{y}_i) \right)^2$$

---

## Project Evolution: 2021 Competition vs. Refactored Upgrade

| Dimension | Original 2021 Submission | Refactored & Upgraded Pipeline |
| :--- | :--- | :--- |
| **Target Objective** | Raw `Sales` with default MSE (`reg:squarederror`) | Metric-aligned `np.log1p(Sales)` transformation (directly minimizes MSLE) |
| **Validation Strategy** | Random shuffled 5-Fold CV (`KFold(shuffle=True)`) | Leak-free **Chronological Split** (`2019-04-01` to `2019-05-31`, 61 days) |
| **Temporal & Promo Features** | `Date` column dropped | Rich calendar, cyclical ($\sin/\cos$), promo-streak (`discount_streak`), and holiday/discount lead-lag features |
| **Handling of `#Order`** | Dropped because absent in `TEST_FINAL.csv` | Used via leak-free historical store `AOV` / `#Order` profiles + **Stage-1 auxiliary `log1p(#Order)` predictor** |
| **Categorical Encoding** | Silent copy-paste bug (`'cat' in col` matched 0 cols) | Proper `OneHotEncoder(sparse_output=False)` fit strictly on training split |
| **Zero-Sales Anomalies** | Left 19 closed-store rows (`Sales == 0`) unfiltered | Filtered out 19 closed-store anomaly rows during training to prevent extreme log-error gradient distortion |
| **Modeling & Regularization** | Single `XGBRegressor` (`7407` trees, no early stopping) | **Two-Stage Blended Ensemble** (`XGBRegressor` with early stopping + `HistGradientBoostingRegressor` + `Ridge` baseline) |

---

## What New Techniques We Tried

1. **Eliminating Temporal Leakage with an Out-of-Time Holdout Split:**
   * In 2021, random 5-fold cross-validation (`KFold(shuffle=True)`) allowed the model to peek at future and same-day store sales to predict past sales, causing a massive gap between local CV (`105.5`) and the Private Leaderboard (`226.93`).
   * We replaced random K-Fold with a **strictly chronological 61-day validation split** (`2019-04-01` to `2019-05-31`, `22,265` rows), matching the exact 61-day test horizon (`2019-06-01` to `2019-07-31`).
2. **Fixing the Silent Categorical Encoding Bug & Retaining `Date`:**
   * Fixed the 2021 bug where `object_cols = [col for col in useful_features if 'cat' in col]` matched zero columns (`Store_Type`, `Location_Type`, `Region_Code` were silently ignored).
   * Retained `Date` and engineered **day-of-week, weekend, month-start/end, cyclical Fourier transforms ($\sin/\cos$), consecutive discount streak counters (`discount_streak`), rolling 7-day/14-day promo density, and lead/lag indicators (`disc_lag1`, `disc_lead1`, `hol_lag1`, `hol_lead1`)** to capture pre- and post-promotion shopping behavior.
3. **Direct Log-Target Optimization (`log1p(Sales)`) & Anomaly Filtering:**
   * Daily store sales are right-skewed (mean USD 42,784, max USD 247,215) with **19 zero-sales store closures** (`Sales == 0`, `#Order == 0`). Optimizing raw MSE heavily over-penalized high-volume days while ignoring proportional log errors.
   * Filtering the 19 closed-store rows during training and predicting $z = \log(1 + \text{Sales})$ (inverting via $\hat{y} = \max(0, \exp(\hat{z}) - 1)$) aligned our loss function directly with $\text{MSLE} \times 1000$.
4. **Two-Stage Auxiliary `#Order` Forecasting & Store Hierarchy Encodings:**
   * Because `Sales` is the product of order volume (`#Order`) and Average Order Value ($\text{AOV} = \frac{\text{Sales}}{\text{Orders}}$), dropping `#Order` threw away critical signal.
   * Using our leak-free `HistoricalStoreAggregator`, we extracted each store's historical `log1p(#Order)` baseline, historical `AOV`, recent 60-day sales momentum, and hierarchical target encodings for `(Store_id, DayOfWeek, Discount)` and `(Store_id, Holiday, Discount)`.
   * We then built a **Two-Stage Pipeline**: **Stage 1** predicts daily store `log1p(#Order)` (`pred_log_orders`), and **Stage 2** combines `pred_log_orders` with store `AOV` and calendar/promo features to forecast `log1p(Sales)`.

---

## How It Improved Our Results

All models were evaluated on the leak-free 61-day chronological holdout window (`2019-04-01` to `2019-05-31`, `22,265` store-days), with **95% bootstrap confidence intervals** computed via 200 date-block resamples:

| Model | Out-of-Time $\text{MSLE} \times 1000$ ↓ | 95% Bootstrap CI | Out-of-Time RMSE (USD) ↓ | Out-of-Time MAE (USD) ↓ |
| :--- | :---: | :---: | :---: | :---: |
| **1. Naive Store Historical Mean** | `124.177` | `[105.04, 141.30]` | `16,523.53` | `11,109.77` |
| **3. Ridge Regression (`log1p` Target)** | `97.638` | `[81.77, 116.71]` | `13,844.70` | `9,275.16` |
| **2. 2021 Baseline (No `Date`, Raw MSE)** | `96.460` | `[80.41, 119.14]` | `14,196.83` | `9,423.21` |
| **4. Two-Stage HistGBM (`log1p` Target)** | `78.046` | `[61.30, 99.34]` | **`12,402.36`** | **`7,946.85`** |
| **6. Ensemble (60% XGB + 40% HistGBM)** | `77.451` | **`[60.89, 97.92]`** | `12,413.29` | `7,962.05` |
| **5. Two-Stage XGBoost (Early Stopped)** | **`77.297`** | `[60.34, 100.99]` | `12,437.06` | `7,987.33` |

### Key Performance Highlights
* **~20% Reduction in Out-of-Time $\text{MSLE} \times 1000$:** The upgraded Two-Stage XGBoost and Blended Ensemble pipelines reduced out-of-time $\text{MSLE} \times 1000$ from **`96.460`** (2021 Baseline) down to **`77.297`** and **`77.451`** respectively—an improvement of **19.16 MSLE points** over the 2021 approach and **46.88 points** over the Naive Store Mean benchmark.
* **Over USD 1,750 Lower Daily Store Error:** Out-of-time RMSE dropped from **USD 14,196.83** to **USD 12,402.36**, and MAE dropped from **USD 9,423.21** to **USD 7,946.85** per store-day.
* **Tighter Tail Risk:** Blending `XGBRegressor` (60%) and `HistGradientBoostingRegressor` (40%) achieved the **tightest 95% bootstrap confidence interval upper bound (`97.92` vs. `100.99` for single XGBoost and `119.14` for the 2021 baseline)**, providing more reliable multi-month forecasts.
* **Top Predictive Drivers:** Feature importance analysis confirmed that our newly engineered **`pred_log_orders` (Stage-1 predicted order volume)**, **`store_dow_disc_te` (store $\times$ day-of-week $\times$ discount encoding)**, and **`store_log_aov_mean` (store historical Average Order Value)** are the top three predictors of daily store revenue.

---

## Repository Structure

* [`job-a-thon-final.ipynb`](job-a-thon-final.ipynb) — End-to-end refactored notebook: EDA, stationarity & anomaly analysis, leak-free chronological feature engineering, multi-model comparison with 95% bootstrap confidence intervals, slice error analysis, and test forecasting.
* [`TRAIN.csv`](TRAIN.csv) — Historical daily sales and order counts for 365 stores (`2018-01-01` to `2019-05-31`).
* [`TEST_FINAL.csv`](TEST_FINAL.csv) — 2-month forecasting horizon (`2019-06-01` to `2019-07-31`).
* [`SAMPLE.csv`](SAMPLE.csv) — Sample submission format.
* [`sample_submission_final.csv`](sample_submission_final.csv) — Generated test predictions (`22,265` rows) from the full-data refit of the Two-Stage Ensemble.
* [`requirements.txt`](requirements.txt) — Python dependencies required to run the notebook.

---

## Original 2021 Competition Result
* **Private Leaderboard Rank:** 152 (Score: `226.93`)
* **Public Leaderboard Rank:** 317 (Score: `228.56`)
