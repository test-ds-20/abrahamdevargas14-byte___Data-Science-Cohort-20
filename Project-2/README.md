
---

# Project 2: Linear Regression Housing Price Prediction
**Deep Dive Data Science Cohort-20 | Abraham De Vargas | April 12, 2026**

---

## Project Overview
This project applies linear regression techniques — including Ridge (L2) and Lasso (L1) regularization — to predict residential housing sale prices using the Ames, Iowa housing dataset. The goal is to build a model that accurately estimates property values while using a minimum set of meaningful features.

---

## Problem Definition
Real estate pricing is data-driven. Buyers, sellers, and lenders all rely on accurate home value estimates, and small errors can translate to thousands of dollars. This project frames the problem as a **supervised regression task**: given ~80 features describing a home's size, quality, age, neighborhood, and more, predict its `SalePrice`.

The primary performance metric is **RMSPE (Root Mean Squared Percentage Error)**, which penalizes errors proportionally rather than in raw dollar terms — making it fair across both inexpensive and high-value homes.

| Question | Answer |
|---|---|
| Business problem | Predict the sale price of a residential home |
| Supervised or unsupervised? | Supervised |
| Target variable | `SalePrice` — continuous numeric value |
| Task type | Regression |
| Performance metric | RMSPE — Root Mean Squared Percentage Error |

---

## Dataset Information
The dataset is the **Ames Housing Dataset**, originally compiled by Professor Dean De Cock, hosted on AWS S3 and provided by Deep Dive Coding. It contains approximately **2,637 rows and 81 columns** covering numeric, ordinal, and nominal features of residential property sales in Ames, Iowa.

Key feature categories include:
- Overall quality and condition ratings
- Above-ground living area (sq ft) and total basement sq ft
- Garage size (cars and area)
- Year built and year remodeled
- Neighborhood
- Basement, fireplace, and pool attributes
- Recent sale prices (`SalePrice` — the target)

> **Note:** Several columns contain nulls that carry meaning — e.g., a null in `Garage Type` indicates no garage exists, not missing data. Around 40 columns are categorical and require encoding before modeling.

---

## Methodology

1. **Data Collection** — Dataset loaded directly from an S3 URL via `pd.read_csv()`. No local download required.

2. **Data Cleaning**
   - Dropped unique identifier (`PID`) and columns with >80% nulls (`Pool QC`, `Misc Feature`, `Alley`)
   - Categorical nulls filled with `'None'` (absence of a feature, not missing data)
   - Numeric nulls imputed with column median (robust to right-skewed distributions)
   - Rows missing `SalePrice` dropped; no duplicate rows found
   - All categorical columns one-hot encoded with `drop_first=True`

3. **Exploratory Data Analysis (EDA)**
   - Analyzed `SalePrice` distribution — confirmed right skew, justified log transformation
   - Computed Pearson correlations to identify top predictors (`Overall Qual`, `Gr Liv Area`, `Garage Cars`, `Total Bsmt SF`)
   - Examined multicollinearity via correlation heatmap
   - Identified and removed 2 outlier homes (>4,000 sq ft, <$200k sale price) as likely non-market transactions

4. **Model Development** — Implemented three models using Scikit-learn: plain `LinearRegression`, `Ridge (alpha=10)`, and `Lasso (alpha=0.001)`. Target variable log-transformed via `np.log1p()` to minimize proportional errors.

5. **Feature Scaling** — `StandardScaler` applied inside each CV fold (fit on train only) to prevent data leakage. Required for Ridge and Lasso so no feature dominates due to scale differences.

6. **Training and Validation** — 100-iteration cross-validation (80/20 splits) for stable, low-variance RMSPE estimates.

7. **Lean Model** — Lasso coefficients used to identify a reduced feature set (177 of 262 features). Ridge refit on that subset for improved accuracy and interpretability.

---

## Results

| Model | Mean RMSPE (100-fold CV) | Features Used |
|---|---|---|
| Linear Regression | 0.1619 | 262 |
| Ridge (alpha=10) | 0.1503 | 262 |
| Lasso (alpha=0.001) | 0.1485 | 262 |
| **Lean Ridge (Lasso-selected)** | **0.1451** | **177** |

The **Lean Ridge model is the recommended final model**. Using Lasso to eliminate 85 low-signal features reduced the feature count from 262 to 177 and *improved* RMSPE from 0.1503 to 0.1451 — confirming the dropped features were adding noise rather than signal.

Removing the 2 outlier homes also had a measurable positive impact on model performance. The strongest predictors of sale price were **Overall Quality, Above-Ground Living Area, Neighborhood, and Year Built** — fully consistent with real-world intuition.

---

## Future Work
- Try non-linear models (gradient boosting, XGBoost) to capture feature interaction effects
- Engineer new features such as house age, years since remodel, and total bathrooms
- Tune regularization strength `alpha` using `GridSearchCV` for an optimal bias/variance tradeoff
- Investigate additional outlier types beyond the two large-area/low-price homes already removed
