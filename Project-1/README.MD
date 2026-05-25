# Naive Bayes Transaction Data Analysis

**Deep Dive Data Science — Cohort 20**
*Abraham De Vargas | March 2026*

---

## Overview

Banks and financial institutions process millions of transactions daily. This project explores whether we can predict — before it happens — whether a customer will complete a transaction.

**Business Question:** Will this customer make a transaction or not?

This is a **supervised binary classification** problem solved using **Gaussian Naive Bayes**. The `target` column encodes the outcome: `1` = transaction occurred, `0` = it did not.

---

## Dataset

- **Source:** AWS S3 (provided by Deep Dive Coding)
- **Full size:** ~833 MB | 1,050,000 rows | 104 columns
- **Labeled rows:** 180,000 (~17.14% of total)
- **Predictors:** 100 anonymized numeric columns
- **Class distribution:** ~90% non-transaction (`0`), ~10% transaction (`1`)

---

## Project Workflow

### 1. Problem Definition
Framed as binary classification. Identified class imbalance (~90/10 split) as a key challenge requiring attention before modeling.

### 2. Data Cleaning
- Dropped ~82.86% of rows with null `target` values (unlabeled — unusable for supervised learning)
- Dropped unique identifier columns (`Unnamed: 0`, `ID_code`)
- Dropped 51 predictor columns that were entirely null after removing unlabeled rows
- Imputed remaining nulls with column medians (robust to skew and outliers)
- Confirmed no duplicate rows

**After cleaning:** ~180,000 rows × 52 columns (1 target + 51 predictors)

### 3. Exploratory Data Analysis
- Most predictor distributions are approximately Gaussian — consistent with GNB assumptions
- No predictor pairs exceeded |0.8| correlation — independence assumption reasonably satisfied
- Confirmed heavy class imbalance (~90% class `0`) requiring balancing before training

### 4. Modeling

**Unbalanced dataset:**
- 100-iteration cross-validated accuracy: **~91.1%**
- Misleading — model nearly always predicts class `0`
- Poor recall for class `1` (successful transactions)

**Balanced dataset (50/50 undersample):**
- 100-iteration cross-validated accuracy: **~75.5%**
- Both classes predicted meaningfully
- Improved recall for class `1` — a more honest and useful model

---

## Key Results

| Metric | Unbalanced | Balanced |
|---|---|---|
| CV Mean Accuracy | ~0.9111 | ~0.7551 |
| Class `1` Recall | Low | Improved |
| Model Bias | Heavy toward `0` | Balanced |

> **Takeaway:** The unbalanced model's high accuracy is deceptive. Balancing the dataset drops raw accuracy but produces a model that genuinely distinguishes between transaction outcomes.

---

## Future Work

- Evaluate alternative models: Logistic Regression, Decision Tree, Random Forest
- Apply **SMOTE** (Synthetic Minority Over-sampling Technique) instead of undersampling
- Perform feature selection to identify the most predictive variables
- Tune classification threshold to optimize precision/recall trade-off
- Evaluate with **AUC-ROC** for a threshold-independent performance metric

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | GaussianNB, train/test split, metrics |
| `matplotlib` / `seaborn` | Visualization |
| `plotly` | Interactive plotting |

---

## How to Run

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn plotly
   ```
3. Open the notebook and run all cells top to bottom — data loads directly from S3, no local files needed.

---

## Author

**Abraham De Vargas**
Deep Dive Data Science, Cohort 20
