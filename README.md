# Credit Risk — Probability of Default (PD) Model

## Overview
An End-to-end credit risk model that predicts the probability of default (PD) for retail loans. Built following industry standards, including WoE/IV feature encoding, model calibration, and IFRS 9 Expected Credit Loss (ECL) computation.

---

## Dataset
- **Source:** Kaggle — Credit Risk Dataset
- **Size:** ~32,000 loan records, 11 features
- **Target:** `loan_status` (1 = Default, 0 = Non-Default)
- **Portfolio Default Rate:** 21.9%

---

## Methodology

### 1. Data Cleaning
- Removed duplicate entries
- Imputed null values via bucket-wise filling
- Standardised data types and removed leaky features
- Removed outliers in `person_age` and `person_emp_length`

Leakage Analysis:
- **Dropped `loan_grade`**, assigned by lender based on borrower risk profile, making it a target-derived feature. Including it would cause data leakage and inflate model performance
- **Retained `loan_int_rate`**, treated as a loan characteristic available at origination. Documented as a potential leakage risk
- **Retained `cb_person_default_on_file`** and **`cb_person_cred_hist_length`**, historical credit bureau data available before loan issuance

### 2. Feature Encoding — WoE/IV Binning
Used Weight of Evidence (WoE) encoding via `optbinning` for both numerical and categorical features. Information Value (IV) is used for feature selection.

| Feature | IV | Predictive Power |
|---|---|---|
| loan_percent_income | 0.950 | Strong |
| loan_int_rate | 0.903 | Strong |
| person_income | 0.567 | Strong |
| person_home_ownership | 0.376 | Strong |
| cb_person_default_on_file | 0.164 | Medium |
| loan_intent | 0.096 | Weak |
| loan_amnt | 0.092 | Weak |
| person_emp_length | 0.059 | Weak |
| person_age | 0.011 | Dropped (Useless) |
| cb_person_cred_hist_length | 0.005 | Dropped (Useless) |

### 3. Train/Test Split and Class Balancing
- 80/20 train/test split with `random_state=42`
- SMOTE applied **only to the training set** to avoid test set contamination
- Test set retained at natural default rate (~21.6%) to reflect real-world portfolio distribution

### 4. Model Training
Three models trained and calibrated using Platt scaling 
(`CalibratedClassifierCV`):
- Logistic Regression
- Random Forest
- XGBoost (Champion Model)

### 5. Model Evaluation

| Model | AUC | Gini | KS | PSI | Mean PD |
|---|---|---|---|---|---|
| Logistic Regression | 0.88 | 0.76 | 0.62 | 0.0011 | 36.3% |
| Random Forest | 0.89 | 0.78 | 0.65 | 0.0010 | 32.2% |
| **XGBoost** | **0.91** | **0.82** | **0.66** | **0.0017** | **27.2%** |

All three models are stable (PSI < 0.10).

**XGBoost selected as the champion model** based on best discrimination (AUC, Gini, KS) and Mean PD closest to the actual portfolio default rate.

**Note on PD overestimation:** Mean PD across all models exceeds the actual default rate of 21.9%. This is expected — SMOTE balanced the training set to 50/50, shifting the model's prior toward higher default probabilities. Platt scaling partially corrects this. In production, post-calibration scaling would align the mean PD with the observed portfolio default rate.

### 6. Feature Importance

Feature importance rankings were broadly consistent across all three models and aligned with IV rankings, validating feature selection.

| Feature | LR Rank | RF Rank | XGB Rank | IV Rank |
|---|---|---|---|---|
| loan_percent_income | 4 | 2 | 1 | 2 |
| loan_int_rate | 2 | 1 | 2 | 1 |
| person_income | 3 | 3 | 4 | 3 |
| person_home_ownership | 5 | 4 | 3 | 5 |
| cb_person_default_on_file | 7 | 7 | 8 | 7 |
| loan_intent | 1 | 5 | 5 | 4 |
| loan_amnt | 6 | 6 | 7 | 6 |
| person_emp_length | 8 | 8 | 6 | 7 |

**Notable finding:** `loan_intent` ranked #1 in Logistic Regression but #5 in tree models. Borrowers taking Medical or Debt Consolidation loans tend to have a higher debt burden relative to income, and tree models capture this signal through `loan_percent_income` directly, while Logistic Regression weights `loan_intent` more heavily via its WoE encoding.

### 7. ECL Computation — IFRS 9
Expected Credit Loss computed at loan level using:

**ECL = PD × LGD × EAD**

- **PD** — XGBoost predicted probability of default
- **LGD** — 45% (Basel Foundation IRB standard assumption)
- **EAD** — Outstanding loan amount

| Metric | Value |
|---|---|
| Total Portfolio EAD: | $310,827,000 |
| Total Portfiolio ECL: | $42,545,435 |
| ECL as % of portfolio: | 13.69% |
| Mean PD: | 27.82% |
| Actual Default Rate: | 21.88% |

**ECL by Loan Intent:**

| Loan Intent | Mean PD | ECL Rate | Risk Level |
|---|---|---|---|
| Debt Consolidation | 33.9% | 17.0% | Highest |
| Medical | 33.6% | 16.8% | High |
| Home Improvement | 31.8% | 15.1% | Medium |
| Personal | 28.4% | 13.5% | Medium |
| Education | 23.0% | 11.3% | Lower |
| Venture | 18.6% | 9.3% | Lowest |

Debt Consolidation and Medical loans are the highest-risk segments, consistent with borrowers already under financial stress. Venture loans are the safest, likely reflecting stronger income profiles among business borrowers.

---

## Requirements

pandas
numpy
scikit-learn
xgboost
imbalanced-learn
optbinning
matplotlib
seaborn
scipy

---

## Key References
- Basel III —  IRB approach, LGD assumptions
- IFRS 9 —  ECL computation framework
- SR 11-7 — Model Risk Management Guidance

