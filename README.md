# Credit Risk Probability of Default (PD) Model

## Overview
This project develops a **Probability of Default (PD)** model to estimate the likelihood of borrower default using machine learning techniques.

The objective is to compare different modeling approaches used in **credit risk analytics** and evaluate their performance in terms of discrimination, calibration, and risk ranking.

## Models Implemented
The following models were trained and evaluated:
+ Logistic Regression
+ Random Forest
+ XGBoost

Logistic regression is widely used in the banking industry due to its **interpretability and regulatory acceptance**, while ensemble methods such as Random Forest and XGBoost are used to **benchmark predictive performance**.

## Project Workflow
1. Data Loading and Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering (including one-hot encoding of categorical variables)
4. Train-test split
5. Model Training
6. Model Comparison
7. Model Evaluation
8. Calibration Analysis

## Model Evaluation
The models were evaluated using both classification metrics and risk modeling metrics.

Classification Metrics
+ Accuracy
+ Precision
+ Recall
+ F1 Score

Risk Modeling Metrics
+ AUC (Area Under the ROC Curve)
+ KS (Kolmogorov-Smirnow Statistic)
+ Calibration (Predicted PD vs Observed Default Rate)

These metrics assess:
+ Model discrimination ability
+ Risk ranking power
+ Alignment between predicted and actual default rates

## Key Observations

+ Tree-based models (Random Forest and XGBoost) achieved higher AUC, indicating stronger discriminatory power
+ Logistic Regression provides better interpretability and is aligned with industry practices
+ All models show consistent rank ordering, with default rates increasing across risk segments
+ Predicted probabilities are slightly higher than observed default rates, indicating a conservative bias

## Dashboard (Power BI)

An interactive Power BI dashboard was developed to visualize model performance and portfolio risk.

Key features include:
+ Model comparison using ROC curves
+ Portfolio distribution across PD risk bins
+ Calibration analysis (Predicted vs Observed Default Rate)
+ Summary metrics (AUC, KS, Precision, Recall, F1 Score)
+ Feature importance comparison across models

## Tools and Libraries
+ Python
+ Pandas
+ NumPy
+ Scikit-learn
+ XGBoost
+ Matplotlib
+ Seaborn
+ Imbalanced-learn
+ Power BI

## Limitations
+ Tree-based models tend to overestimate probabilities in higher risk segments
+ Calibration can be further improved
+ Feature importance varies across models due to differences in modeling approaches

## Purpose
This project is a part of my preparation for roles in Credit Risk / Risk Analytics, focusing on building practical skills in PD modeling, model evaluation, and data visualization in a financial context.
