# Credit Risk PD Model
This project develops a **Probability of Default (PD)** model to estimate the likelihood of borrower default using machine learning techniques.

The objective is to compare different modeling approaches  used in **credit risk analytics** and evaluate their performance in terms of discrimination and calibration.

## Models Implemented
The following models were trained and evaluated:
+ Logistic Regression
+ Random Forest
+ XGBoost

Logistic regression is widely used in the banking industry due to its **interpretability and regulatory acceptance**, while ensemble methods such as Random Forest and XGBoost are used for **benchmarking predictive performance**.

## Project Workflow
1. Data Loading and Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Train-test split
5. Model Training
6. Model Comparison

## Model Evaluation
The models were evaluated using both classification metrics and risk modeling metrics.

# Classification Metrics
+ Accuracy
+ Precision
+ Recall
+ F1 score

# Risk Modeling Metrics
+ AUC (Area Under the RIC Curve)
+ KS (Kolmogorov-Smirnow Statistic)
+ Calibration (Predicted PD vs Observed Default Rate)

These metrics assess:
+ Model discrimination ability
+ Risk ranking power
+ Alignment between predicted and actual defaults

## Dashboard (Power BI)

An interactive Power BI dashboard was developed to visualize model performance and portfolio risk.

# Key features include:
+ Model comparison using ROC curves
+ Portfolio distribution across PD risk bins
+ Calibration analysis (Predicted vs Observed Default Rate)
+ Summary metrics for model performance

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

## Purpose

This project is a part of my preparation for roles in Credit Risk / Risk Analytics, focusing on building practical skills in PD modeling, model evaluation, and dashboarding using in financial industry.
