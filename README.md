# EMIPredict AI — Loan Default Risk Assessment

> Predicts EMI default probability before loan disbursement using XGBoost and Random Forest on 400K+ financial records — helping NBFCs and banks reduce credit risk with data-driven underwriting decisions.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-blue?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Live%20Demo-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen?style=flat-square)

🔗 **[Live Demo → Hugging Face Spaces](https://huggingface.co/spaces/Elansurya/EMI-Predict-AI)**

---

## Problem Statement

India's banking sector loses crores annually to non-performing assets (NPAs) caused by EMI defaults. Traditional credit scoring models like CIBIL often miss complex behavioral patterns — particularly for first-time borrowers and thin-file customers applying for personal loans, home loans, and vehicle EMIs.

This project builds an end-to-end ML pipeline that predicts the probability of EMI default using borrower financial profiles — enabling lenders to pre-screen high-risk applications before disbursement and reduce NPA exposure.

---

## Dataset

| Property | Detail |
|---|---|
| Total records | 400,000+ loan transactions |
| EMI types | Personal loan, home loan, vehicle, consumer durables |
| Features | 28 variables — income, loan amount, tenure, CIBIL, employment type, etc. |
| Target variable | Default (1) / No Default (0) |
| Class distribution | 82% No Default / 18% Default (imbalanced) |
| Imbalance handling | SMOTE oversampling + class_weight balancing |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10 |
| Data processing | Pandas, NumPy |
| ML Models | Random Forest, XGBoost |
| Hyperparameter tuning | RandomizedSearchCV (50 iterations) |
| Imbalance handling | SMOTE (imblearn) |
| Experiment tracking | MLflow 2.5 |
| Deployment | Hugging Face Spaces (Gradio) |
| Evaluation | AUC-ROC, KS Statistic, Precision, Recall, F1-Score, Gini Coefficient |

---

## Workflow

```
Raw 400K+ Financial Records
        ↓
Data Cleaning & Preprocessing
  ├── Missing value imputation (median for numeric, mode for categorical)
  ├── Outlier treatment via IQR winsorization (income, loan amount)
  ├── Label encoding for employment type, loan category
  └── Standard scaling for numerical features
        ↓
Feature Engineering
  ├── EMI-to-income ratio       → monthly_emi / monthly_income
  ├── Debt burden index         → total_outstanding / annual_income
  ├── Loan-to-value ratio       → loan_amount / asset_value
  ├── Repayment capacity score  → income_after_emi / min_expenses
  └── Credit utilization flag   → existing_loans > 2
        ↓
Class Imbalance Handling
  └── SMOTE → balanced training set (50:50)
        ↓
Model Training
  ├── Random Forest (n_estimators=200, max_depth=15)  — baseline
  └── XGBoost (n_estimators=300, learning_rate=0.05)  — optimized
        ↓
Hyperparameter Tuning
  └── RandomizedSearchCV (50 iterations, 5-fold CV)
        ↓
MLflow Experiment Tracking
  └── Logged: params, AUC, KS, F1, model artifacts (32 experiments)
        ↓
Deployment
  └── Hugging Face Spaces — real-time default probability scoring
```

---

## Results

| Model | AUC-ROC | KS Statistic | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| Random Forest (baseline) | 0.87 | 0.61 | 0.79 | 0.74 | 0.76 |
| Random Forest (tuned) | 0.89 | 0.65 | 0.82 | 0.77 | 0.79 |
| XGBoost (baseline) | 0.90 | 0.67 | 0.83 | 0.79 | 0.81 |
| **XGBoost (tuned)** | **🏆 0.93** | **🏆 0.72** | **0.87** | **0.84** | **0.85** |

**Gini Coefficient (XGBoost tuned): 0.86** — indicating strong discriminating power for credit risk applications

---

## Feature Importance (Top 6)

| Rank | Feature | Importance Score |
|---|---|---|
| 1 | EMI-to-income ratio | 0.187 |
| 2 | CIBIL score | 0.163 |
| 3 | Debt burden index | 0.141 |
| 4 | Loan tenure | 0.112 |
| 5 | Number of existing loans | 0.098 |
| 6 | Employment type | 0.079 |

---

## MLflow Experiment Tracking

32 experiments tracked across model types and hyperparameter combinations:

```bash
# View all experiments locally
mlflow ui
# Opens at http://localhost:5000
```

Tracked per experiment:
- All hyperparameter values (n_estimators, max_depth, learning_rate, subsample)
- AUC-ROC, KS Statistic, F1-Score, Precision, Recall
- Model artifacts (pickled .pkl files)
- Training duration and dataset hash

---

## Live Demo

🔗 **[Try EMIPredict AI on Hugging Face Spaces](https://huggingface.co/spaces/Elansurya/EMI-Predict-AI)**

Input a borrower profile and get:
- Default probability score (0–100%)
- Risk category: 🟢 Low / 🟡 Medium / 🔴 High
- Top 3 risk factors driving the prediction

> **Screenshots:**
> Add these to a `/screenshots` folder:
> 1. `app_interface.png` — Hugging Face Spaces input form
> 2. `prediction_output.png` — Risk score output screen
> 3. `feature_importance.png` — Feature importance bar chart
> 4. `mlflow_ui.png` — MLflow experiment comparison view

![App Interface](screenshots/app_interface.png)
![Prediction Output](screenshots/prediction_output.png)
![Feature Importance](screenshots/feature_importance.png)

---

## Business Impact

- **NPA reduction:** Pre-screens high-risk borrowers before disbursement — directly reduces default exposure for NBFCs and banks
- **KS Statistic of 0.72** exceeds the industry benchmark of 0.40 for credit scorecards — indicating production-grade discriminating power
- **Regulatory alignment:** MLflow experiment tracking creates auditable, reproducible model history — critical for RBI model governance requirements
- **Applicable to:** Loan origination systems, credit underwriting automation, BNPL risk scoring, co-lending platforms

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Elansurya/EMI-Predict-AI.git
cd EMI-Predict-AI

# Install dependencies
pip install -r requirements.txt

# Start MLflow tracking server
mlflow ui

# Run Hugging Face app locally
python app.py
```

---

## Project Structure

```
EMI-Predict-AI/
├── data_preprocessing.ipynb    # Cleaning, encoding, feature engineering
├── model_training.ipynb        # RF + XGBoost training + tuning
├── evaluation.ipynb            # Metrics, ROC curves, KS plots
├── app.py                      # Hugging Face Spaces Gradio app
├── mlflow_experiments/         # Saved experiment runs
├── models/                     # Pickled best model
│   └── xgb_tuned_v3.pkl
├── requirements.txt
├── screenshots/
└── README.md
```

---

## Requirements

```
xgboost==1.7.6
scikit-learn==1.3.0
imbalanced-learn==0.11.0
mlflow==2.5.0
pandas==2.0.3
numpy==1.24.3
gradio==3.40.1
matplotlib==3.7.2
seaborn==0.12.2
shap==0.42.1
```

---

## Author

**Elansurya K** — Aspiring Data Scientist | ML · Python · SQL · MLflow

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/elansurya-karthikeyan-3b6636380)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat-square&logo=github)](https://github.com/Elansurya)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Live%20Demo-yellow?style=flat-square)](https://huggingface.co/spaces/Elansurya/EMI-Predict-AI)
