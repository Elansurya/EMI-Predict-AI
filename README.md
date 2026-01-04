💰 EMIPredict AI
AI-Powered EMI Eligibility & Credit Risk Assessment System

A production-ready FinTech Machine Learning application that predicts loan approval and affordable EMI using real-world financial data.

🚀 Project Overview

EMIPredict AI is an end-to-end Machine Learning system designed to simulate how banks, NBFCs, and FinTech companies assess customer loan eligibility and EMI affordability.

The project covers the complete ML lifecycle:

Data preprocessing

Feature engineering

Model training & evaluation

Model deployment using Streamlit

Robust error handling & production-safe UI

🎯 Business Problem

Financial institutions must quickly decide:

Is a customer eligible for a loan?

What is the maximum safe EMI they can afford?

Manual evaluation is slow and inconsistent.
EMIPredict AI automates this decision using Machine Learning.

🧠 Solution Approach

This project uses two ML models:

1️⃣ Classification Model

Predicts:

Loan Approved (Yes / No)

2️⃣ Regression Model

Predicts:

Maximum Affordable EMI Amount

Both models are trained on processed financial data and deployed in a real-time Streamlit web application.

🏗️ Project Architecture
EMIPredict AI/
│
├── app/
│   ├── app.py                 # Streamlit UI (production-safe)
│   └── models/
│       ├── best_classification_model.pkl
│       ├── best_regression_model.pkl
│       ├── scaler.pkl
│       ├── label_encoders.pkl
│       ├── performance_metrics.json
│       └── performance_summary.csv
│
├── data/
│   ├── emi_prediction_dataset.csv
│   ├── engineered_data.csv
│   └── processed_data.csv     # Ignored in GitHub (size > 25MB)
│
├── data_preprocessing/
│   └── data_preprocessing.py
│
├── feature_engineering/
│   └── feature_engineering.py
│
├── eda_analysis/
│   ├── eda_analysis.py
│   └── eda_analysis.ipynb
│
├── README.md
└── .gitignore

📊 Exploratory Data Analysis (EDA)

EDA was performed to understand:

Feature distributions

Missing values

Credit score patterns

Loan approval trends

Outliers and correlations

⚠️ Dataset Note

Due to GitHub file size limits (>25MB), the full dataset is not included.
python eda_analysis/eda_analysis.py

🤖 Machine Learning Models
Algorithms Used
Random Forest Classifier

Random Forest Regressor

Feature Scaling (StandardScaler)

Label Encoding for categorical features

Model Outputs
Classification Metrics: Accuracy, Precision, Recall, F1-Score

Regression Metrics: RMSE, MAE, R² Score

🖥️ Web Application (Streamlit)
Key UI Features

FinTech-style dashboard

Real-time loan approval prediction

EMI estimation

Interactive EDA charts

Model performance visualization

Robust error handling

Feature mismatch protection
🛠️ Technology Stack

Language: Python

Data: Pandas, NumPy

ML: Scikit-learn

Visualization: Matplotlib, Seaborn, Plotly

Web App: Streamlit

Version Control: Git & GitHub
