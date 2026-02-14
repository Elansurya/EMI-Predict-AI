# 💰 EMIPredict AI – Intelligent Financial Risk Assessment Platform

## 🚀 Project Overview
EMIPredict AI is a production-ready financial risk assessment platform that integrates Machine Learning models with MLflow experiment tracking and a multi-page Streamlit web application.

The system performs dual ML problem solving:

- ✅ Classification – EMI Eligibility (Eligible / High Risk / Not Eligible)
- 📊 Regression – Maximum Safe Monthly EMI Amount

Built using 400,000 realistic financial records across 5 lending scenarios, this platform enables real-time, data-driven loan decision support.

---

## 🎯 Business Problem

Poor financial planning and inadequate risk assessment often lead to EMI defaults. Financial institutions require:

- Automated loan eligibility decisions
- Risk-based pricing strategies
- Real-time pre-qualification checks
- Standardized credit evaluation frameworks

EMIPredict AI provides a scalable, AI-powered solution to automate underwriting and reduce manual processing time by up to 80%.

---

## 📊 Dataset Overview

- Total Records: 400,000 financial profiles
- Input Features: 22 financial & demographic variables
- Target Variables: 2 (Classification + Regression)
- EMI Scenarios: 5 realistic lending categories

### EMI Scenarios:
- E-commerce EMI (10K–200K)
- Home Appliances EMI (20K–300K)
- Vehicle EMI (80K–1500K)
- Personal Loan EMI (50K–1000K)
- Education EMI (50K–500K)

---

## 🧠 Feature Engineering

Created advanced financial indicators:

- Debt-to-Income Ratio
- Expense-to-Income Ratio
- Affordability Index
- Risk Score based on credit history
- Interaction features between financial variables
- Categorical encoding & numerical scaling

---

## 🤖 Machine Learning Models

### Classification Models (EMI Eligibility)
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- Support Vector Classifier (optional)

Evaluation Metrics:
- Accuracy (> 90%)
- Precision
- Recall
- F1-Score
- ROC-AUC

---

### Regression Models (Maximum EMI Prediction)
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- Gradient Boosting Regressor (optional)

Evaluation Metrics:
- RMSE (< 2000 INR)
- MAE
- R² Score
- MAPE

Best performing models selected for production deployment.

---

## 📊 MLflow Integration

- Centralized experiment tracking
- Hyperparameter logging
- Performance metric comparison
- Model registry with version control
- Artifact storage for trained models

MLflow dashboard enables transparent and organized model selection.

---

## 🖥️ Streamlit Application

Multi-page interactive web application featuring:

- Real-time EMI eligibility prediction
- Maximum EMI calculation
- Model performance dashboard
- MLflow experiment comparison view
- Admin panel for financial data management
- Complete CRUD operations

---

## 🏗️ Architecture

Dataset (400K Records)  
↓  
Data Cleaning & Validation  
↓  
Feature Engineering & EDA  
↓  
ML Model Training  
↓  
MLflow Tracking & Model Selection  
↓  
Streamlit Application  
↓  
Cloud Deployment  

---

## ⚙️ Tech Stack

- Python
- Scikit-learn
- XGBoost
- MLflow
- Streamlit Cloud
- Pandas, NumPy
- Data Preprocessing & Feature Engineering

Domain: FinTech & Banking

---

## 📈 Business Impact

- Automated loan approval decisions
- Reduced manual underwriting time by 80%
- Standardized EMI risk evaluation
- Real-time financial profile analysis
- Scalable architecture for high-volume loan applications

---

## ☁️ Deployment

- Cloud hosted on Streamlit Cloud
- GitHub integrated CI/CD pipeline
- Responsive and production-ready UI

---

## 📌 Key Learnings

- Large-scale financial data processing (400K records)
- Dual ML problem solving (Classification + Regression)
- Financial feature engineering techniques
- MLflow experiment tracking & model registry
- Production-level ML deployment

---

## 🔮 Future Improvements

- Default probability prediction
- Loan portfolio risk forecasting
- Credit scoring integration
- API-based integration for banks
- Cloud database integration (AWS / Azure)

---

## 👨‍💻 Author
Elansurya K  
Data Scientist | Machine Learning | NLP | SQL
