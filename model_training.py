import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# FIX IMPORT PATH (VERY IMPORTANT)
# =====================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from data_preprocessing.data_preprocessing import DataPreprocessor

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, mean_squared_error,
    mean_absolute_error, r2_score
)

# =====================================================
# MODEL TRAINER CLASS
# =====================================================
class MLModelTrainer:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        os.makedirs("models", exist_ok=True)

    # -------------------------------------------------
    # CLASSIFICATION MODELS
    # -------------------------------------------------
    def train_classification_models(self):

        models = {
            "Logistic_Regression": LogisticRegression(max_iter=1000),
            "Random_Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Decision_Tree": DecisionTreeClassifier(random_state=42),
            "Gradient_Boosting": GradientBoostingClassifier(random_state=42),
            "SVC": SVC(probability=True, random_state=42)
        }

        results = {}

        print("\n=== CLASSIFICATION TRAINING ===")

        for name, model in models.items():
            print(f"\nTraining {name}...")

            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            results[name] = {
                "model": model,
                "accuracy": accuracy_score(self.y_test, y_pred),
                "precision": precision_score(self.y_test, y_pred, average="weighted"),
                "recall": recall_score(self.y_test, y_pred, average="weighted"),
                "f1": f1_score(self.y_test, y_pred, average="weighted")
            }

            print(
                f"✓ {name} | "
                f"Accuracy={results[name]['accuracy']:.4f} | "
                f"F1={results[name]['f1']:.4f}"
            )

        best_model = max(results, key=lambda x: results[x]["f1"])
        print(f"\n🏆 Best Classification Model: {best_model}")

        return results, best_model

    # -------------------------------------------------
    # REGRESSION MODELS
    # -------------------------------------------------
    def train_regression_models(self):

        models = {
            "Linear_Regression": LinearRegression(),
            "Random_Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "Decision_Tree": DecisionTreeRegressor(random_state=42),
            "Gradient_Boosting": GradientBoostingRegressor(random_state=42),
            "SVR": SVR()
        }

        results = {}

        print("\n=== REGRESSION TRAINING ===")

        for name, model in models.items():
            print(f"\nTraining {name}...")

            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            results[name] = {
                "model": model,
                "rmse": np.sqrt(mean_squared_error(self.y_test, y_pred)),
                "mae": mean_absolute_error(self.y_test, y_pred),
                "r2": r2_score(self.y_test, y_pred)
            }

            print(
                f"✓ {name} | "
                f"RMSE={results[name]['rmse']:.2f} | "
                f"R²={results[name]['r2']:.4f}"
            )

        best_model = min(results, key=lambda x: results[x]["rmse"])
        print(f"\n🏆 Best Regression Model: {best_model}")

        return results, best_model

    # -------------------------------------------------
    # SAVE MODELS
    # -------------------------------------------------
    def save_models(self, class_results, reg_results, best_class, best_reg):

        joblib.dump(
            class_results[best_class]["model"],
            "models/best_classification_model.pkl"
        )

        joblib.dump(
            reg_results[best_reg]["model"],
            "models/best_regression_model.pkl"
        )

        print("\n✅ Best models saved successfully in /models folder")


# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":

    DATA_PATH = r"C:\project\EMIPredict AI\data\processed_data.csv"

    print("\nLoading & preprocessing data...")
    preprocessor = DataPreprocessor(DATA_PATH)
    data_splits, _, _, _ = preprocessor.preprocess_pipeline()

    print("\nAvailable keys from DataPreprocessor:")
    print(data_splits.keys())

    # ---------------- CLASSIFICATION ----------------
    trainer = MLModelTrainer(
        data_splits["X_train"],
        data_splits["X_test"],
        data_splits["y_class_train"],
        data_splits["y_class_test"]
    )

    class_results, best_class = trainer.train_classification_models()

    # ---------------- REGRESSION ----------------
    reg_results, best_reg = trainer.train_regression_models()

    # ---------------- SAVE MODELS ----------------
    trainer.save_models(class_results, reg_results, best_class, best_reg)

    print("\n🎯 MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
