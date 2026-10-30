
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student-mat.csv"

MODEL_PATH = BASE_DIR / "model.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"
PREDICTIONS_PATH = BASE_DIR / "test_predictions.csv"
IMPORTANCE_PATH = BASE_DIR / "feature_importance.csv"

FEATURES = [
    "age",
    "studytime",
    "failures",
    "schoolsup",
    "internet",
    "absences"
]

TARGET = "G3"


# ==========================================
# 2. LOAD AND VALIDATE DATASET
# ==========================================

def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "student-mat.csv not found. "
            "Place the dataset inside D:\\StudentPerformanceML"
        )

    df = pd.read_csv(DATA_PATH, sep=";")

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in dataset: {missing_columns}"
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    print("\nDataset loaded successfully!")
    print("Dataset shape:", df.shape)

    return df


# ==========================================
# 3. DATA PREPROCESSING
# ==========================================

def preprocess_data(df):

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Convert categorical yes/no values into numbers
    for column in ["schoolsup", "internet"]:

        X[column] = X[column].map({
            "yes": 1,
            "no": 0
        })

    # Convert columns to numeric values
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    # Handle missing values
    X = X.fillna(X.median())
    y = y.fillna(y.median())

    return X, y


# ==========================================
# 4. TRAIN AND EVALUATE MODELS
# ==========================================

def train_models(X_train, X_test, y_train, y_test):

    models = {

        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )

    }

    results = {}
    predictions = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)

        rmse = np.sqrt(
            mean_squared_error(y_test, y_pred)
        )

        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "MAE": round(float(mae), 3),
            "RMSE": round(float(rmse), 3),
            "R2": round(float(r2), 3)
        }

        predictions[name] = y_pred

        print("MAE:", round(mae, 3))
        print("RMSE:", round(rmse, 3))
        print("R2 Score:", round(r2, 3))

    return models, results, predictions


# ==========================================
# 5. SAVE TRAINED MODEL
# ==========================================

def save_model(model):

    joblib.dump(model, MODEL_PATH)

    print("\nRandom Forest model saved successfully!")


# ==========================================
# 6. SAVE EVALUATION METRICS
# ==========================================

def save_metrics(results):

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(results, file, indent=4)

    print("Evaluation metrics saved!")


# ==========================================
# 7. SAVE ACTUAL VS PREDICTED RESULTS
# ==========================================

def save_predictions(y_test, predictions):

    comparison = pd.DataFrame({

        "Actual Grade": y_test.to_numpy(),

        "Linear Regression": predictions[
            "Linear Regression"
        ],

        "Random Forest": predictions[
            "Random Forest"
        ]

    })

    comparison.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    print("Prediction comparison saved!")


# ==========================================
# 8. SAVE FEATURE IMPORTANCE
# ==========================================

def save_feature_importance(model):

    importance = pd.DataFrame({

        "Feature": FEATURES,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    importance.to_csv(
        IMPORTANCE_PATH,
        index=False
    )

    print("\nFeature Importance:")
    print(importance.to_string(index=False))

    print("\nFeature importance saved!")


# ==========================================
# 9. MAIN FUNCTION
# ==========================================

def main():

    print("=" * 50)
    print("STUDENT PERFORMANCE PREDICTION")
    print("MACHINE LEARNING MODEL TRAINING")
    print("=" * 50)

    # Load dataset
    df = load_dataset()

    # Preprocess dataset
    X, y = preprocess_data(df)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    # Train models
    models, results, predictions = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Save Random Forest model
    random_forest_model = models["Random Forest"]

    save_model(random_forest_model)

    # Save evaluation results
    save_metrics(results)

    # Save predictions
    save_predictions(y_test, predictions)

    # Save feature importance
    save_feature_importance(random_forest_model)

    # Display final comparison
    print("\n" + "=" * 50)
    print("FINAL MODEL COMPARISON")
    print("=" * 50)

    results_df = pd.DataFrame(results).T

    print(results_df)

    print("\nAll training operations completed!")
    print("\nGenerated files:")

    print("1. model.joblib")
    print("2. metrics.json")
    print("3. test_predictions.csv")
    print("4. feature_importance.csv")


if __name__ == "__main__":
    main()
