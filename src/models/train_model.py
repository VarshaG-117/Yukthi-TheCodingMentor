"""
Trains Problem B: a classifier predicting the probability a user
solves a given problem, based on their skill rating and the problem's
difficulty.

Core feature: rating_gap = problem_difficulty - user_rating_before
This is the SAME quantity our Elo formula uses to compute expected
score - so this model should learn a similar relationship, which is
a nice sanity check that our Elo assumptions and the real data agree.

MLflow tracks every run: parameters, metrics, feature importances, and
the trained model itself - so we can compare runs over time (e.g.
after retraining on fresh data) instead of just overwriting one file.

Usage:
    python src/models/train_model.py
    mlflow ui    # then open http://localhost:5000 to see the dashboard
"""

import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import lightgbm as lgb

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

MLFLOW_EXPERIMENT_NAME = "coding-mentor-success-classifier"

# Model hyperparameters - defined up top so they're easy to tweak and
# so we log the EXACT values used for each run.
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "random_state": 42,
}


def load_training_data():
    path = os.path.join(PROCESSED_DIR, "community_training_data.csv")
    df = pd.read_csv(path)
    return df


def engineer_features(df):
    """
    Builds the feature matrix X and label vector y.
    rating_gap is the key feature - how much harder/easier the
    problem is relative to the user's current topic skill.
    """
    df = df.copy()
    df["rating_gap"] = df["difficulty"] - df["user_rating_before"]

    feature_cols = ["difficulty", "user_rating_before", "rating_gap"]
    X = df[feature_cols]
    y = df["solved"]
    return X, y, feature_cols


def train_and_evaluate():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        print("Loading training data...")
        df = load_training_data()
        print(f"Loaded {len(df)} rows.")
        mlflow.log_param("training_rows", len(df))

        X, y, feature_cols = engineer_features(df)
        mlflow.log_param("feature_columns", ",".join(feature_cols))

        # Stratified split keeps the solved/failed ratio consistent between
        # train and test sets.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        # Log every hyperparameter so this exact run is reproducible
        for param_name, param_value in MODEL_PARAMS.items():
            mlflow.log_param(param_name, param_value)

        model = lgb.LGBMClassifier(**MODEL_PARAMS, verbose=-1)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_pred_proba)

        print("\n=== Evaluation ===")
        print(f"AUC:      {auc:.4f}   (0.5 = random guessing, 1.0 = perfect)")
        print(f"Accuracy: {acc:.4f}")
        print(f"LogLoss:  {ll:.4f}   (lower is better)")

        mlflow.log_metric("auc", auc)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("log_loss", ll)

        print("\n=== Feature importances ===")
        importances = dict(zip(feature_cols, model.feature_importances_))
        for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"  {feat}: {imp}")
            mlflow.log_metric(f"importance_{feat}", float(imp))

        # Sanity check: does the model roughly match the theoretical Elo curve?
        print("\n=== Sanity check: predicted solve probability vs rating_gap ===")
        print("(should DECREASE as rating_gap increases - harder problem relative to skill)")
        for gap in [-400, -200, 0, 200, 400]:
            sample = pd.DataFrame([{
                "difficulty": 1200 + gap,
                "user_rating_before": 1200,
                "rating_gap": gap,
            }])
            pred = model.predict_proba(sample)[0][1]
            print(f"  rating_gap={gap:+d} -> predicted solve probability: {pred:.3f}")
            mlflow.log_metric(f"sanity_check_gap_{gap}", float(pred))

        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, "success_classifier.pkl")
        joblib.dump(model, model_path)
        print(f"\nModel saved -> {model_path}")

        # Log the model itself as an MLflow artifact - this is what lets
        # you (or the Model Registry, later) pull back this EXACT trained
        # model from any past run.
        mlflow.lightgbm.log_model(model, "model")

        run = mlflow.active_run()
        print(f"\nMLflow run logged: {run.info.run_id}")
        print("Run 'mlflow ui' in your terminal and open http://localhost:5000 to view it.")

        return model, {"auc": auc, "accuracy": acc, "log_loss": ll}


if __name__ == "__main__":
    train_and_evaluate()