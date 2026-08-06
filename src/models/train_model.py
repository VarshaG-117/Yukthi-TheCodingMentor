"""
Trains Problem B: a classifier predicting the probability a user
solves a given problem, based on their skill rating and the problem's
difficulty.

Core feature: rating_gap = problem_difficulty - user_rating_before
This is the SAME quantity our Elo formula uses to compute expected
score - so this model should learn a similar relationship, which is
a nice sanity check that our Elo assumptions and the real data agree.

Usage:
    python src/models/train_model.py
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import lightgbm as lgb

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


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
    print("Loading training data...")
    df = load_training_data()
    print(f"Loaded {len(df)} rows.")

    X, y, feature_cols = engineer_features(df)

    # Stratified split keeps the solved/failed ratio consistent between
    # train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
    )
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

    print("\n=== Feature importances ===")
    importances = dict(zip(feature_cols, model.feature_importances_))
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp}")

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

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "success_classifier.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved -> {model_path}")

    return model, {"auc": auc, "accuracy": acc, "log_loss": ll}


if __name__ == "__main__":
    train_and_evaluate()