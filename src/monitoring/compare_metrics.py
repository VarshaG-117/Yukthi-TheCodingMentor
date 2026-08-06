"""
Promotion gate for continuous training.

After a scheduled retrain, we DON'T want to blindly replace the
production model - a bad batch of data or an unlucky random seed
could make the new model worse, and silently deploying a worse
model is a real MLOps failure mode.

This script compares the freshly trained model's metrics
(models/latest_run_metrics.json) against the last model we
explicitly approved (models/approved_metrics.json) and prints a
clear PROMOTE or REJECT decision.

First run ever: if there's no approved_metrics.json yet, this run's
metrics automatically become the baseline (nothing to compare against).

Usage:
    python src/monitoring/compare_metrics.py
Exit code 0 = promote (or first run), exit code 1 = reject.
"""

import json
import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

AUC_REGRESSION_TOLERANCE = 0.01


def compare_metrics():
    latest_path = os.path.join(MODELS_DIR, "latest_run_metrics.json")
    approved_path = os.path.join(MODELS_DIR, "approved_metrics.json")

    if not os.path.exists(latest_path):
        print("No latest_run_metrics.json found - run train_model.py first.")
        sys.exit(1)

    with open(latest_path) as f:
        latest = json.load(f)

    if not os.path.exists(approved_path):
        print("No previously approved model found - this run becomes the baseline.")
        with open(approved_path, "w") as f:
            json.dump(latest, f, indent=2)
        print(f"Approved. AUC: {latest['auc']:.4f}")
        sys.exit(0)

    with open(approved_path) as f:
        approved = json.load(f)

    print(f"Approved (current) model AUC: {approved['auc']:.4f}")
    print(f"New (candidate) model AUC:    {latest['auc']:.4f}")

    diff = latest["auc"] - approved["auc"]
    print(f"Difference: {diff:+.4f}")

    if diff >= -AUC_REGRESSION_TOLERANCE:
        print(f"\nPROMOTE: new model is within tolerance ({AUC_REGRESSION_TOLERANCE}) "
              f"or better than the approved model.")
        with open(approved_path, "w") as f:
            json.dump(latest, f, indent=2)
        sys.exit(0)
    else:
        print(f"\nREJECT: new model AUC regressed by more than "
              f"{AUC_REGRESSION_TOLERANCE} - keeping the currently approved model.")
        sys.exit(1)


if __name__ == "__main__":
    compare_metrics()
