"""
Lightweight data drift detection.

Compares the statistical shape of NEW training data against a saved
baseline snapshot. If the underlying population of users/problems
shifts meaningfully (e.g. average difficulty jumps, class balance
swings), that's a signal the model may need attention even before
its accuracy visibly drops.

Usage:
    python src/monitoring/check_drift.py
"""

import pandas as pd
import json
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

DRIFT_THRESHOLD_PCT = 15.0


def compute_stats(df):
    return {
        "mean_difficulty": float(df["difficulty"].mean()),
        "mean_user_rating": float(df["user_rating_before"].mean()),
        "solve_rate_pct": float(df["solved"].mean() * 100),
        "row_count": int(len(df)),
    }


def pct_change(old, new):
    if old == 0:
        return 0.0
    return abs(new - old) / abs(old) * 100


def check_drift():
    data_path = os.path.join(PROCESSED_DIR, "community_training_data.csv")
    baseline_path = os.path.join(PROCESSED_DIR, "drift_baseline.json")

    df = pd.read_csv(data_path)
    current_stats = compute_stats(df)

    print("Current data stats:")
    for key, value in current_stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

    if not os.path.exists(baseline_path):
        print("\nNo baseline found - saving current stats AS the baseline.")
        with open(baseline_path, "w") as f:
            json.dump(current_stats, f, indent=2)
        print("Baseline saved. Nothing to compare yet.")
        return

    with open(baseline_path) as f:
        baseline_stats = json.load(f)

    print("\nBaseline stats:")
    for key, value in baseline_stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n=== Drift check ===")
    drifted = []
    for key in ["mean_difficulty", "mean_user_rating", "solve_rate_pct"]:
        change = pct_change(baseline_stats[key], current_stats[key])
        status = "DRIFT" if change > DRIFT_THRESHOLD_PCT else "ok"
        print(f"  {key}: {change:.1f}% change [{status}]")
        if status == "DRIFT":
            drifted.append(key)

    if drifted:
        print(f"\nWARNING: drift detected in: {', '.join(drifted)}")
        print("Consider investigating data collection or retraining with fresh data.")
    else:
        print("\nNo significant drift detected.")


if __name__ == "__main__":
    check_drift()
