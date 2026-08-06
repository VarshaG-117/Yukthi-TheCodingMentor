"""
Builds features from raw data collected in data/raw/.

Combines LeetCode solved-problems + Codeforces submissions into ONE
chronological timeline of attempts, runs them through the Elo module
to compute per-topic skill ratings, and produces two outputs:

1. data/processed/topic_ratings_<user>.json
   -> current skill rating per DSA topic (used at serving time to
      pick what to recommend NEXT)

2. data/processed/training_attempts_<user>.csv
   -> one row per historical attempt: the user's rating BEFORE that
      attempt, the problem's difficulty, and whether they solved it.
      This becomes training data for Problem B (the success classifier).

IMPORTANT LIMITATION (be upfront about this in your report):
LeetCode's public API only exposes a user's 20 MOST RECENT solved
problems, with no failed-attempt history. Codeforces gives full
history including failures. So LeetCode data is a small recent
slice, Codeforces data is much richer. This is a real constraint
of the free public APIs, not a bug in our code.

Usage:
    python src/features/build_features.py --leetcode varsha0101 --codeforces Varsha-117
"""

import argparse
import json
import os
import glob
import csv
from datetime import datetime

from elo import TopicRatings

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

# Rough numeric mapping for LeetCode's Easy/Medium/Hard so we can mix
# it with Codeforces' numeric ratings on the same Elo scale.
# These are approximate - tuned to roughly line up with Codeforces bands.
LEETCODE_DIFFICULTY_MAP = {
    "Easy": 900,
    "Medium": 1400,
    "Hard": 1900,
}


def load_json(filename):
    path = os.path.join(RAW_DIR, filename)
    with open(path) as f:
        return json.load(f)


def find_latest_file(pattern):
    """Finds the most recently saved raw file matching a glob pattern
    (user files have a timestamp in their name, so 'latest' = newest run)."""
    matches = glob.glob(os.path.join(RAW_DIR, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def build_leetcode_attempts(username):
    """
    Returns a list of unified attempt dicts from LeetCode data:
    {timestamp, topics, difficulty, solved}
    Looks up each solved problem's difficulty/tags from the catalog.
    """
    user_file = find_latest_file(f"leetcode_user_{username}_*.json")
    if not user_file:
        print(f"No LeetCode data found for {username}, skipping.")
        return []

    with open(user_file) as f:
        user_data = json.load(f)

    catalog = load_json("leetcode_catalog.json")
    catalog_by_slug = {p["slug"]: p for p in catalog}

    attempts = []
    for solved in user_data["recent_solved"]:
        problem = catalog_by_slug.get(solved["slug"])
        if not problem:
            continue  # problem not in our catalog snapshot, skip it

        attempts.append({
            "timestamp": int(solved["timestamp"]),
            "platform": "leetcode",
            "problem_id": solved["slug"],
            "topics": problem["tags"],
            "difficulty": LEETCODE_DIFFICULTY_MAP[problem["difficulty"]],
            "solved": True,  # LeetCode's recent_solved list is ALWAYS solved by definition
        })
    return attempts


def build_codeforces_attempts(handle):
    """
    Returns a list of unified attempt dicts from Codeforces data.
    Codeforces gives us BOTH solved and failed attempts, which is
    richer training signal than LeetCode's solved-only list.
    """
    user_file = find_latest_file(f"codeforces_user_{handle}_*.json")
    if not user_file:
        print(f"No Codeforces data found for {handle}, skipping.")
        return []

    with open(user_file) as f:
        user_data = json.load(f)

    attempts = []
    for attempt in user_data["all_attempts"]:
        if attempt["rating"] is None:
            continue  # unrated problem, can't use it for difficulty matching

        attempts.append({
            "timestamp": int(attempt["timestamp"]),
            "platform": "codeforces",
            "problem_id": attempt["problem_id"],
            "topics": attempt["tags"],
            "difficulty": attempt["rating"],
            "solved": attempt["verdict"] == "OK",
        })
    return attempts


def build_features(leetcode_username=None, codeforces_handle=None):
    all_attempts = []

    if leetcode_username:
        all_attempts.extend(build_leetcode_attempts(leetcode_username))

    if codeforces_handle:
        all_attempts.extend(build_codeforces_attempts(codeforces_handle))

    if not all_attempts:
        print("No attempts found - check your raw data files exist.")
        return

    # CRITICAL: sort chronologically, oldest first, so Elo updates
    # reflect the user's actual skill progression over time.
    all_attempts.sort(key=lambda a: a["timestamp"])

    tr = TopicRatings()
    training_rows = []

    for attempt in all_attempts:
        # Snapshot each topic's rating BEFORE this attempt updates it -
        # that's the feature the classifier will see (we don't want to
        # leak the outcome of THIS attempt into its own feature).
        pre_attempt_ratings = {t: tr.get_rating(t) for t in attempt["topics"]}
        avg_pre_rating = sum(pre_attempt_ratings.values()) / len(pre_attempt_ratings) if pre_attempt_ratings else tr.default_rating

        training_rows.append({
            "timestamp": attempt["timestamp"],
            "platform": attempt["platform"],
            "problem_id": attempt["problem_id"],
            "topics": "|".join(attempt["topics"]),
            "difficulty": attempt["difficulty"],
            "user_rating_before": round(avg_pre_rating, 1),
            "solved": int(attempt["solved"]),
        })

        tr.record_attempt(attempt["topics"], attempt["difficulty"], attempt["solved"])

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Output 1: final topic ratings snapshot (for serving/recommending)
    user_tag = leetcode_username or codeforces_handle or "user"
    ratings_path = os.path.join(PROCESSED_DIR, f"topic_ratings_{user_tag}.json")
    with open(ratings_path, "w") as f:
        json.dump(tr.as_dict(), f, indent=2)
    print(f"Saved topic ratings -> {ratings_path}")
    print(json.dumps(tr.as_dict(), indent=2))

    # Output 2: training table (for the classifier)
    training_path = os.path.join(PROCESSED_DIR, f"training_attempts_{user_tag}.csv")
    with open(training_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=training_rows[0].keys())
        writer.writeheader()
        writer.writerows(training_rows)
    print(f"\nSaved {len(training_rows)} training rows -> {training_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leetcode", help="LeetCode username")
    parser.add_argument("--codeforces", help="Codeforces handle")
    args = parser.parse_args()

    build_features(leetcode_username=args.leetcode, codeforces_handle=args.codeforces)