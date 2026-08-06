"""
Builds the ACTUAL training dataset for Problem B (the success classifier)
from data/raw/community_codeforces.json - hundreds of real users' real
submission histories.

For EACH user, we track their OWN separate per-topic Elo ratings (skill
never leaks between users - that would be nonsensical). We run each
user's attempts through Elo in chronological order, same as we did for
you individually in build_features.py, and stack every user's rows into
one big table.

TWO DATA-QUALITY FIXES applied here (v2):

1. Seed each user's starting Elo rating with their REAL Codeforces
   rating (which we already collected) instead of a flat 1200 default
   for everyone. Without this, a genuinely 1900-rated user looks like
   a beginner for their first several attempts, injecting noise.

2. Deduplicate to the FIRST attempt per (user, problem) pair. Codeforces
   submission history often has several WRONG_ANSWER submissions
   followed by an eventual OK on the SAME problem - these aren't
   independent trials, they're one struggle counted multiple times,
   which biases the training data toward "eventually solved everything."

Usage:
    python src/features/build_community_features.py
"""

import json
import os
import csv

from elo import TopicRatings

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def deduplicate_to_first_attempt(attempts):
    """
    Keeps only the FIRST submission per problem (by timestamp).
    This represents the genuine 'first try' difficulty signal instead
    of counting repeated struggles on the same problem as separate
    independent examples.
    """
    first_seen = {}
    for attempt in attempts:
        pid = attempt["problem_id"]
        if pid not in first_seen or attempt["timestamp"] < first_seen[pid]["timestamp"]:
            first_seen[pid] = attempt
    return list(first_seen.values())


def build_community_training_data():
    community_path = os.path.join(RAW_DIR, "community_codeforces.json")
    with open(community_path) as f:
        community_data = json.load(f)

    all_rows = []

    for user in community_data:
        handle = user["handle"]
        cf_rating = user["rating"]  # their REAL Codeforces rating - use as starting point

        deduped_attempts = deduplicate_to_first_attempt(user["attempts"])
        attempts = sorted(deduped_attempts, key=lambda a: a["timestamp"])

        # Seed with their real rating instead of a flat 1200 for everyone
        tr = TopicRatings(default_rating=cf_rating)

        for attempt in attempts:
            topics = attempt["tags"]
            if not topics:
                continue  # skip problems with no tags, nothing to update

            pre_ratings = {t: tr.get_rating(t) for t in topics}
            avg_pre_rating = sum(pre_ratings.values()) / len(pre_ratings)
            solved = attempt["verdict"] == "OK"

            all_rows.append({
                "handle": handle,
                "timestamp": attempt["timestamp"],
                "problem_id": attempt["problem_id"],
                "topics": "|".join(topics),
                "difficulty": attempt["rating"],
                "user_rating_before": round(avg_pre_rating, 1),
                "solved": int(solved),
            })

            tr.record_attempt(topics, attempt["rating"], solved)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "community_training_data.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Built {len(all_rows)} training rows from {len(community_data)} users -> {out_path}")

    solved_count = sum(r["solved"] for r in all_rows)
    print(f"Class balance: {solved_count} solved ({solved_count/len(all_rows)*100:.1f}%), "
          f"{len(all_rows) - solved_count} failed ({(1 - solved_count/len(all_rows))*100:.1f}%)")


if __name__ == "__main__":
    build_community_training_data()