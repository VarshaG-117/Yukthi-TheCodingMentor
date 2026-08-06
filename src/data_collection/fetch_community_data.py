"""
Fetches real submission histories from a diverse SAMPLE of public
Codeforces users - not just you. This becomes the training data for
Problem B's classifier (predicting solve probability).

Why: a single user's history (even a very active one) is far too small
to train a generalizable model on. Codeforces' API is public for ANY
handle, so we can legitimately build a much larger, real dataset this
way - no scraping, no auth needed, fully within their public API.

Approach:
1. Call user.ratedList ONCE to get every rated Codeforces user.
2. Bucket them by rating (beginner/intermediate/advanced/expert) so
   our sample isn't biased toward only strong competitive programmers -
   we want a realistic spread of skill levels, same as our real users.
3. Sample N handles per bucket.
4. For each sampled handle, fetch their recent submissions (user.status).
5. Save everything to data/raw/community_codeforces.json

IMPORTANT: Codeforces asks API consumers not to hammer their servers.
We sleep between requests. This script takes a few minutes to run -
that's expected, let it finish. Run it ONCE (or occasionally to refresh),
not on every pipeline run.

Usage:
    python src/data_collection/fetch_community_data.py --per-bucket 40
"""

import requests
import json
import time
import os
import argparse
import random

API_BASE = "https://codeforces.com/api"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

# (min_rating, max_rating, label) - covers beginner through expert
RATING_BUCKETS = [
    (0, 1200, "beginner"),
    (1200, 1600, "intermediate"),
    (1600, 2000, "advanced"),
    (2000, 4000, "expert"),
]

REQUEST_DELAY_SECONDS = 1.0  # be polite to Codeforces' servers


def get_rated_handles():
    """
    ONE API call to get every rated user on Codeforces, with their
    current rating. We'll sample from this list ourselves rather
    than hitting the API repeatedly.
    """
    response = requests.get(f"{API_BASE}/user.ratedList", params={"activeOnly": "true"}, timeout=30)
    response.raise_for_status()
    data = response.json()
    if data["status"] != "OK":
        raise ValueError(f"Codeforces API error: {data.get('comment')}")
    return [{"handle": u["handle"], "rating": u["rating"]} for u in data["result"]]


def sample_diverse_handles(all_users, per_bucket=40, seed=42):
    """
    Picks a random sample of handles from EACH rating bucket, so our
    training data represents beginners through experts, not just
    whoever happens to be top-ranked.
    """
    random.seed(seed)
    sampled = []
    for min_r, max_r, label in RATING_BUCKETS:
        bucket_users = [u for u in all_users if min_r <= u["rating"] < max_r]
        chosen = random.sample(bucket_users, min(per_bucket, len(bucket_users)))
        print(f"  {label} ({min_r}-{max_r}): {len(bucket_users)} available, sampled {len(chosen)}")
        sampled.extend(chosen)
    return sampled


def fetch_user_submissions(handle, max_submissions=300):
    """
    Fetches up to max_submissions recent submissions for one user.
    Returns None if the request fails (private profile, deleted
    account, rate limit, etc.) - we just skip that user, no big deal
    when sampling from hundreds of handles.
    """
    try:
        response = requests.get(
            f"{API_BASE}/user.status",
            params={"handle": handle, "from": 1, "count": max_submissions},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        if data["status"] != "OK":
            return None

        attempts = []
        for sub in data["result"]:
            problem = sub["problem"]
            if "rating" not in problem:
                continue  # skip unrated problems
            attempts.append({
                "problem_id": f"{problem.get('contestId', '')}{problem.get('index', '')}",
                "tags": problem.get("tags", []),
                "rating": problem["rating"],
                "verdict": sub.get("verdict"),
                "timestamp": sub.get("creationTimeSeconds"),
            })
        return attempts
    except requests.RequestException:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-bucket", type=int, default=40,
                         help="How many users to sample per rating bucket")
    args = parser.parse_args()

    print("Fetching full rated user list (1 API call)...")
    all_users = get_rated_handles()
    print(f"Found {len(all_users)} rated users total.\n")

    print("Sampling diverse handles across skill levels:")
    sampled = sample_diverse_handles(all_users, per_bucket=args.per_bucket)
    print(f"\nTotal sampled: {len(sampled)} users. Fetching their submissions...")
    print(f"(This takes ~{len(sampled) * REQUEST_DELAY_SECONDS:.0f} seconds - be patient)\n")

    community_data = []
    for i, user in enumerate(sampled, 1):
        handle = user["handle"]
        attempts = fetch_user_submissions(handle)
        if attempts:
            community_data.append({
                "handle": handle,
                "rating": user["rating"],
                "attempts": attempts,
            })
            print(f"  [{i}/{len(sampled)}] {handle}: {len(attempts)} rated attempts fetched")
        else:
            print(f"  [{i}/{len(sampled)}] {handle}: skipped (no data / private / error)")
        time.sleep(REQUEST_DELAY_SECONDS)

    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, "community_codeforces.json")
    with open(out_path, "w") as f:
        json.dump(community_data, f, indent=2)

    total_attempts = sum(len(u["attempts"]) for u in community_data)
    print(f"\nDone. Saved {len(community_data)} users, {total_attempts} total attempts -> {out_path}")


if __name__ == "__main__":
    main()
