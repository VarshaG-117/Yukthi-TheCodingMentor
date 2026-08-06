"""
Codeforces data client.

Codeforces has a genuinely official, public, well-documented API - no
scraping, no login, no reverse-engineering needed. Docs:
    https://codeforces.com/apiHelp

Two endpoints we need:
1. problemset.problems -> the full problem catalog, WITH a numeric
   'rating' (difficulty) and 'tags' per problem. This is exactly the
   difficulty signal our Elo-style model needs.
2. user.status -> a specific user's full submission history (every
   attempt, verdict OK/WRONG_ANSWER/etc, which problem, when).

Run this file directly to test it:
    python src/data_collection/codeforces_client.py your_codeforces_handle
"""

import requests
import json
import sys

API_BASE = "https://codeforces.com/api"


def get_problem_catalog():
    """
    Fetches every problem on Codeforces with its rating and tags.
    Note: some problems have no 'rating' yet (very new problems) -
    we skip those since we can't estimate difficulty for them.
    """
    response = requests.get(f"{API_BASE}/problemset.problems", timeout=15)
    response.raise_for_status()
    data = response.json()

    if data["status"] != "OK":
        raise ValueError(f"Codeforces API error: {data.get('comment')}")

    problems = []
    for p in data["result"]["problems"]:
        if "rating" not in p:
            continue  # skip unrated problems - can't use them for difficulty matching
        problems.append({
            "platform": "codeforces",
            "id": f"{p['contestId']}{p['index']}",       # e.g. "1500A"
            "contest_id": p["contestId"],
            "index": p["index"],
            "title": p["name"],
            "url": f"https://codeforces.com/problemset/problem/{p['contestId']}/{p['index']}",
            "difficulty": p["rating"],                    # numeric, e.g. 800, 1200, 2100...
            "tags": p["tags"],                             # e.g. ["dp", "graphs"]
        })
    return problems


def get_user_solved(handle):
    """
    Fetches a user's full submission history and derives which
    problems they've solved (verdict == 'OK').
    """
    response = requests.get(
        f"{API_BASE}/user.status",
        params={"handle": handle},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    if data["status"] != "OK":
        raise ValueError(f"Codeforces user '{handle}' not found or API error: {data.get('comment')}")

    submissions = data["result"]

    solved_problem_ids = set()
    all_attempts = []
    for sub in submissions:
        problem = sub["problem"]
        problem_id = f"{problem.get('contestId', '')}{problem.get('index', '')}"
        verdict = sub.get("verdict")  # "OK" means accepted/solved

        all_attempts.append({
            "problem_id": problem_id,
            "problem_name": problem.get("name"),
            "tags": problem.get("tags", []),
            "rating": problem.get("rating"),
            "verdict": verdict,
            "timestamp": sub.get("creationTimeSeconds"),
        })

        if verdict == "OK":
            solved_problem_ids.add(problem_id)

    return {
        "platform": "codeforces",
        "handle": handle,
        "solved_count": len(solved_problem_ids),
        "solved_problem_ids": list(solved_problem_ids),
        "all_attempts": all_attempts,  # includes failed attempts too - useful signal for the model
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python codeforces_client.py <codeforces_handle>")
        sys.exit(1)

    handle = sys.argv[1]

    print(f"Fetching submissions for '{handle}'...")
    user_data = get_user_solved(handle)
    print(f"Solved {user_data['solved_count']} unique problems.")
    print(json.dumps(user_data["all_attempts"][:3], indent=2))  # show first 3 as a sample

    print("\nFetching problem catalog...")
    catalog = get_problem_catalog()
    print(f"Fetched {len(catalog)} rated problems. Example:")
    print(json.dumps(catalog[0], indent=2))
