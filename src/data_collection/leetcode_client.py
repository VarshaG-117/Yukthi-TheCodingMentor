"""
LeetCode data client.

LeetCode doesn't have an official public API, but its website itself calls
a GraphQL endpoint (https://leetcode.com/graphql) to load profile pages.
We call the same endpoint directly - no login/API key needed for public data.

Two things we need from here:
1. get_problem_catalog()   -> ALL problems on LeetCode (id, title, difficulty, tags)
2. get_user_solved(username) -> which problems a specific user has solved

Run this file directly to test it:
    python src/data_collection/leetcode_client.py your_leetcode_username
"""

import requests
import json
import sys
import time

GRAPHQL_URL = "https://leetcode.com/graphql"

HEADERS = {
    "Content-Type": "application/json",
    # LeetCode blocks requests with no browser-like User-Agent
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://leetcode.com",
}


def get_problem_catalog(limit=3000):
    """
    Fetches the full list of LeetCode problems with difficulty, tags,
    and acceptance rate. This becomes our 'problem pool' to recommend from.
    """
    query = """
    query problemsetQuestionList($limit: Int) {
      problemsetQuestionList: questionList(
        categorySlug: ""
        limit: $limit
        skip: 0
        filters: {}
      ) {
        total: totalNum
        questions: data {
          questionFrontendId
          title
          titleSlug
          difficulty
          acRate
          topicTags {
            name
            slug
          }
          paidOnly: isPaidOnly
        }
      }
    }
    """
    response = requests.post(
        GRAPHQL_URL,
        headers=HEADERS,
        json={"query": query, "variables": {"limit": limit}},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    questions = data["data"]["problemsetQuestionList"]["questions"]

    problems = []
    for q in questions:
        if q["paidOnly"]:
            continue  # skip premium-only problems, most students can't access them
        problems.append({
            "platform": "leetcode",
            "id": q["questionFrontendId"],
            "title": q["title"],
            "slug": q["titleSlug"],
            "url": f"https://leetcode.com/problems/{q['titleSlug']}/",
            "difficulty": q["difficulty"],          # Easy / Medium / Hard
            "acceptance_rate": q["acRate"],
            "tags": [t["slug"] for t in q["topicTags"]],
        })
    return problems


def get_user_solved(username):
    """
    Fetches a user's solved-problem stats (counts by difficulty) and
    their recent accepted submissions.
    """
    query = """
    query userProfile($username: String!) {
      matchedUser(username: $username) {
        username
        submitStats: submitStatsGlobal {
          acSubmissionNum {
            difficulty
            count
          }
        }
      }
      recentAcSubmissionList(username: $username, limit: 50) {
        title
        titleSlug
        timestamp
      }
    }
    """
    response = requests.post(
        GRAPHQL_URL,
        headers=HEADERS,
        json={"query": query, "variables": {"username": username}},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("data", {}).get("matchedUser") is None:
        raise ValueError(f"LeetCode user '{username}' not found")

    user = data["data"]["matchedUser"]
    recent = data["data"]["recentAcSubmissionList"]

    solved_counts = {d["difficulty"]: d["count"] for d in user["submitStats"]["acSubmissionNum"]}

    return {
        "platform": "leetcode",
        "username": username,
        "solved_counts": solved_counts,  # e.g. {'All': 120, 'Easy': 60, 'Medium': 50, 'Hard': 10}
        "recent_solved": [
            {"title": r["title"], "slug": r["titleSlug"], "timestamp": r["timestamp"]}
            for r in recent
        ],
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python leetcode_client.py <leetcode_username>")
        sys.exit(1)

    username = sys.argv[1]

    print(f"Fetching profile for '{username}'...")
    user_data = get_user_solved(username)
    print(json.dumps(user_data, indent=2))

    print("\nFetching problem catalog (this takes a few seconds)...")
    catalog = get_problem_catalog(limit=200)  # small limit just for a quick test
    print(f"Fetched {len(catalog)} problems. Example:")
    print(json.dumps(catalog[0], indent=2))
