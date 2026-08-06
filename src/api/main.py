"""
FastAPI serving layer for the coding mentor recommender.

Loads the trained classifier + both problem catalogs once at startup,
then for each request:
  1. Loads the user's current per-topic skill ratings
  2. Scores every candidate problem's predicted solve probability
  3. Filters out problems they've already solved
  4. Returns the ones closest to the TARGET probability (default 0.7 -
     challenging but achievable, same "desirable difficulty" idea from
     our original design discussion)

Run locally:
    uvicorn src.api.main:app --reload
Then open http://localhost:8000/docs for interactive API docs.

KNOWN LIMITATION: LeetCode and Codeforces use different tag vocabularies
(e.g. "two-pointers" vs "two pointers", "dp" vs "dynamic-programming").
Where a LeetCode problem's tag doesn't match anything in the user's
Codeforces-derived topic ratings, we fall back to their default/overall
rating. This is a reasonable approximation, not a perfect cross-platform
mapping - worth mentioning as a known limitation in your report.
"""

import json
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

LEETCODE_DIFFICULTY_MAP = {"Easy": 900, "Medium": 1400, "Hard": 1900}
DEFAULT_RATING = 1200

app = FastAPI(title="Coding Mentor Recommender API")

# ---- Loaded once at startup, reused across requests ----
_model = None
_candidate_problems = None


def load_model():
    global _model
    if _model is None:
        model_path = os.path.join(MODELS_DIR, "success_classifier.pkl")
        _model = joblib.load(model_path)
    return _model


def load_candidate_problems():
    """
    Combines LeetCode + Codeforces catalogs into one unified list of
    candidate problems, each with a numeric difficulty and topic tags.
    """
    global _candidate_problems
    if _candidate_problems is not None:
        return _candidate_problems

    problems = []

    lc_path = os.path.join(DATA_RAW, "leetcode_catalog.json")
    if os.path.exists(lc_path):
        with open(lc_path) as f:
            for p in json.load(f):
                problems.append({
                    "platform": "leetcode",
                    "id": p["slug"],
                    "title": p["title"],
                    "url": p["url"],
                    "difficulty": LEETCODE_DIFFICULTY_MAP[p["difficulty"]],
                    "tags": p["tags"],
                })

    cf_path = os.path.join(DATA_RAW, "codeforces_catalog.json")
    if os.path.exists(cf_path):
        with open(cf_path) as f:
            for p in json.load(f):
                problems.append({
                    "platform": "codeforces",
                    "id": p["id"],
                    "title": p["title"],
                    "url": p["url"],
                    "difficulty": p["difficulty"],
                    "tags": p["tags"],
                })

    _candidate_problems = problems
    return problems


def load_user_topic_ratings(handle):
    """
    Loads a user's current per-topic skill ratings, built by
    build_features.py. Returns an empty dict (meaning: use the
    default rating for everything) if we've never seen this user -
    this IS our cold-start path for brand new users.
    """
    path = os.path.join(DATA_PROCESSED, f"topic_ratings_{handle}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_solved_ids(handle):
    """
    Best-effort set of problem IDs this user has already solved, so we
    don't recommend something they've done before. Checks both a
    LeetCode and a Codeforces raw file matching this handle, if present.
    """
    solved = set()
    for pattern_prefix in ["leetcode_user_", "codeforces_user_"]:
        import glob
        matches = glob.glob(os.path.join(DATA_RAW, f"{pattern_prefix}{handle}_*.json"))
        if not matches:
            continue
        latest = max(matches, key=os.path.getmtime)
        with open(latest) as f:
            data = json.load(f)
        if "solved_problem_ids" in data:
            solved.update(data["solved_problem_ids"])
        if "recent_solved" in data:
            solved.update(s["slug"] for s in data["recent_solved"])
    return solved


class RecommendedProblem(BaseModel):
    platform: str
    title: str
    url: str
    difficulty: float
    tags: List[str]
    predicted_solve_probability: float


class RecommendationResponse(BaseModel):
    handle: str
    is_cold_start: bool
    recommendations: List[RecommendedProblem]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{handle}", response_model=RecommendationResponse)
def recommend(
    handle: str,
    count: int = Query(10, ge=1, le=50, description="How many problems to recommend"),
    target_probability: float = Query(0.7, ge=0.1, le=0.95,
                                       description="Target solve probability - the 'sweet spot' difficulty"),
):
    model = load_model()
    candidates = load_candidate_problems()
    if not candidates:
        raise HTTPException(status_code=500, detail="No candidate problems loaded - check data/raw catalogs exist")

    topic_ratings = load_user_topic_ratings(handle)
    is_cold_start = len(topic_ratings) == 0
    solved_ids = load_solved_ids(handle)

    rows = []
    kept_candidates = []
    for problem in candidates:
        if problem["id"] in solved_ids:
            continue  # don't recommend what they've already solved

        if problem["tags"]:
            ratings = [topic_ratings.get(t, DEFAULT_RATING) for t in problem["tags"]]
            user_rating = sum(ratings) / len(ratings)
        else:
            user_rating = DEFAULT_RATING

        rating_gap = problem["difficulty"] - user_rating
        rows.append({
            "difficulty": problem["difficulty"],
            "user_rating_before": user_rating,
            "rating_gap": rating_gap,
        })
        kept_candidates.append(problem)

    if not rows:
        raise HTTPException(status_code=404, detail="No unsolved candidate problems found")

    X = pd.DataFrame(rows)
    predicted_probs = model.predict_proba(X)[:, 1]

    scored = []
    for problem, prob in zip(kept_candidates, predicted_probs):
        scored.append((abs(prob - target_probability), problem, prob))

    scored.sort(key=lambda x: x[0])
    top = scored[:count]

    recommendations = [
        RecommendedProblem(
            platform=p["platform"],
            title=p["title"],
            url=p["url"],
            difficulty=p["difficulty"],
            tags=p["tags"],
            predicted_solve_probability=round(float(prob), 3),
        )
        for _, p, prob in top
    ]

    return RecommendationResponse(
        handle=handle,
        is_cold_start=is_cold_start,
        recommendations=recommendations,
    )