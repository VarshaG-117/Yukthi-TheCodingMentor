# Coding Mentor — Personalized DSA Problem Recommender

An MLOps project that recommends coding problems (LeetCode + Codeforces) to a user
based on their solve history and skill level, using an Elo-based skill estimate
plus an ML classifier that predicts the probability a user will solve a given problem.

## How it works (high level)

1. **New user** picks a starting level (Beginner / Intermediate / Advanced) →
   they follow a fixed DSA curriculum order until we have enough solve data.
2. Once there's history, we compute a **per-topic Elo rating** for the user
   (arrays, DP, graphs, etc.) from their solved/failed problems.
3. A trained **ML model (Problem B)** predicts, for each candidate problem,
   the probability the user solves it — using features like topic rating,
   recency, and problem difficulty.
4. We recommend problems where success probability is in the "sweet spot"
   (~60-80%) — challenging but achievable.

## Project structure

```
coding-mentor/
├── data/
│   ├── raw/              # raw pulled data from LeetCode/CodeChef (DVC tracked)
│   └── processed/        # engineered feature tables (DVC tracked)
├── src/
│   ├── data_collection/  # scripts to fetch user + problem data (LeetCode + Codeforces)
│   ├── features/         # Elo rating + feature engineering
│   ├── models/           # training + evaluation scripts
│   └── api/              # FastAPI serving layer
├── notebooks/            # exploration notebooks
├── tests/                # unit tests
├── configs/              # config files (yaml) for pipeline params
└── .github/workflows/    # CI/CD pipeline definitions
```

## Build phases (we're doing this step by step)

- [x] Phase 1: Project structure + environment
- [x] Phase 2: Data collection (LeetCode + Codeforces)
- [ ] Phase 3: Data versioning with DVC
- [ ] Phase 4: Feature engineering (Elo ratings)
- [ ] Phase 5: Model training (success-probability classifier)
- [ ] Phase 6: Experiment tracking with MLflow
- [ ] Phase 7: Serving API (FastAPI)
- [ ] Phase 8: Containerization (Docker)
- [ ] Phase 9: CI/CD (GitHub Actions)
- [ ] Phase 10: Continuous Training + monitoring
