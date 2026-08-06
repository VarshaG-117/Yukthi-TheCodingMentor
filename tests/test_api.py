"""
Tests for src/api/main.py

Run with: pytest tests/test_api.py -v
"""

import sys
import os
import json
import pytest
import pandas as pd
import lightgbm as lgb
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def test_env(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    models_dir = tmp_path / "models"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    leetcode_catalog = [
        {"platform": "leetcode", "id": "1", "title": "Two Sum", "slug": "two-sum",
         "url": "https://leetcode.com/problems/two-sum/", "difficulty": "Easy",
         "acceptance_rate": 50, "tags": ["array"]},
    ]
    (raw_dir / "leetcode_catalog.json").write_text(json.dumps(leetcode_catalog))

    codeforces_catalog = [
        {"platform": "codeforces", "id": "1A", "contest_id": 1, "index": "A",
         "title": "CF Easy", "url": "https://codeforces.com/problemset/problem/1/A",
         "difficulty": 900, "tags": ["dp"]},
        {"platform": "codeforces", "id": "2B", "contest_id": 2, "index": "B",
         "title": "CF Hard", "url": "https://codeforces.com/problemset/problem/2/B",
         "difficulty": 2000, "tags": ["graphs"]},
    ]
    (raw_dir / "codeforces_catalog.json").write_text(json.dumps(codeforces_catalog))

    (processed_dir / "topic_ratings_knownuser.json").write_text(json.dumps({"dp": 1300}))

    rows = []
    for diff in [800, 1200, 1600, 2000]:
        for rating in [900, 1200, 1500]:
            gap = diff - rating
            prob = 1 / (1 + 10 ** (gap / 400))
            solved = 1 if prob > 0.5 else 0
            rows.append({"difficulty": diff, "user_rating_before": rating,
                         "rating_gap": gap, "solved": solved})
    df = pd.DataFrame(rows)
    model = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
    model.fit(df[["difficulty", "user_rating_before", "rating_gap"]], df["solved"])
    joblib.dump(model, models_dir / "success_classifier.pkl")

    import api.main as main_module
    main_module.DATA_RAW = str(raw_dir)
    main_module.DATA_PROCESSED = str(processed_dir)
    main_module.MODELS_DIR = str(models_dir)
    main_module._model = None
    main_module._candidate_problems = None

    from fastapi.testclient import TestClient
    return TestClient(main_module.app)


def test_health_check(test_env):
    response = test_env.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_known_user_returns_recommendations(test_env):
    response = test_env.get("/recommend/knownuser?count=5")
    assert response.status_code == 200
    data = response.json()
    assert data["handle"] == "knownuser"
    assert data["is_cold_start"] is False
    assert len(data["recommendations"]) > 0


def test_recommend_new_user_is_cold_start(test_env):
    response = test_env.get("/recommend/brandnewuser?count=5")
    assert response.status_code == 200
    data = response.json()
    assert data["is_cold_start"] is True


def test_recommendations_have_required_fields(test_env):
    response = test_env.get("/recommend/knownuser?count=5")
    data = response.json()
    for rec in data["recommendations"]:
        assert "platform" in rec
        assert "title" in rec
        assert "url" in rec
        assert "predicted_solve_probability" in rec
        assert 0 <= rec["predicted_solve_probability"] <= 1


def test_count_parameter_limits_results(test_env):
    response = test_env.get("/recommend/knownuser?count=1")
    data = response.json()
    assert len(data["recommendations"]) <= 1