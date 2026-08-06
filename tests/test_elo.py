"""
Unit tests for src/features/elo.py

Run with: pytest tests/test_elo.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "features"))

from elo import expected_score, update_rating, TopicRatings


def test_expected_score_equal_ratings_is_fifty_fifty():
    assert abs(expected_score(1200, 1200) - 0.5) < 0.01


def test_expected_score_higher_rating_means_higher_chance():
    weak_user_score = expected_score(800, 1200)
    strong_user_score = expected_score(1600, 1200)
    assert strong_user_score > weak_user_score


def test_update_rating_increases_on_solve():
    new_rating = update_rating(1200, 1200, solved=True)
    assert new_rating > 1200


def test_update_rating_decreases_on_fail():
    new_rating = update_rating(1200, 1200, solved=False)
    assert new_rating < 1200


def test_solving_harder_problem_gains_more_than_solving_easier():
    gain_easy = update_rating(1200, 900, solved=True) - 1200
    gain_hard = update_rating(1200, 1600, solved=True) - 1200
    assert gain_hard > gain_easy


def test_topic_ratings_starts_at_default():
    tr = TopicRatings(default_rating=1200)
    assert tr.get_rating("dp") == 1200


def test_topic_ratings_updates_independently_per_topic():
    tr = TopicRatings()
    tr.record_attempt(["dp"], 1600, solved=True)
    assert tr.get_rating("graphs") == tr.default_rating
    assert tr.get_rating("dp") != tr.default_rating


def test_topic_ratings_custom_seed():
    tr = TopicRatings(default_rating=1800)
    assert tr.get_rating("anything") == 1800