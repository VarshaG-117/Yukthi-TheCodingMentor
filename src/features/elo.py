"""
Per-topic Elo skill rating.

Same math as chess Elo / Codeforces ratings, applied per DSA topic
(arrays, dp, graphs, etc.) instead of per player-vs-player.

Core idea:
    expected_score = how likely the user is to solve a problem of a
                      given difficulty, based on their CURRENT rating.
    new_rating = old_rating + K * (actual_outcome - expected_score)

    actual_outcome: 1.0 if solved, 0.0 if failed
    K: how much a single result should move the rating (bigger K =
       rating reacts faster but is noisier; smaller K = more stable
       but slower to adapt)
"""

DEFAULT_RATING = 1200  # sensible starting point - "average beginner"
K_FACTOR = 32           # standard chess default, works fine here too


def expected_score(user_rating, problem_difficulty):
    """
    Probability the user solves a problem of this difficulty, given
    their current rating. Standard Elo logistic formula.

    Returns a value between 0 and 1.
    Example: same rating vs difficulty -> 0.5 (coin flip)
             user rating 400 pts higher -> ~0.91 (very likely to solve)
             user rating 400 pts lower  -> ~0.09 (very unlikely to solve)
    """
    return 1.0 / (1.0 + 10 ** ((problem_difficulty - user_rating) / 400))


def update_rating(user_rating, problem_difficulty, solved: bool, k=K_FACTOR):
    """
    Returns the user's NEW rating after attempting one problem.

    user_rating: their rating BEFORE this attempt
    problem_difficulty: numeric difficulty of the problem they attempted
    solved: True if they solved it, False if they failed/gave up
    """
    actual = 1.0 if solved else 0.0
    expected = expected_score(user_rating, problem_difficulty)
    new_rating = user_rating + k * (actual - expected)
    return round(new_rating, 1)


class TopicRatings:
    """
    Tracks a user's Elo rating PER TOPIC (arrays, dp, graphs, etc.)
    and updates them as we feed in solved/failed attempts in
    chronological order.
    """

    def __init__(self, default_rating=DEFAULT_RATING, k=K_FACTOR):
        self.default_rating = default_rating
        self.k = k
        self.ratings = {}  # topic -> current rating

    def get_rating(self, topic):
        """Returns current rating for a topic, or the default if unseen."""
        return self.ratings.get(topic, self.default_rating)

    def record_attempt(self, topics, difficulty, solved):
        """
        Updates rating for EVERY topic tag on this problem (a problem
        can have multiple tags, e.g. ["dp", "graphs"] - both topic
        ratings get updated).

        Call this in chronological order (oldest attempt first) for
        each of a user's submissions.
        """
        for topic in topics:
            current = self.get_rating(topic)
            new_rating = update_rating(current, difficulty, solved, self.k)
            self.ratings[topic] = new_rating

    def as_dict(self):
        """Returns a plain dict snapshot of all topic ratings so far."""
        return dict(self.ratings)


if __name__ == "__main__":
    # Quick sanity check - run this file directly to see Elo in action.
    # A user starts at 1200 in "dp" and solves 3 increasingly hard
    # DP problems, then fails one that's too hard.
    tr = TopicRatings()

    attempts = [
        (["dp"], 900, True),    # easy win, small rating gain
        (["dp"], 1300, True),   # harder win, bigger rating gain
        (["dp"], 1600, True),   # even harder win, big rating gain
        (["dp"], 2400, False),  # way too hard, small rating loss (expected to fail anyway)
    ]

    print("Starting dp rating:", tr.get_rating("dp"))
    for topics, difficulty, solved in attempts:
        tr.record_attempt(topics, difficulty, solved)
        outcome = "solved" if solved else "failed"
        print(f"  attempted difficulty {difficulty}, {outcome} -> new dp rating: {tr.get_rating('dp')}")

    print("\nFinal ratings:", tr.as_dict())
