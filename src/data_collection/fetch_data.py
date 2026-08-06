"""
Fetches data from both LeetCode and CodeChef and saves it to data/raw/.

Usage:
    python src/data_collection/fetch_data.py --leetcode <username> --codechef <username>

This is intentionally simple (no async, no retries) for Phase 2.
We'll harden it later once the pipeline shape is proven.
"""

import argparse
import json
import os
from datetime import datetime, timezone

from leetcode_client import get_user_solved as lc_get_user, get_problem_catalog as lc_get_catalog
from codeforces_client import get_user_solved as cf_get_user, get_problem_catalog as cf_get_catalog

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def save_json(obj, filename):
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, filename)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leetcode", help="LeetCode username")
    parser.add_argument("--codeforces", help="Codeforces handle")
    parser.add_argument("--skip-catalog", action="store_true",
                         help="Skip re-fetching problem catalogs (they don't change often)")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    if args.leetcode:
        print(f"\n=== LeetCode: {args.leetcode} ===")
        user_data = lc_get_user(args.leetcode)
        save_json(user_data, f"leetcode_user_{args.leetcode}_{timestamp}.json")

    if args.codeforces:
        print(f"\n=== Codeforces: {args.codeforces} ===")
        user_data = cf_get_user(args.codeforces)
        save_json(user_data, f"codeforces_user_{args.codeforces}_{timestamp}.json")

    if not args.skip_catalog:
        print("\n=== Fetching problem catalogs ===")
        lc_catalog = lc_get_catalog(limit=2500)
        save_json(lc_catalog, "leetcode_catalog.json")

        cf_catalog = cf_get_catalog()
        save_json(cf_catalog, "codeforces_catalog.json")


if __name__ == "__main__":
    main()
