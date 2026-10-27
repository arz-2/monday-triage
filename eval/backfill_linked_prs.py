"""Adds linked pull requests to a week recorded before they were fetched.

Comments are left as recorded, so existing hand labels stay valid. Only PRs
linked before the week's recorded_at are kept. A PR's state (open, merged,
closed) is as of the backfill, not the recording, so the script lists every
PR that isn't open for a check by hand.

Usage (from the repo root):
    python eval/backfill_linked_prs.py \
        tests/fixtures/huggingface__datasets_week.json
"""

import argparse
import dataclasses
import os
import pathlib

import dotenv
from week import load_week, save_week

from monday_triage.github_client import GitHubClient

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    """Fetches linked PRs for every issue in the week and saves it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fixture")
    args = parser.parse_args()

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    path = ROOT / args.fixture
    week = load_week(path)
    client = GitHubClient(token=os.environ.get("GITHUB_TOKEN") or None)
    linked_prs = {}
    for issue in week.issues:
        prs = client.get_linked_prs(week.repo, issue.number)
        linked_prs[issue.number] = [
            pr for pr in prs if pr.linked_at <= week.recorded_at
        ]
        for pr in linked_prs[issue.number]:
            if pr.state != "open":
                print(f"  #{issue.number}: PR #{pr.number} is {pr.state} now")
    save_week(dataclasses.replace(week, linked_prs=linked_prs), path)
    total = sum(len(prs) for prs in linked_prs.values())
    print(
        f"Added {total} linked PRs to {len(week.issues)} issues "
        f"({client.requests_made} requests)."
    )


if __name__ == "__main__":
    main()
