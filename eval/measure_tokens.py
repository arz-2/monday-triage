"""Measures input tokens per thread and per week over past weeks.

The token caps in config.py (MAX_TOKENS_PER_ISSUE, MAX_TOTAL_TOKENS,
DEFAULT_MAX_ISSUES) were set from this measurement, not guessed. It rebuilds
each past week's set of changed issues (opened, closed, or commented on in
that week), trims every thread with the real trim_thread (but with the
per-issue cap lifted, to see how often the cap would bite), and counts tokens
the same way the budget guard does.

Usage (from the repo root; about 1 GitHub request per issue, so set
GITHUB_TOKEN):
    python eval/measure_tokens.py huggingface/datasets --weeks 8
"""

import argparse
import datetime
import os
import statistics
from unittest import mock

import dotenv

from monday_triage import config
from monday_triage.budget import estimate_tokens
from monday_triage.delta import NEWLY_ACTIVE, DeltaItem
from monday_triage.github_client import Comment, GitHubClient, Issue
from monday_triage.thread_state import render_thread, trim_thread


def main() -> None:
    """Fetches the weeks, measures them, and prints a table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo")
    parser.add_argument("--weeks", type=int, default=8)
    args = parser.parse_args()

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    client = GitHubClient(token=os.environ.get("GITHUB_TOKEN") or None)
    end = datetime.datetime.now(datetime.timezone.utc)
    start = end - datetime.timedelta(weeks=args.weeks)
    issues = client.list_changed_issues(args.repo, start)
    comments = {
        i.number: client.get_comments(args.repo, i.number)[0] for i in issues
    }
    print(f"Fetched {len(issues)} issues ({client.requests_made} requests).\n")

    cap = config.MAX_TOKENS_PER_ISSUE
    all_threads: list[int] = []
    print("week_start  issues  median    max   total  over_cap")
    for week in range(args.weeks):
        hi = end - datetime.timedelta(weeks=week)
        lo = hi - datetime.timedelta(weeks=1)
        sizes = [
            _thread_tokens(issue, comments[issue.number], lo, hi)
            for issue in issues
            if _active(issue, comments[issue.number], lo, hi)
        ]
        all_threads += sizes
        print(
            f"{lo:%Y-%m-%d}  {len(sizes):6}  "
            f"{statistics.median(sizes) if sizes else 0:6.0f}  "
            f"{max(sizes, default=0):5}  {sum(sizes):6}  "
            f"{sum(s > cap for s in sizes):8}"
        )

    sizes = sorted(all_threads)
    if not sizes:
        return
    print(
        f"\nPer thread (n={len(sizes)}): "
        f"median {statistics.median(sizes):.0f}, "
        f"p90 {sizes[int(0.9 * len(sizes))]}, "
        f"p95 {sizes[int(0.95 * len(sizes))]}, max {sizes[-1]}"
    )
    print(f"Threads over the {cap}-token cap: {sum(s > cap for s in sizes)}")


def _active(
    issue: Issue,
    comments: list[Comment],
    lo: datetime.datetime,
    hi: datetime.datetime,
) -> bool:
    """True if the issue was opened, closed, or commented on in [lo, hi)."""

    def inside(moment: datetime.datetime | None) -> bool:
        return moment is not None and lo <= moment < hi

    return (
        inside(issue.created_at)
        or inside(issue.closed_at)
        or any(inside(c.created_at) for c in comments)
    )


def _thread_tokens(
    issue: Issue,
    comments: list[Comment],
    lo: datetime.datetime,
    hi: datetime.datetime,
) -> int:
    """Tokens of the trimmed thread as of `hi`, per-issue cap lifted."""
    before = [c for c in comments if c.created_at < hi]
    with mock.patch.object(config, "MAX_TOKENS_PER_ISSUE", 10**9):
        thread = trim_thread(issue, before, lo)
    item = DeltaItem(issue, NEWLY_ACTIVE)
    return estimate_tokens(render_thread(item, thread, lo))


if __name__ == "__main__":
    main()
