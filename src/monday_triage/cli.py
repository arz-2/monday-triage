"""Command-line entry point: monday-triage OWNER/REPO [options].

One run:

1. Load settings (.env and flags), the repo's last-run snapshot, and the
   chat models, so setup problems fail before any GitHub request.
2. Search GitHub for issues updated since the window start and label each
   new, newly active, closed, or quiet (delta.py, plain code).
3. Fetch linked PRs, reuse cached summaries for threads that didn't change,
   and summarize the rest (thread_state.py).
4. Group the week's issues into themes (themes.py).
5. Print the report, then save the snapshot. Nothing is saved after a
   failure, so a failed run can simply be run again.

If nothing changed since the last run, the last report is reprinted with 0
LLM calls. Every failure prints one line to stderr and exits non-zero:
2 for bad settings, 1 for anything that went wrong during the run.
"""

import argparse
import concurrent.futures
import dataclasses
import datetime
import os
import sys
from collections.abc import Callable, Mapping
from typing import Any

import dotenv
from langchain_core.language_models import BaseChatModel

from monday_triage import config
from monday_triage.budget import BudgetGuard
from monday_triage.delta import (
    NEWLY_ACTIVE,
    QUIET,
    DeltaItem,
    classify,
    split_by_cap,
)
from monday_triage.github_client import Comment, GitHubClient, LinkedPR
from monday_triage.llm import build_chat_model
from monday_triage.report import (
    CACHED_SUMMARY,
    CACHED_WHAT_CHANGED,
    FALLBACK,
    NEW_SUMMARY,
    Outcome,
    RunStats,
    build_report,
    mark_unchanged,
    over_cap_warning,
    render,
)
from monday_triage.state import (
    IssueRecord,
    Snapshot,
    load_snapshot,
    save_snapshot,
)
from monday_triage.themes import find_themes
from monday_triage.thread_state import summarize_threads

EXIT_OK, EXIT_ERROR, EXIT_USAGE = 0, 1, 2

ModelFactory = Callable[[config.Settings, int], BaseChatModel]


def main() -> None:
    """Console-script entry point: loads .env, runs, exits with the code."""
    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    sys.exit(run())


def run(
    argv: list[str] | None = None,
    env: Mapping[str, str] | None = None,
    client: GitHubClient | None = None,
    model_factory: ModelFactory = build_chat_model,
    now: datetime.datetime | None = None,
) -> int:
    """Runs one triage and prints the report.

    Args:
        argv: Command-line arguments; None reads sys.argv.
        env: Environment variables; None reads os.environ. Tests pass their
            own so the real .env is never used.
        client: GitHub client; None builds one from the settings.
        model_factory: Builds a chat model capped at a max output size.
        now: Current time; None uses the clock.

    Returns:
        The exit code.
    """
    args = _parse_args(argv)
    try:
        settings = config.load_settings(
            args.repo,
            since=args.since,
            max_issues=args.max_issues,
            budget_usd=args.budget_usd,
            price_per_mtok=args.price_per_mtok,
            output_format=args.format,
            env=os.environ if env is None else env,
        )
    except config.ConfigError as e:
        _error(str(e))
        return EXIT_USAGE

    if settings.github_token is None:
        _warn(
            "GITHUB_TOKEN is not set; GitHub allows 60 requests an hour "
            "(about 25 issues)."
        )
    client = client or GitHubClient(token=settings.github_token)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    try:
        report, snapshot = _triage(settings, client, model_factory, now)
        print(render(report, settings.output_format))
        save_snapshot(settings.state_dir, snapshot)
    except Exception as e:  # noqa: BLE001 - every failure must exit cleanly.
        _error(f"{type(e).__name__}: {e}")
        return EXIT_ERROR
    return EXIT_OK


def _triage(
    settings: config.Settings,
    client: GitHubClient,
    model_factory: ModelFactory,
    now: datetime.datetime,
) -> tuple[dict[str, Any], Snapshot]:
    """Runs the pipeline and returns the report and the snapshot to save."""
    snapshot = load_snapshot(settings.state_dir, settings.repo)
    summary_model = model_factory(settings, config.MAX_OUTPUT_TOKENS_SUMMARY)
    themes_model = model_factory(settings, config.MAX_OUTPUT_TOKENS_THEMES)
    since, window = _window(settings, snapshot, now)

    issues = client.list_changed_issues(settings.repo, since)
    items = classify(issues, snapshot, since)
    to_summarize, over_cap = split_by_cap(items, settings.max_issues)
    if over_cap:
        _warn(over_cap_warning(len(items), len(to_summarize)))
    linked = _fetch_linked_prs(client, settings.repo, to_summarize)
    pr_keys = {n: _pr_key(prs) for n, prs in linked.items()}
    to_summarize = [_relabel(i, snapshot, pr_keys) for i in to_summarize]
    cached = {
        item.issue.number: summary
        for item in to_summarize
        if (summary := _cached(snapshot, item, pr_keys)) is not None
    }

    unchanged = (
        settings.since is None
        and snapshot.last_report is not None
        and snapshot.last_run is not None
        and not over_cap
        and all(item.label == QUIET for item in to_summarize)
        and len(cached) == len(to_summarize)
    )
    if unchanged:
        report = mark_unchanged(
            snapshot.last_report, snapshot.last_run, client.requests_made
        )
        return report, dataclasses.replace(snapshot, last_run=now)

    guard = BudgetGuard(settings.budget_usd, settings.price_per_mtok)
    misses = [i for i in to_summarize if i.issue.number not in cached]
    results = summarize_threads(
        misses,
        lambda number: _comments(client, settings.repo, number),
        since,
        summary_model,
        guard,
        get_linked_prs=linked.__getitem__,
    )
    outcomes = {}
    for item in to_summarize:
        number = item.issue.number
        if number in cached:
            shown = dict(cached[number], what_changed=CACHED_WHAT_CHANGED)
            outcomes[number] = Outcome(shown, CACHED_SUMMARY)
        elif results[number].summary is not None:
            outcomes[number] = Outcome(results[number].summary, NEW_SUMMARY)
        else:
            outcomes[number] = Outcome(None, FALLBACK, results[number].error)
    themes = find_themes(
        to_summarize,
        {n: o.summary for n, o in outcomes.items()},
        snapshot.theme_names,
        themes_model,
        guard,
    )

    report = build_report(
        settings.repo,
        since,
        now,
        window,
        to_summarize,
        outcomes,
        over_cap,
        themes,
        RunStats(
            llm_calls=guard.calls,
            cost_usd=guard.spent_usd,
            budget_usd=settings.budget_usd,
            github_requests=client.requests_made,
            model=settings.llm_model,
        ),
        {
            n: [pr.number for pr in prs if pr.state == "open"]
            for n, prs in linked.items()
        },
    )
    records = dict(snapshot.issues)
    for item in to_summarize:
        number = item.issue.number
        # Cache the summary as generated, not the "no new activity" copy.
        summary = cached.get(number) or outcomes[number].summary
        records[number] = IssueRecord(
            state=item.issue.state,
            comments_count=item.issue.comments_count,
            summary=summary,
            linked_prs=pr_keys[number],
        )
    new_snapshot = Snapshot(
        repo=settings.repo,
        last_run=now,
        issues=records,
        # A week with no valid themes keeps last week's names for reuse.
        theme_names=[t.name for t in themes.themes] or snapshot.theme_names,
        last_report=report,
    )
    return report, new_snapshot


def _window(
    settings: config.Settings, snapshot: Snapshot, now: datetime.datetime
) -> tuple[datetime.datetime, str]:
    """Picks the window start: --since, else the last run, else 7 days."""
    if settings.since is not None:
        return settings.since, "from --since"
    if snapshot.last_run is not None:
        return snapshot.last_run, "since last run"
    days = config.DEFAULT_LOOKBACK_DAYS
    return now - datetime.timedelta(days=days), f"last {days} days"


def _fetch_linked_prs(
    client: GitHubClient, repo: str, items: list[DeltaItem]
) -> dict[int, list[LinkedPR]]:
    with concurrent.futures.ThreadPoolExecutor(
        config.MAX_CONCURRENT_CALLS
    ) as pool:
        numbers = [item.issue.number for item in items]
        prs = pool.map(lambda n: client.get_linked_prs(repo, n), numbers)
        return dict(zip(numbers, prs, strict=True))


def _pr_key(prs: list[LinkedPR]) -> list[str]:
    return [f"{pr.number}:{pr.state}" for pr in prs]


def _relabel(
    item: DeltaItem, snapshot: Snapshot, pr_keys: dict[int, list[str]]
) -> DeltaItem:
    """Marks a quiet issue newly active if a linked PR changed state.

    delta.py can't see linked PRs (they are fetched after labeling), so
    this keeps "quiet" meaning "nothing that needs a new summary".
    """
    record = snapshot.issues.get(item.issue.number)
    if item.label != QUIET or record is None:
        return item
    if record.linked_prs == pr_keys[item.issue.number]:
        return item
    return dataclasses.replace(item, label=NEWLY_ACTIVE)


def _cached(
    snapshot: Snapshot, item: DeltaItem, pr_keys: dict[int, list[str]]
) -> dict[str, Any] | None:
    issue = item.issue
    return snapshot.cached_summary(
        issue.number, issue.state, issue.comments_count, pr_keys[issue.number]
    )


def _comments(client: GitHubClient, repo: str, number: int) -> list[Comment]:
    comments, truncated = client.get_comments(repo, number)
    if truncated:
        _warn(
            f"#{number} has more comments than fetched; using the first ones."
        )
    return comments


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="monday-triage",
        description="What changed in a GitHub repo's issues since last run.",
    )
    parser.add_argument("repo", help="repository as OWNER/NAME")
    parser.add_argument("--since", help="window start, YYYY-MM-DD (UTC)")
    parser.add_argument(
        "--max-issues",
        type=int,
        help=f"most issues to summarize (default {config.DEFAULT_MAX_ISSUES})",
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        help=f"spending cap (default {config.DEFAULT_BUDGET_USD})",
    )
    parser.add_argument(
        "--price-per-mtok",
        metavar="IN,OUT",
        help="model price in USD per million tokens, if not built in",
    )
    parser.add_argument("--format", choices=config.FORMATS)
    return parser.parse_args(argv)


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


if __name__ == "__main__":
    main()
