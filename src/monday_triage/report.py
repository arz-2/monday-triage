"""Builds the Monday report and renders it as text or JSON.

`build_report` turns one run's results into a plain dict, which is also what
the snapshot saves. Both renderers work only from that dict, so a reprinted
report (nothing changed since the last run) looks exactly like a fresh one,
in either format.

Text layout, most useful first:

1. Needs your attention: the issues where it's the maintainer's move, split
   by what to do (decided by plain code from the status and linked PRs):
   first reply needed, PR ready to review, reply waiting, and summaries that
   failed (check by hand, so a failure never hides an issue).
2. Themes.
3. Other issues, grouped by status.
4. Issues over --max-issues, then run stats.
"""

import collections
import dataclasses
import datetime
import json
from typing import Any

from monday_triage import config
from monday_triage.delta import DeltaItem
from monday_triage.themes import ThemesResult

NEW_SUMMARY, CACHED_SUMMARY, FALLBACK = "new", "cached", "fallback"
CACHED_WHAT_CHANGED = "No new activity since the last run."

FIRST_REPLY = "First reply needed"
PR_TO_REVIEW = "PR ready to review"
REPLY_WAITING = "Reply waiting"
CHECK_BY_HAND = "Summary unavailable, check by hand"
ATTENTION_GROUPS = (FIRST_REPLY, PR_TO_REVIEW, REPLY_WAITING, CHECK_BY_HAND)


@dataclasses.dataclass(frozen=True)
class Outcome:
    """How one issue's summary was obtained.

    Attributes:
        summary: Thread summary (thread_state's JSON form), or None.
        source: NEW_SUMMARY, CACHED_SUMMARY, or FALLBACK.
        error: Why the fallback happened, or None.
    """

    summary: dict[str, Any] | None
    source: str
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class RunStats:
    """Cost and effort of one run, shown at the end of the report.

    Attributes:
        llm_calls: LLM calls made, including retries.
        cost_usd: Dollars spent.
        budget_usd: The run's spending cap.
        github_requests: GitHub requests made.
        model: Model name.
    """

    llm_calls: int
    cost_usd: float
    budget_usd: float
    github_requests: int
    model: str


def build_report(
    repo: str,
    since: datetime.datetime,
    until: datetime.datetime,
    window: str,
    summarized: list[DeltaItem],
    outcomes: dict[int, Outcome],
    over_cap: list[DeltaItem],
    themes: ThemesResult,
    stats: RunStats,
    open_prs: dict[int, list[int]],
) -> dict[str, Any]:
    """Collects one run's results into a JSON-ready dict.

    Args:
        repo: Repository as "owner/name".
        since: Window start.
        until: When the run started.
        window: How the start was chosen, e.g. "since last run".
        summarized: Issues within --max-issues.
        outcomes: Summary outcome per summarized issue number.
        over_cap: Issues over --max-issues, listed without a summary.
        themes: Result of the themes step.
        stats: Cost and effort of the run.
        open_prs: Open linked PR numbers per summarized issue number.

    Returns:
        The report data.
    """
    issues = [
        _issue_entry(
            item, outcomes[item.issue.number], open_prs[item.issue.number]
        )
        for item in summarized
    ]
    order = {status: i for i, status in enumerate(config.STATUS_LABELS)}
    # Action first; fallbacks last. Ties keep most-recently-updated order.
    issues.sort(key=lambda e: order.get(e["status"], len(order)))
    themed = {n for theme in themes.themes for n in theme.issues}
    all_items = summarized + over_cap
    warnings = []
    if over_cap:
        warnings.append(over_cap_warning(len(all_items), len(summarized)))
    return {
        "repo": repo,
        "since": since.isoformat(),
        "until": until.isoformat(),
        "window": window,
        "warnings": warnings,
        "counts": dict(collections.Counter(item.label for item in all_items)),
        "themes": [dataclasses.asdict(theme) for theme in themes.themes],
        "unthemed": [
            item.issue.number
            for item in summarized
            if item.issue.number not in themed
        ],
        "themes_note": _themes_note(themes),
        "issues": issues,
        "not_summarized": [_basic_entry(item) for item in over_cap],
        "max_issues": len(summarized) if over_cap else None,
        "run": {
            **dataclasses.asdict(stats),
            "cached": sum(
                o.source == CACHED_SUMMARY for o in outcomes.values()
            ),
        },
    }


def over_cap_warning(changed: int, summarized: int) -> str:
    """Says that issues were left unsummarized and how to include them."""
    left = changed - summarized
    return (
        f"{changed} issues changed but only the {summarized} most recently "
        f"updated were summarized. {left} are listed at the end without a "
        f"summary; use --max-issues {changed} to include them."
    )


def mark_unchanged(
    report: dict[str, Any],
    last_run: datetime.datetime,
    github_requests: int,
) -> dict[str, Any]:
    """Returns the last report, flagged as reprinted, with this run's stats.

    Args:
        report: The last run's report data.
        last_run: When the last run started.
        github_requests: GitHub requests this run made.

    Returns:
        A copy of the report with "unchanged_since" set and zero LLM spend.
    """
    run = dict(report["run"], llm_calls=0, cost_usd=0.0)
    run["github_requests"] = github_requests
    return dict(report, unchanged_since=last_run.isoformat(), run=run)


def render(report: dict[str, Any], output_format: str) -> str:
    """Renders report data as "text" or "json"."""
    if output_format == "json":
        return json.dumps(report, indent=2)
    return render_text(report)


def render_text(report: dict[str, Any]) -> str:
    """Renders report data for the terminal."""
    lines: list[str] = []
    if "unchanged_since" in report:
        lines += [
            f"No changes since {_time(report['unchanged_since'])}. "
            "Showing the last report (0 LLM calls).",
            "",
        ]
    lines += _header(report)
    lines += [f"WARNING: {w}" for w in report.get("warnings", [])]
    if report.get("warnings"):
        lines.append("")
    lines += _attention_section(report)
    lines += _themes_section(report)
    lines += _other_section(report)
    lines += _over_cap_section(report)
    run = report["run"]
    lines.append(
        f"Run: {run['llm_calls']} LLM calls, ${run['cost_usd']:.4f} of "
        f"${run['budget_usd']:.2f} budget, {run['cached']} cached "
        f"summaries, {run['github_requests']} GitHub requests, "
        f"model {run['model']}"
    )
    return "\n".join(lines)


def _issue_entry(
    item: DeltaItem, outcome: Outcome, open_prs: list[int]
) -> dict[str, Any]:
    summary = outcome.summary or {}
    status = summary.get("status")
    return {
        **_basic_entry(item),
        "state": item.issue.state,
        "status": status,
        "where_it_stands": summary.get("where_it_stands"),
        "what_changed": summary.get("what_changed"),
        "open_prs": open_prs,
        "attention": _attention(status, open_prs),
        "source": outcome.source,
        "error": outcome.error,
    }


def _attention(status: str | None, open_prs: list[int]) -> str | None:
    """Says what the maintainer should do, or None if it's not their move."""
    if status is None:
        return CHECK_BY_HAND
    if status == "needs_triage":
        return FIRST_REPLY
    if status == "waiting_on_maintainer":
        return PR_TO_REVIEW if open_prs else REPLY_WAITING
    return None


def _basic_entry(item: DeltaItem) -> dict[str, Any]:
    return {
        "number": item.issue.number,
        "title": item.issue.title,
        "url": item.issue.url,
        "delta": item.label,
    }


def _themes_note(themes: ThemesResult) -> str | None:
    if themes.error:
        return f"Themes unavailable: {themes.error}"
    if themes.dropped:
        return "Dropped themes: " + "; ".join(themes.dropped)
    return None


def _time(iso: str) -> str:
    utc = datetime.datetime.fromisoformat(iso).astimezone(datetime.timezone.utc)
    return f"{utc:%Y-%m-%d %H:%M} UTC"


def _header(report: dict[str, Any]) -> list[str]:
    counts = report["counts"]
    total = sum(counts.values())
    by_label = ", ".join(
        f"{counts[label]} {label}"
        for label in config.DELTA_LABELS
        if counts.get(label)
    )
    return [
        f"{report['repo']}: issues changed {_time(report['since'])} to "
        f"{_time(report['until'])} ({report['window']})",
        f"{total} changed" + (f": {by_label}" if by_label else ""),
        "",
    ]


def _themes_section(report: dict[str, Any]) -> list[str]:
    if not report["themes"] and not report["themes_note"]:
        return []
    lines = ["THEMES"]
    for theme in report["themes"]:
        lines.append(
            f"  {theme['name']} ({len(theme['issues'])}): "
            f"{theme['description']}"
        )
        lines.append("    " + " ".join(f"#{n}" for n in theme["issues"]))
    if report["themes"] and report["unthemed"]:
        lines.append(
            "  Not in a theme: " + " ".join(f"#{n}" for n in report["unthemed"])
        )
    if report["themes_note"]:
        lines.append(f"  {report['themes_note']}")
    return [*lines, ""]


def _attention_section(report: dict[str, Any]) -> list[str]:
    entries = [e for e in report["issues"] if e.get("attention")]
    if not entries:
        return ["NEEDS YOUR ATTENTION: nothing this week", ""]
    lines = [f"NEEDS YOUR ATTENTION ({len(entries)})"]
    for group in ATTENTION_GROUPS:
        members = [e for e in entries if e.get("attention") == group]
        if members:
            lines.append(f"  {group} ({len(members)})")
            for entry in members:
                lines += _entry_lines(entry)
    return [*lines, ""]


def _other_section(report: dict[str, Any]) -> list[str]:
    entries = [e for e in report["issues"] if not e.get("attention")]
    if not entries:
        return []
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for entry in entries:
        groups[entry["status"]].append(entry)
    lines = [f"OTHER ISSUES ({len(entries)})"]
    for status, members in groups.items():
        lines.append(f"  {status} ({len(members)})")
        for entry in members:
            lines += _entry_lines(entry)
    return [*lines, ""]


def _entry_lines(entry: dict[str, Any]) -> list[str]:
    prs = ", ".join(f"#{n}" for n in entry.get("open_prs", []))
    lines = [
        f"    #{entry['number']} [{entry['delta']}] {entry['title']}"
        + (f" -> PR {prs}" if prs else "")
    ]
    if entry["status"] is None:
        lines.append(f"      {config.FALLBACK_TEXT} ({entry['error']})")
    else:
        lines.append(f"      {entry['where_it_stands']}")
        lines.append(f"      Changed: {entry['what_changed']}")
    lines.append(f"      {entry['url']}")
    return lines


def _over_cap_section(report: dict[str, Any]) -> list[str]:
    if not report["not_summarized"]:
        return []
    lines = [
        f"NOT SUMMARIZED (over --max-issues {report['max_issues']}, "
        "least recently updated)"
    ]
    for entry in report["not_summarized"]:
        lines.append(
            f"  #{entry['number']} [{entry['delta']}] {entry['title']}  "
            f"{entry['url']}"
        )
    return [*lines, ""]
