"""Labels each changed issue new, newly active, closed, or quiet.

Plain code, no LLM: the label follows from the issue's timestamps and what
the last-run snapshot remembers. Rules, first match wins:

1. closed: closed within the window. Wins over "new" for an issue opened and
   closed in the same week, since nothing is left for the maintainer to do.
2. new: opened within the window.
3. newly active: reopened, comment count changed, or never seen before (a
   first run has nothing to compare against).
4. quiet: in the window only because of a label or title edit; state and
   comment count match the snapshot. This mirrors the summary cache key.
"""

import dataclasses
import datetime

from monday_triage import config
from monday_triage.github_client import Issue
from monday_triage.state import Snapshot

NEW, NEWLY_ACTIVE, CLOSED, QUIET = config.DELTA_LABELS


@dataclasses.dataclass(frozen=True)
class DeltaItem:
    """A changed issue and its delta label.

    Attributes:
        issue: The issue as fetched.
        label: One of config.DELTA_LABELS.
    """

    issue: Issue
    label: str


def classify(
    issues: list[Issue], snapshot: Snapshot, since: datetime.datetime
) -> list[DeltaItem]:
    """Labels each changed issue.

    Args:
        issues: Issues updated at or after `since`.
        snapshot: What the last run saw.
        since: Start of the window (inclusive).

    Returns:
        One item per issue, most recently updated first.
    """
    items = [
        DeltaItem(issue, _label(issue, snapshot, since)) for issue in issues
    ]
    return sorted(items, key=lambda item: item.issue.updated_at, reverse=True)


def split_by_cap(
    items: list[DeltaItem], max_issues: int
) -> tuple[list[DeltaItem], list[DeltaItem]]:
    """Splits items into those to summarize and those over the cap.

    Args:
        items: Labeled issues, most recently updated first.
        max_issues: Most issues to summarize.

    Returns:
        The first `max_issues` items, and the rest (listed without a
        summary in the report).
    """
    return items[:max_issues], items[max_issues:]


def _label(issue: Issue, snapshot: Snapshot, since: datetime.datetime) -> str:
    closed_in_window = issue.closed_at is not None and issue.closed_at >= since
    if issue.state == "closed" and closed_in_window:
        return CLOSED
    if issue.created_at >= since:
        return NEW
    record = snapshot.issues.get(issue.number)
    if record is None:
        return NEWLY_ACTIVE
    if record.state != issue.state:
        return NEWLY_ACTIVE
    if record.comments_count != issue.comments_count:
        return NEWLY_ACTIVE
    return QUIET
