import datetime

from monday_triage.delta import (
    CLOSED,
    NEW,
    NEWLY_ACTIVE,
    QUIET,
    classify,
    split_by_cap,
)
from monday_triage.github_client import Issue
from monday_triage.state import IssueRecord, Snapshot

UTC = datetime.timezone.utc
SINCE = datetime.datetime(2026, 10, 19, tzinfo=UTC)
BEFORE = SINCE - datetime.timedelta(days=30)
DURING = SINCE + datetime.timedelta(days=2)


def _issue(
    number=1,
    created=BEFORE,
    updated=DURING,
    state="open",
    closed=None,
    comments=0,
):
    return Issue(
        number=number,
        title=f"Issue {number}",
        body="",
        author="someone",
        state=state,
        created_at=created,
        updated_at=updated,
        closed_at=closed,
        comments_count=comments,
        url="",
    )


def _label(issue, snapshot=None):
    snapshot = snapshot or Snapshot(repo="o/r")
    return classify([issue], snapshot, SINCE)[0].label


def _seen(number=1, state="open", comments=0):
    return Snapshot(repo="o/r", issues={number: IssueRecord(state, comments)})


def test_opened_in_window_is_new():
    assert _label(_issue(created=DURING)) == NEW


def test_closed_in_window_is_closed():
    assert _label(_issue(state="closed", closed=DURING)) == CLOSED


def test_opened_and_closed_in_same_week_is_closed():
    issue = _issue(created=DURING, state="closed", closed=DURING)
    assert _label(issue) == CLOSED


def test_new_comment_on_seen_issue_is_newly_active():
    assert _label(_issue(comments=3), _seen(comments=2)) == NEWLY_ACTIVE


def test_reopened_issue_is_newly_active():
    # Closed before the window, reopened during it: closed_at stays old.
    issue = _issue(state="open", closed=BEFORE)
    assert _label(issue, _seen(state="closed")) == NEWLY_ACTIVE


def test_unseen_old_issue_is_newly_active_on_first_run():
    assert _label(_issue(comments=4)) == NEWLY_ACTIVE


def test_label_edit_only_is_quiet():
    assert _label(_issue(comments=2), _seen(comments=2)) == QUIET


def test_comment_on_issue_closed_before_window_is_newly_active():
    issue = _issue(state="closed", closed=BEFORE, comments=5)
    assert _label(issue, _seen(state="closed", comments=4)) == NEWLY_ACTIVE


def test_boundary_at_exactly_since_counts_as_in_window():
    assert _label(_issue(created=SINCE)) == NEW
    assert _label(_issue(state="closed", closed=SINCE)) == CLOSED


def test_items_sorted_most_recently_updated_first():
    older = _issue(number=1, updated=SINCE)
    newer = _issue(number=2, updated=DURING)
    items = classify([older, newer], Snapshot(repo="o/r"), SINCE)
    assert [item.issue.number for item in items] == [2, 1]


def test_split_by_cap_keeps_most_recent():
    issues = [
        _issue(number=n, updated=SINCE + datetime.timedelta(hours=n))
        for n in range(1, 6)
    ]
    items = classify(issues, Snapshot(repo="o/r"), SINCE)
    summarize, over = split_by_cap(items, 3)
    assert [i.issue.number for i in summarize] == [5, 4, 3]
    assert [i.issue.number for i in over] == [2, 1]
