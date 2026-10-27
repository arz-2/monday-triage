import datetime
import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from monday_triage import config
from monday_triage.budget import BudgetGuard
from monday_triage.delta import NEW, NEWLY_ACTIVE, DeltaItem
from monday_triage.github_client import Issue
from monday_triage.themes import (
    ProposedTheme,
    check_themes,
    find_themes,
    render_issues,
)

NOW = datetime.datetime(2026, 10, 20, tzinfo=datetime.timezone.utc)


def _item(number, label=NEWLY_ACTIVE):
    issue = Issue(
        number=number,
        title=f"Issue {number}",
        body="",
        author="someone",
        state="open",
        created_at=NOW,
        updated_at=NOW,
        closed_at=None,
        comments_count=0,
        url="",
    )
    return DeltaItem(issue, label)


def _theme(name, *issues):
    return ProposedTheme(name=name, description="Shared cause.", issues=issues)


def _reply(*themes):
    return json.dumps({"themes": [t.model_dump() for t in themes]})


def _find(responses, items=None, previous=None):
    items = items or [_item(n) for n in (1, 2, 3, 4)]
    guard = BudgetGuard(1.0, (0.10, 0.50))
    model = FakeListChatModel(responses=responses)
    result = find_themes(items, {}, previous or [], model, guard)
    return result, guard


# --- Per-theme checks ---


def test_valid_themes_are_kept():
    kept, dropped = check_themes(
        [_theme("Streaming", 1, 2), _theme("Export", 3, 4)], {1, 2, 3, 4}
    )
    assert [(t.name, t.issues) for t in kept] == [
        ("Streaming", [1, 2]),
        ("Export", [3, 4]),
    ]
    assert dropped == []


def test_theme_citing_unknown_issue_is_dropped():
    kept, dropped = check_themes(
        [_theme("Streaming", 1, 2), _theme("Ghost", 3, 99)], {1, 2, 3}
    )
    assert [t.name for t in kept] == ["Streaming"]
    assert "#99" in dropped[0]


def test_theme_with_one_issue_is_dropped():
    kept, dropped = check_themes([_theme("Lonely", 1)], {1, 2})
    assert kept == [] and "fewer than 2" in dropped[0]


def test_issue_stays_in_its_first_theme_only():
    kept, _ = check_themes(
        [_theme("A", 1, 2, 3), _theme("B", 3, 4, 5)], {1, 2, 3, 4, 5}
    )
    assert [t.issues for t in kept] == [[1, 2, 3], [4, 5]]


def test_theme_left_too_small_by_overlap_is_dropped():
    kept, dropped = check_themes(
        [_theme("A", 1, 2), _theme("B", 2, 3)], {1, 2, 3}
    )
    assert [t.name for t in kept] == ["A"]
    assert "'B'" in dropped[0]


def test_at_most_five_themes():
    proposed = [_theme(f"T{i}", 2 * i, 2 * i + 1) for i in range(7)]
    kept, dropped = check_themes(proposed, set(range(14)))
    assert len(kept) == config.MAX_THEMES
    assert len(dropped) == 2


# --- Prompt input ---


def test_previous_names_and_summaries_reach_the_prompt():
    items = [_item(1, NEW), _item(2)]
    summaries = {
        1: {"status": "needs_triage", "where_it_stands": "No reply yet."},
        2: None,
    }
    text = render_issues(items, summaries, ["Streaming", "Export"])
    assert "Last run's theme names: Streaming; Export" in text
    assert "#1 [new] Issue 1 | needs_triage: No reply yet." in text
    assert f"#2 [newly active] Issue 2 | {config.FALLBACK_TEXT}" in text


# --- Model calls ---


def test_one_call_returns_checked_themes():
    result, guard = _find(
        [_reply(_theme("Streaming", 1, 2), _theme("Ghost", 3, 42))]
    )
    assert [t.name for t in result.themes] == ["Streaming"]
    assert len(result.dropped) == 1
    assert guard.calls == 1


def test_invalid_reply_twice_means_no_themes():
    result, guard = _find(["not json", '{"themes": "nope"}'])
    assert result.themes == [] and result.error is not None
    assert guard.calls == 2


def test_too_few_issues_skips_the_call():
    result, guard = _find([], items=[_item(1)])
    assert result.called is False and guard.calls == 0
