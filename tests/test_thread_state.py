import dataclasses
import datetime
import importlib.resources
import json

import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage

from monday_triage import config
from monday_triage.budget import BudgetExceeded, BudgetGuard, estimate_tokens
from monday_triage.delta import NEWLY_ACTIVE, DeltaItem
from monday_triage.github_client import Comment, Issue, LinkedPR
from monday_triage.llm import parse_json
from monday_triage.thread_state import (
    ThreadSummary,
    render_thread,
    summarize_thread,
    summarize_threads,
    trim_thread,
)

UTC = datetime.timezone.utc
SINCE = datetime.datetime(2026, 10, 19, tzinfo=UTC)
START = datetime.datetime(2026, 9, 1, tzinfo=UTC)

VALID = json.dumps(
    {
        "status": "waiting_on_reporter",
        "where_it_stands": "A maintainer asked for a minimal reproduction.",
        "what_changed": "The maintainer asked for a repro on Tuesday.",
    }
)


def _issue(body="Opening post", comments=0):
    return Issue(
        number=7,
        title="Crash when loading",
        body=body,
        author="reporter",
        state="open",
        created_at=START,
        updated_at=SINCE,
        closed_at=None,
        comments_count=comments,
        url="",
    )


def _comment(i, role="NONE", body=None, day=None):
    created = day or START + datetime.timedelta(days=i)
    return Comment(
        author=f"user{i}",
        author_association=role,
        body=body if body is not None else f"Comment {i}",
        created_at=created,
    )


def _positions(thread):
    return [post.position for post in thread.posts]


def _guard(budget=1.0):
    return BudgetGuard(budget, (0.10, 0.50))


# --- Trimming ---


def test_short_thread_is_kept_whole():
    comments = [_comment(i) for i in range(1, 4)]
    thread = trim_thread(_issue(), comments, SINCE)
    assert _positions(thread) == [0, 1, 2, 3]


def test_long_thread_keeps_first_last_five_and_key_comments():
    comments = [_comment(i) for i in range(1, 41)]
    comments[9] = _comment(10, role="MEMBER")
    comments[19] = _comment(20, body="Fixed in #7012, please retest.")
    thread = trim_thread(_issue(), comments, SINCE)
    assert _positions(thread) == [0, 10, 20, 36, 37, 38, 39, 40]
    assert thread.total_comments == 40


@pytest.mark.parametrize("role", sorted(config.MAINTAINER_ROLES))
def test_every_maintainer_role_is_kept(role):
    comments = [_comment(i) for i in range(1, 11)]
    comments[1] = _comment(2, role=role)
    assert 2 in _positions(trim_thread(_issue(), comments, SINCE))


@pytest.mark.parametrize(
    "body, kept",
    [
        ("Fixed in #7012", True),
        ("See https://github.com/o/r/pull/123", True),
        ("Reverted in a1b2c3d", True),
        ("I see this on issue twelve too", False),
        ("Use the #general channel", False),
        ("Same here, version 2.1.0", False),
    ],
)
def test_fix_link_pattern(body, kept):
    comments = [_comment(i) for i in range(1, 11)]
    comments[1] = _comment(2, body=body)
    assert (2 in _positions(trim_thread(_issue(), comments, SINCE))) is kept


def test_key_comment_in_last_five_is_not_duplicated():
    comments = [_comment(i) for i in range(1, 11)]
    comments[8] = _comment(9, role="OWNER")
    assert _positions(trim_thread(_issue(), comments, SINCE)) == [
        0,
        6,
        7,
        8,
        9,
        10,
    ]


def test_oversized_thread_fits_cap_and_keeps_newest_comment():
    huge = "x" * (config.MAX_TOKENS_PER_ISSUE * config.CHARS_PER_TOKEN)
    comments = [_comment(i, body=huge) for i in range(1, 7)]
    thread = trim_thread(_issue(body=huge), comments, SINCE)

    total = sum(estimate_tokens(post.body) for post in thread.posts)
    assert total <= config.MAX_TOKENS_PER_ISSUE
    assert _positions(thread)[-1] == 6
    assert thread.posts[0].body.endswith("[...truncated]")


def test_render_marks_new_comments_and_gaps():
    comments = [_comment(i) for i in range(1, 11)]
    comments[-1] = _comment(10, day=SINCE + datetime.timedelta(days=1))
    item = DeltaItem(_issue(comments=10), NEWLY_ACTIVE)
    text = render_thread(item, trim_thread(item.issue, comments, SINCE), SINCE)

    assert "[5 comments omitted]" in text
    assert text.count("NEW THIS WINDOW") == 1
    assert "<thread>" in text and "</thread>" in text


def test_render_lists_linked_prs_inside_thread_tags():
    item = DeltaItem(_issue(), NEWLY_ACTIVE)
    prs = [
        LinkedPR(8, "Old attempt", "closed", START),
        LinkedPR(9, "Fix crash", "open", SINCE),
    ]
    text = render_thread(item, trim_thread(item.issue, [], SINCE), SINCE, prs)

    block = text.split("<thread>")[1]
    assert "- PR #8 (closed, not merged): Old attempt\n" in block
    assert "- PR #9 (open): Fix crash · NEW THIS WINDOW" in block


def test_render_says_when_no_prs_are_linked():
    item = DeltaItem(_issue(), NEWLY_ACTIVE)
    text = render_thread(item, trim_thread(item.issue, [], SINCE), SINCE)
    assert "Linked pull requests: none" in text


def test_prompt_defines_every_status_label():
    prompt = (
        importlib.resources.files("monday_triage")
        .joinpath("prompts/thread_state.md")
        .read_text(encoding="utf-8")
    )
    for label in config.STATUS_LABELS:
        assert f"- {label}:" in prompt


# --- Parsing ---


def test_parse_accepts_code_fence():
    summary = parse_json(f"```json\n{VALID}\n```", ThreadSummary)
    assert summary.status == "waiting_on_reporter"


@pytest.mark.parametrize(
    "reply",
    [
        "Sorry, I can't tell.",
        '{"status": "waiting_on_reporter"',
        VALID.replace("waiting_on_reporter", "blocked"),
        VALID.replace("A maintainer asked for a minimal reproduction.", ""),
    ],
)
def test_parse_rejects_bad_replies(reply):
    with pytest.raises(ValueError):
        parse_json(reply, ThreadSummary)


# --- Model calls ---


def _summarize(responses, guard=None):
    model = FakeListChatModel(responses=responses)
    guard = guard or _guard()
    item = DeltaItem(_issue(comments=2), NEWLY_ACTIVE)
    comments = [_comment(1), _comment(2, role="MEMBER")]
    return summarize_thread(item, comments, SINCE, model, guard), guard


def test_valid_reply_is_parsed():
    result, guard = _summarize([VALID])
    assert result.summary["status"] == "waiting_on_reporter"
    assert result.error is None
    assert guard.calls == 1


def test_thinking_blocks_in_reply_are_ignored():
    reply = AIMessage(
        content=[
            {"type": "thinking", "thinking": "Let me see.", "signature": "s"},
            {"type": "text", "text": VALID},
        ]
    )
    model = GenericFakeChatModel(messages=iter([reply]))
    item = DeltaItem(_issue(comments=1), NEWLY_ACTIVE)
    result = summarize_thread(item, [_comment(1)], SINCE, model, _guard())
    assert result.summary["status"] == "waiting_on_reporter"


def test_bad_reply_is_retried_once_then_succeeds():
    result, guard = _summarize(["not json", VALID])
    assert result.summary is not None
    assert guard.calls == 2


def test_two_bad_replies_fall_back():
    result, guard = _summarize(["not json", '{"status": "blocked"}'])
    assert result.summary is None
    assert "status" in result.error
    assert guard.calls == 2


def test_budget_stops_before_the_call():
    with pytest.raises(BudgetExceeded, match="--budget-usd"):
        _summarize([VALID], guard=_guard(budget=0.000001))


def test_summarize_threads_runs_every_item():
    items = [
        DeltaItem(dataclasses.replace(_issue(), number=n), NEWLY_ACTIVE)
        for n in range(1, 9)
    ]
    model = FakeListChatModel(responses=[VALID])
    guard = _guard()
    results = summarize_threads(items, lambda n: [], SINCE, model, guard)
    assert sorted(results) == list(range(1, 9))
    assert all(r.summary is not None for r in results.values())
    assert guard.calls == 8


def test_summarize_threads_stops_on_budget():
    items = [
        DeltaItem(dataclasses.replace(_issue(), number=n), NEWLY_ACTIVE)
        for n in range(1, 9)
    ]
    model = FakeListChatModel(responses=[VALID])
    with pytest.raises(BudgetExceeded):
        summarize_threads(items, lambda n: [], SINCE, model, _guard(1e-6))
