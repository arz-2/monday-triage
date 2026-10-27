import dataclasses
import datetime
import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from monday_triage import cli, config
from monday_triage.github_client import Comment, GitHubError, Issue, LinkedPR
from monday_triage.report import CACHED_WHAT_CHANGED

UTC = datetime.timezone.utc
NOW = datetime.datetime(2026, 10, 26, 9, tzinfo=UTC)
LATER = NOW + datetime.timedelta(days=7)

SUMMARY = json.dumps(
    {
        "status": "waiting_on_maintainer",
        "where_it_stands": "The reporter added a reproduction.",
        "what_changed": "The reporter posted a repro.",
    }
)
THEMES = json.dumps(
    {
        "themes": [
            {"name": "Casting", "description": "Cast bugs.", "issues": [1, 2]}
        ]
    }
)


def _issue(number, comments=1, updated=NOW, state="open"):
    return Issue(
        number=number,
        title=f"Issue {number}",
        body="It breaks.",
        author="reporter",
        state=state,
        created_at=datetime.datetime(2026, 9, 1, tzinfo=UTC),
        updated_at=updated - datetime.timedelta(hours=number),
        closed_at=None,
        comments_count=comments,
        url=f"https://github.com/o/r/issues/{number}",
    )


class FakeClient:
    """Stands in for GitHubClient; counts requests like the real one."""

    def __init__(self, issues, linked=None, error=None):
        self.issues = issues
        self.linked = linked or {}
        self.error = error
        self.requests_made = 0

    def list_changed_issues(self, repo, since):
        self.requests_made += 1
        if self.error:
            raise self.error
        return self.issues

    def get_comments(self, repo, number):
        self.requests_made += 1
        comment = Comment("user", "NONE", "Same here.", NOW)
        return [comment], False

    def get_linked_prs(self, repo, number):
        self.requests_made += 1
        return self.linked.get(number, [])


@pytest.fixture
def env(tmp_path, monkeypatch):
    # Run outside the repo so nothing can pick up its .env or state dir.
    monkeypatch.chdir(tmp_path)
    return {
        "LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "fake-key-for-tests",
        "GITHUB_TOKEN": "fake-token-for-tests",
        "MONDAY_TRIAGE_STATE_DIR": str(tmp_path / "state"),
    }


def _models(settings, max_tokens):
    if max_tokens == config.MAX_OUTPUT_TOKENS_THEMES:
        return FakeListChatModel(responses=[THEMES])
    return FakeListChatModel(responses=[SUMMARY])


def _run(env, client, *flags, now=NOW):
    return cli.run(
        ["o/r", "--format", "json", *flags],
        env=env,
        client=client,
        model_factory=_models,
        now=now,
    )


def _report(capsys):
    return json.loads(capsys.readouterr().out)


def test_first_run_summarizes_everything_and_saves_snapshot(
    env, capsys, tmp_path
):
    code = _run(env, FakeClient([_issue(1), _issue(2), _issue(3)]))
    report = _report(capsys)

    assert code == cli.EXIT_OK
    assert report["window"] == "last 7 days"
    assert report["run"]["llm_calls"] == 4  # 3 summaries + themes
    assert [t["name"] for t in report["themes"]] == ["Casting"]
    assert report["unthemed"] == [3]
    assert {e["source"] for e in report["issues"]} == {"new"}
    saved = json.loads((tmp_path / "state" / "o__r.json").read_text())
    assert saved["last_run"] == NOW.isoformat()


def test_rerun_with_no_changes_reprints_with_zero_llm_calls(env, capsys):
    _run(env, FakeClient([_issue(1), _issue(2)]))
    capsys.readouterr()

    # Nothing updated since the last run.
    assert _run(env, FakeClient([]), now=LATER) == cli.EXIT_OK
    report = _report(capsys)
    assert report["unchanged_since"] == NOW.isoformat()
    assert report["run"]["llm_calls"] == 0
    assert len(report["issues"]) == 2


def test_label_edit_only_counts_as_no_change(env, capsys):
    _run(env, FakeClient([_issue(1), _issue(2)]))
    capsys.readouterr()

    edited = [_issue(1, updated=LATER), _issue(2, updated=LATER)]
    _run(env, FakeClient(edited), now=LATER)
    assert _report(capsys)["run"]["llm_calls"] == 0


def test_rerun_text_says_nothing_changed(env, capsys):
    _run(env, FakeClient([_issue(1), _issue(2)]))
    capsys.readouterr()

    cli.run(
        ["o/r"],
        env=env,
        client=FakeClient([]),
        model_factory=_models,
        now=LATER,
    )
    out = capsys.readouterr().out
    assert out.startswith("No changes since 2026-10-26 09:00 UTC.")


def test_only_changed_threads_are_resummarized(env, capsys):
    _run(env, FakeClient([_issue(1), _issue(2)]))
    capsys.readouterr()

    _run(env, FakeClient([_issue(1, comments=2), _issue(2)]), now=LATER)
    report = _report(capsys)
    by_number = {e["number"]: e for e in report["issues"]}

    assert report["run"]["llm_calls"] == 2  # issue 1 + themes
    assert by_number[1]["source"] == "new"
    assert by_number[2]["source"] == "cached"
    assert by_number[2]["what_changed"] == CACHED_WHAT_CHANGED


def test_merged_linked_pr_makes_a_quiet_issue_newly_active(env, capsys):
    pr = LinkedPR(9, "Fix", "open", NOW)
    _run(env, FakeClient([_issue(1), _issue(2)], linked={1: [pr]}))
    capsys.readouterr()

    merged = {1: [dataclasses.replace(pr, state="merged")]}
    _run(env, FakeClient([_issue(1), _issue(2)], linked=merged), now=LATER)
    report = _report(capsys)
    by_number = {e["number"]: e for e in report["issues"]}

    assert by_number[1]["delta"] == "newly active"
    assert by_number[1]["source"] == "new"


def test_over_max_issues_are_listed_and_warned_loudly(env, capsys):
    issues = [_issue(1), _issue(2), _issue(3)]
    _run(env, FakeClient(issues), "--max-issues", "2")
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert len(report["issues"]) == 2
    assert [e["number"] for e in report["not_summarized"]] == [3]
    assert "use --max-issues 3" in report["warnings"][0]
    assert "use --max-issues 3" in captured.err

    cli.run(
        ["o/r", "--max-issues", "2", "--since", "2026-10-20"],
        env=env,
        client=FakeClient(issues),
        model_factory=_models,
        now=NOW,
    )
    out = capsys.readouterr().out
    # Above everything else, not only at the end.
    top = out.split("NEEDS YOUR ATTENTION")[0]
    assert "WARNING: 3 issues changed" in top


def test_missing_api_key_exits_with_usage_error(env, capsys):
    del env["ANTHROPIC_API_KEY"]
    assert _run(env, FakeClient([_issue(1)])) == cli.EXIT_USAGE
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


def test_tiny_budget_stops_and_saves_nothing(env, capsys, tmp_path):
    code = _run(
        env, FakeClient([_issue(1), _issue(2)]), "--budget-usd", "0.000001"
    )
    captured = capsys.readouterr()

    assert code == cli.EXIT_ERROR
    assert "BudgetExceeded" in captured.err
    assert captured.out == ""
    assert not (tmp_path / "state" / "o__r.json").exists()


def test_github_rate_limit_exits_with_clear_error(env, capsys, tmp_path):
    error = GitHubError("GitHub rate limit reached; it resets at 10:00.")
    assert _run(env, FakeClient([], error=error)) == cli.EXIT_ERROR
    assert "rate limit reached" in capsys.readouterr().err
    assert not (tmp_path / "state" / "o__r.json").exists()


def test_json_output_stays_clean_without_github_token(env, capsys):
    del env["GITHUB_TOKEN"]
    _run(env, FakeClient([_issue(1)]))
    captured = capsys.readouterr()

    assert "GITHUB_TOKEN is not set" in captured.err
    assert json.loads(captured.out)["repo"] == "o/r"


def test_attention_section_splits_by_what_to_do(env, capsys):
    pr = LinkedPR(9, "Fix", "open", NOW)
    client = FakeClient([_issue(1), _issue(2)], linked={1: [pr]})
    cli.run(["o/r"], env=env, client=client, model_factory=_models, now=NOW)
    out = capsys.readouterr().out

    attention = out.split("NEEDS YOUR ATTENTION (2)")[1].split("THEMES")[0]
    assert "PR ready to review (1)" in attention
    assert "#1 [newly active] Issue 1 -> PR #9" in attention
    assert "Reply waiting (1)" in attention
    assert out.index("NEEDS YOUR ATTENTION") < out.index("THEMES")


def test_failed_summary_is_flagged_for_a_manual_check(env, capsys):
    def models(settings, max_tokens):
        return FakeListChatModel(responses=["not json"])

    cli.run(
        ["o/r", "--format", "json"],
        env=env,
        client=FakeClient([_issue(1)]),
        model_factory=models,
        now=NOW,
    )
    (entry,) = _report(capsys)["issues"]
    assert entry["attention"] == "Summary unavailable, check by hand"
