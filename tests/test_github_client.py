import datetime

import pytest

from monday_triage import config
from monday_triage.github_client import GitHubClient, GitHubError

SINCE = datetime.datetime(2026, 10, 17, tzinfo=datetime.timezone.utc)


class FakeResponse:
    def __init__(self, body, status_code=200, headers=None, next_url=None):
        self._body = body
        self.status_code = status_code
        self.headers = headers or {}
        self.links = {"next": {"url": next_url}} if next_url else {}
        self.text = str(body)

    def json(self):
        return self._body


class FakeSession:
    """Returns queued responses in order and records each request."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self._responses.pop(0)


def _issue(number, pull_request=False):
    item = {
        "number": number,
        "title": f"Issue {number}",
        "body": None,
        "user": {"login": "someone"},
        "state": "open",
        "created_at": "2026-10-18T09:00:00Z",
        "updated_at": "2026-10-20T09:00:00Z",
        "closed_at": None,
        "comments": 0,
        "html_url": f"https://github.com/o/r/issues/{number}",
    }
    if pull_request:
        item["pull_request"] = {"url": "..."}
    return item


def _search(*items, incomplete=False):
    return {
        "total_count": len(items),
        "incomplete_results": incomplete,
        "items": list(items),
    }


def test_search_query_asks_for_issues_only():
    session = FakeSession(FakeResponse(_search(_issue(1))))
    GitHubClient(session=session).list_changed_issues("o/r", SINCE)

    call = session.calls[0]
    assert call["url"].endswith("/search/issues")
    assert call["params"]["q"] == (
        "repo:o/r is:issue updated:>=2026-10-17T00:00:00Z"
    )
    assert call["params"]["sort"] == "updated"


def test_pull_requests_are_dropped_if_search_returns_any():
    session = FakeSession(
        FakeResponse(_search(_issue(1), _issue(2, pull_request=True)))
    )
    issues = GitHubClient(session=session).list_changed_issues("o/r", SINCE)
    assert [i.number for i in issues] == [1]


def test_follows_pagination():
    session = FakeSession(
        FakeResponse(_search(_issue(1)), next_url="https://next/page2"),
        FakeResponse(_search(_issue(2))),
    )
    client = GitHubClient(session=session)
    issues = client.list_changed_issues("o/r", SINCE)

    assert [i.number for i in issues] == [1, 2]
    assert session.calls[1]["url"] == "https://next/page2"
    assert session.calls[1]["params"] is None
    assert client.requests_made == 2


def test_too_many_pages_fails_loudly(monkeypatch):
    monkeypatch.setattr(config, "MAX_ISSUE_PAGES", 1)
    session = FakeSession(
        FakeResponse(_search(_issue(1)), next_url="https://next/page2")
    )
    with pytest.raises(GitHubError, match="Use a later --since"):
        GitHubClient(session=session).list_changed_issues("o/r", SINCE)


def test_partial_search_results_fail_loudly():
    session = FakeSession(FakeResponse(_search(_issue(1), incomplete=True)))
    with pytest.raises(GitHubError, match="partial results"):
        GitHubClient(session=session).list_changed_issues("o/r", SINCE)


def test_rate_limit_without_token_suggests_one():
    session = FakeSession(
        FakeResponse(
            {"message": "rate limited"},
            status_code=403,
            headers={"X-RateLimit-Remaining": "0"},
        )
    )
    with pytest.raises(GitHubError, match="Set GITHUB_TOKEN"):
        GitHubClient(session=session).list_changed_issues("o/r", SINCE)


def test_unknown_repo_gives_clear_error():
    session = FakeSession(FakeResponse({"message": "Validation"}, 422))
    with pytest.raises(GitHubError, match="Check the repo name"):
        GitHubClient(session=session).list_changed_issues("o/missing", SINCE)


def test_token_is_sent_as_bearer_header():
    session = FakeSession(FakeResponse(_search()))
    GitHubClient(token="t0ken", session=session).list_changed_issues(
        "o/r", SINCE
    )
    assert session.calls[0]["headers"]["Authorization"] == "Bearer t0ken"


def _xref(number, repo="o/r", pull=True, state="open", merged_at=None):
    issue = {
        "number": number,
        "title": f"PR {number}",
        "state": state,
        "repository": {"full_name": repo},
    }
    if pull:
        issue["pull_request"] = {"merged_at": merged_at}
    return {
        "event": "cross-referenced",
        "created_at": "2026-10-19T09:00:00Z",
        "source": {"type": "issue", "issue": issue},
    }


def test_linked_prs_come_from_same_repo_cross_references():
    events = [
        {"event": "commented", "created_at": "2026-10-18T09:00:00Z"},
        _xref(10),
        _xref(11, state="closed", merged_at="2026-10-20T09:00:00Z"),
        _xref(12, state="closed"),
        _xref(13, pull=False),  # an issue, not a PR
        _xref(14, repo="fork/r"),  # a PR in someone's fork
        _xref(10),  # referenced twice
    ]
    session = FakeSession(FakeResponse(events))
    prs = GitHubClient(session=session).get_linked_prs("o/r", 5)

    assert session.calls[0]["url"].endswith("/repos/o/r/issues/5/timeline")
    assert [(pr.number, pr.state) for pr in prs] == [
        (10, "open"),
        (11, "merged"),
        (12, "closed"),
    ]
