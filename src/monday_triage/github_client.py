"""Read-only GitHub REST client for issues, comments, and linked PRs.

Three calls are needed: search for the issues (not PRs) updated since a time,
fetch one issue's comments, and fetch the pull requests linked to it (from the
issue's timeline; a PR linked only there never shows up in the comments). All
follow pagination up to a page cap and turn every failure into a
`GitHubError` whose message says what went wrong and what to do.
"""

import dataclasses
import datetime
import threading
from typing import Any

import requests

from monday_triage import config


class GitHubError(RuntimeError):
    """A GitHub request failed. The message says why and how to fix it."""


@dataclasses.dataclass(frozen=True)
class Issue:
    """One issue as the pipeline needs it.

    Attributes:
        number: Issue number.
        title: Issue title.
        body: Opening post text ("" if empty).
        author: Login of the person who opened it.
        state: "open" or "closed".
        created_at: When it was opened.
        updated_at: Last activity of any kind (comment, label, edit).
        closed_at: When it was closed, or None if open.
        comments_count: Number of comments, not counting the opening post.
        url: Link to the issue on github.com.
    """

    number: int
    title: str
    body: str
    author: str
    state: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    closed_at: datetime.datetime | None
    comments_count: int
    url: str


@dataclasses.dataclass(frozen=True)
class Comment:
    """One comment on an issue.

    Attributes:
        author: Commenter's login.
        author_association: GitHub role such as OWNER, MEMBER, or NONE.
        body: Comment text ("" if empty).
        created_at: When it was posted.
    """

    author: str
    author_association: str
    body: str
    created_at: datetime.datetime


@dataclasses.dataclass(frozen=True)
class LinkedPR:
    """A pull request in the same repo that references an issue.

    Attributes:
        number: Pull request number.
        title: Pull request title.
        state: "open", "merged", or "closed" (closed without merging).
        linked_at: When the PR first referenced the issue.
    """

    number: int
    title: str
    state: str
    linked_at: datetime.datetime


class GitHubClient:
    """Fetches issues and comments from the GitHub REST API.

    Attributes:
        requests_made: Number of HTTP requests sent so far, for the run log.
    """

    def __init__(
        self,
        token: str | None = None,
        session: requests.Session | None = None,
        base_url: str = config.GITHUB_API_URL,
    ) -> None:
        """Creates a client.

        Args:
            token: Optional GitHub token. Without one, GitHub allows 60
                requests an hour.
            session: HTTP session to use; tests pass a fake one.
            base_url: API root, overridable for tests.
        """
        self._session = session or requests.Session()
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "monday-triage",
        }
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        # Comment fetches run from several threads at once.
        self._lock = threading.Lock()
        self.requests_made = 0

    def list_changed_issues(
        self, repo: str, since: datetime.datetime
    ) -> list[Issue]:
        """Lists issues (open or closed) updated at or after `since`.

        Args:
            repo: Repository as "owner/name".
            since: Start of the window (timezone-aware).

        Uses the Search API with `is:issue`, so GitHub filters out pull
        requests before anything is downloaded. (The plain issues endpoint
        returns PRs too, which wastes requests and the page cap.)

        Returns:
            Issues, most recently updated first. Pull requests are excluded.

        Raises:
            GitHubError: On any failed request, if more issues changed than
                MAX_ISSUE_PAGES pages can hold, or if GitHub's search
                returned partial results.
        """
        params = {
            "q": f"repo:{repo} is:issue updated:>={_format_time(since)}",
            "sort": "updated",
            "order": "desc",
            "per_page": config.GITHUB_PER_PAGE,
        }
        items, truncated = self._get_pages(
            "/search/issues", params, config.MAX_ISSUE_PAGES, key="items"
        )
        if truncated:
            raise GitHubError(
                f"More than {config.MAX_ISSUE_PAGES * config.GITHUB_PER_PAGE} "
                f"issues changed in {repo} since {since:%Y-%m-%d}. "
                "Use a later --since."
            )
        # Safety net: `is:issue` should already exclude every PR.
        return [
            _parse_issue(item) for item in items if "pull_request" not in item
        ]

    def get_comments(
        self, repo: str, number: int
    ) -> tuple[list[Comment], bool]:
        """Fetches an issue's comments, oldest first.

        Args:
            repo: Repository as "owner/name".
            number: Issue number.

        Returns:
            The comments, and True if the thread was longer than
            MAX_COMMENT_PAGES pages and was cut short.

        Raises:
            GitHubError: On any failed request.
        """
        items, truncated = self._get_pages(
            f"/repos/{repo}/issues/{number}/comments",
            {"per_page": config.GITHUB_PER_PAGE},
            config.MAX_COMMENT_PAGES,
        )
        return [_parse_comment(item) for item in items], truncated

    def get_linked_prs(self, repo: str, number: int) -> list[LinkedPR]:
        """Fetches pull requests in `repo` that reference an issue.

        Reads the issue's timeline for cross-reference events. PRs from
        forks or other repos are skipped.

        Args:
            repo: Repository as "owner/name".
            number: Issue number.

        Returns:
            Linked PRs in the order they were first linked, one per PR.

        Raises:
            GitHubError: On any failed request.
        """
        events, _ = self._get_pages(
            f"/repos/{repo}/issues/{number}/timeline",
            {"per_page": config.GITHUB_PER_PAGE},
            config.MAX_TIMELINE_PAGES,
        )
        linked: dict[int, LinkedPR] = {}
        for event in events:
            pr = _parse_linked_pr(event, repo)
            if pr is not None and pr.number not in linked:
                linked[pr.number] = pr
        return list(linked.values())

    def _get_pages(
        self,
        path: str,
        params: dict[str, Any] | None,
        max_pages: int,
        key: str | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Follows `Link: rel="next"` up to `max_pages` pages.

        Args:
            path: API path under the base URL.
            params: Query parameters for the first page.
            max_pages: Most pages to fetch.
            key: For endpoints that wrap results in an object (search), the
                field holding the list; None when the body is the list.

        Returns:
            All items fetched, and True if more pages were left unfetched.

        Raises:
            GitHubError: On any failed request, or partial search results.
        """
        url: str | None = self._base_url + path
        items: list[dict[str, Any]] = []
        for _ in range(max_pages):
            response = self._get(url, params)
            body = response.json()
            if key is None:
                items.extend(body)
            else:
                if body.get("incomplete_results"):
                    raise GitHubError(
                        "GitHub search timed out and returned partial "
                        "results. Try again in a minute."
                    )
                items.extend(body[key])
            url = response.links.get("next", {}).get("url")
            params = None  # The next-page URL already carries the query.
            if url is None:
                return items, False
        return items, True

    def _get(
        self, url: str, params: dict[str, Any] | None
    ) -> requests.Response:
        """Sends one GET and raises a clear GitHubError on failure."""
        with self._lock:
            self.requests_made += 1
        try:
            response = self._session.get(
                url,
                params=params,
                headers=self._headers,
                timeout=config.GITHUB_TIMEOUT_S,
            )
        except requests.RequestException as e:
            raise GitHubError(f"Could not reach GitHub: {e}") from e

        if response.status_code == 200:
            return response
        if response.status_code in (403, 429) and _rate_limited(response):
            reset = response.headers.get("X-RateLimit-Reset")
            when = (
                datetime.datetime.fromtimestamp(int(reset)).strftime("%H:%M")
                if reset
                else "later"
            )
            hint = (
                ""
                if "Authorization" in self._headers
                else " Set GITHUB_TOKEN to raise the limit to 5,000 an hour."
            )
            raise GitHubError(
                f"GitHub rate limit reached; it resets at {when}.{hint}"
            )
        if response.status_code == 401:
            raise GitHubError("GITHUB_TOKEN was rejected. Check or remove it.")
        # Search answers 422 when the repo in the query doesn't exist.
        if response.status_code in (404, 422):
            raise GitHubError(
                f"Not found: {url}. Check the repo name and that it is public."
            )
        raise GitHubError(
            f"GitHub returned {response.status_code} for {url}: "
            f"{response.text[:200]}"
        )


def _rate_limited(response: requests.Response) -> bool:
    return response.headers.get(
        "X-RateLimit-Remaining"
    ) == "0" or "retry-after" in {k.lower() for k in response.headers}


def _format_time(moment: datetime.datetime) -> str:
    utc = moment.astimezone(datetime.timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time(value: str) -> datetime.datetime:
    # Python 3.10's fromisoformat does not accept a trailing "Z".
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_issue(item: dict[str, Any]) -> Issue:
    closed_at = item.get("closed_at")
    return Issue(
        number=item["number"],
        title=item["title"],
        body=item.get("body") or "",
        author=(item.get("user") or {}).get("login", "ghost"),
        state=item["state"],
        created_at=_parse_time(item["created_at"]),
        updated_at=_parse_time(item["updated_at"]),
        closed_at=_parse_time(closed_at) if closed_at else None,
        comments_count=item.get("comments", 0),
        url=item["html_url"],
    )


def _parse_comment(item: dict[str, Any]) -> Comment:
    return Comment(
        author=(item.get("user") or {}).get("login", "ghost"),
        author_association=item.get("author_association", "NONE"),
        body=item.get("body") or "",
        created_at=_parse_time(item["created_at"]),
    )


def _parse_linked_pr(event: dict[str, Any], repo: str) -> LinkedPR | None:
    if event.get("event") != "cross-referenced":
        return None
    source = (event.get("source") or {}).get("issue") or {}
    pull = source.get("pull_request")
    same_repo = (source.get("repository") or {}).get("full_name") == repo
    if pull is None or not same_repo:
        return None
    if pull.get("merged_at"):
        state = "merged"
    else:
        state = source.get("state", "open")
    return LinkedPR(
        number=source["number"],
        title=source.get("title", ""),
        state=state,
        linked_at=_parse_time(event["created_at"]),
    )
