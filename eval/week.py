"""Saves and loads a frozen week of issues and comments as JSON.

A frozen week lets the eval and the tests run on exactly the same threads
every time, with no network calls and no answers changing as new comments
arrive.
"""

import dataclasses
import datetime
import json
import pathlib
from typing import Any

from monday_triage.github_client import Comment, Issue, LinkedPR


@dataclasses.dataclass(frozen=True)
class Week:
    """One recorded window of a repo.

    Attributes:
        repo: Repository as "owner/name".
        since: Window start.
        recorded_at: When it was recorded.
        issues: Issues updated in the window, most recent first.
        comments: Each issue's full comment list, oldest first.
        linked_prs: Each issue's linked pull requests. Empty for weeks
            recorded before linked PRs were fetched.
    """

    repo: str
    since: datetime.datetime
    recorded_at: datetime.datetime
    issues: list[Issue]
    comments: dict[int, list[Comment]]
    linked_prs: dict[int, list[LinkedPR]] = dataclasses.field(
        default_factory=dict
    )


def save_week(week: Week, path: pathlib.Path) -> None:
    """Writes a week to JSON."""
    data = {
        "repo": week.repo,
        "since": week.since.isoformat(),
        "recorded_at": week.recorded_at.isoformat(),
        "issues": [_dump(issue) for issue in week.issues],
        "comments": {
            str(number): [_dump(c) for c in comments]
            for number, comments in week.comments.items()
        },
        "linked_prs": {
            str(number): [_dump(pr) for pr in prs]
            for number, prs in week.linked_prs.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_week(path: pathlib.Path) -> Week:
    """Reads a week written by `save_week`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return Week(
        repo=data["repo"],
        since=datetime.datetime.fromisoformat(data["since"]),
        recorded_at=datetime.datetime.fromisoformat(data["recorded_at"]),
        issues=[_load(Issue, item) for item in data["issues"]],
        comments={
            int(number): [_load(Comment, c) for c in comments]
            for number, comments in data["comments"].items()
        },
        linked_prs={
            int(number): [_load(LinkedPR, pr) for pr in prs]
            for number, prs in data.get("linked_prs", {}).items()
        },
    )


def _dump(obj: Any) -> dict[str, Any]:
    return {
        key: value.isoformat()
        if isinstance(value, datetime.datetime)
        else value
        for key, value in dataclasses.asdict(obj).items()
    }


def _load(cls: type, data: dict[str, Any]) -> Any:
    fields = {}
    for field in dataclasses.fields(cls):
        value = data[field.name]
        if field.name.endswith("_at") and value is not None:
            value = datetime.datetime.fromisoformat(value)
        fields[field.name] = value
    return cls(**fields)
