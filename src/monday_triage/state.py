"""Last-run snapshot: one JSON file per repo.

The snapshot remembers what the previous run saw, so the next run can tell
what changed and skip LLM calls for threads that didn't. It holds:

- when the last run happened (the default start of the next window),
- each seen issue's state, comment count, and linked PR states, plus its
  cached summary,
- last run's theme names (passed back to the LLM so names stay stable),
- the last report's data (reprinted when nothing changed).

A missing file means a first run. A file that can't be read stops the run
with a message, because a wrong snapshot would make every delta wrong.
"""

import dataclasses
import datetime
import json
import os
import pathlib
from typing import Any

from monday_triage import config


class StateError(RuntimeError):
    """The snapshot file can't be used. The message says how to recover."""


@dataclasses.dataclass
class IssueRecord:
    """What the last run knew about one issue.

    Attributes:
        state: "open" or "closed".
        comments_count: Comment count when last seen.
        linked_prs: Linked PR states when last seen, e.g. ["8688:open"].
        summary: Cached LLM summary (thread_state's JSON form), or None if
            there is none or it fell back.
    """

    state: str
    comments_count: int
    summary: dict[str, Any] | None = None
    linked_prs: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Snapshot:
    """Everything remembered about one repo between runs.

    Attributes:
        repo: Repository as "owner/name".
        last_run: When the last run finished, or None before the first run.
        issues: Seen issues by number.
        theme_names: Theme names from the last run.
        last_report: The last report's data, or None.
    """

    repo: str
    last_run: datetime.datetime | None = None
    issues: dict[int, IssueRecord] = dataclasses.field(default_factory=dict)
    theme_names: list[str] = dataclasses.field(default_factory=list)
    last_report: dict[str, Any] | None = None

    def cached_summary(
        self,
        number: int,
        state: str,
        comments_count: int,
        linked_prs: list[str],
    ) -> dict[str, Any] | None:
        """Returns the cached summary if the thread hasn't changed.

        The cache key is (comment count, open/closed state, linked PR
        states): a new comment, a close, a reopen, or a PR being opened,
        merged, or closed invalidates it; label or title edits don't.

        Args:
            number: Issue number.
            state: The issue's current state.
            comments_count: The issue's current comment count.
            linked_prs: The issue's linked PR states, as in IssueRecord.

        Returns:
            The cached summary, or None if it must be regenerated.
        """
        record = self.issues.get(number)
        if record is None or record.summary is None:
            return None
        if (record.state, record.comments_count, record.linked_prs) != (
            state,
            comments_count,
            linked_prs,
        ):
            return None
        return record.summary


def snapshot_path(state_dir: pathlib.Path, repo: str) -> pathlib.Path:
    """Returns the snapshot file for a repo, e.g. huggingface__datasets.json."""
    return state_dir / (repo.replace("/", "__") + ".json")


def load_snapshot(state_dir: pathlib.Path, repo: str) -> Snapshot:
    """Reads a repo's snapshot.

    Args:
        state_dir: Directory holding snapshots.
        repo: Repository as "owner/name".

    Returns:
        The saved snapshot, or an empty one if this repo was never run.

    Raises:
        StateError: If the file exists but can't be read or parsed.
    """
    path = snapshot_path(state_dir, repo)
    if not path.exists():
        return Snapshot(repo=repo)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _from_json(data, repo)
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise StateError(
            f"Can't read snapshot {path} ({e}). Delete it to start fresh; "
            "the next run will cover the last "
            f"{config.DEFAULT_LOOKBACK_DAYS} days."
        ) from e


def save_snapshot(state_dir: pathlib.Path, snapshot: Snapshot) -> None:
    """Writes a repo's snapshot atomically.

    The file is written to a temporary name and then renamed, so a crash
    mid-write never leaves a half-written snapshot behind.

    Args:
        state_dir: Directory holding snapshots; created if missing.
        snapshot: The snapshot to save.

    Raises:
        StateError: If the file can't be written.
    """
    path = snapshot_path(state_dir, snapshot.repo)
    tmp = path.with_suffix(".json.tmp")
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            json.dumps(_to_json(snapshot), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except OSError as e:
        raise StateError(f"Can't write snapshot {path}: {e}") from e


def _to_json(snapshot: Snapshot) -> dict[str, Any]:
    return {
        "version": config.STATE_VERSION,
        "repo": snapshot.repo,
        "last_run": (
            snapshot.last_run.isoformat() if snapshot.last_run else None
        ),
        "issues": {
            str(number): dataclasses.asdict(record)
            for number, record in snapshot.issues.items()
        },
        "theme_names": snapshot.theme_names,
        "last_report": snapshot.last_report,
    }


def _from_json(data: dict[str, Any], repo: str) -> Snapshot:
    if data.get("version") != config.STATE_VERSION:
        raise ValueError(
            f"version {data.get('version')} is not {config.STATE_VERSION}"
        )
    if data["repo"] != repo:
        raise ValueError(f"it belongs to {data['repo']}, not {repo}")
    last_run = data["last_run"]
    return Snapshot(
        repo=repo,
        last_run=(
            datetime.datetime.fromisoformat(last_run) if last_run else None
        ),
        issues={
            int(number): IssueRecord(**record)
            for number, record in data["issues"].items()
        },
        theme_names=list(data["theme_names"]),
        last_report=data["last_report"],
    )
