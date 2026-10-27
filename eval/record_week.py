"""Records one week of a repo's changed issues for the eval and tests.

Usage (from the repo root):
    python eval/record_week.py huggingface/datasets --since 2026-10-18

Writes tests/fixtures/<owner>__<name>_week.json and, if it doesn't exist yet,
a label template at eval/labels.json. Existing labels are never overwritten.
"""

import argparse
import datetime
import json
import os
import pathlib

import dotenv
from week import Week, save_week

from monday_triage import config
from monday_triage.github_client import GitHubClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
LABEL_HELP = {
    "needs_triage": "no maintainer has replied yet and no PR is open",
    "waiting_on_maintainer": "maintainer's move: a linked PR is open and "
    "awaits review/merge, or the reporter or a user replied last",
    "waiting_on_reporter": "a maintainer asked for info/repro; reporter's move",
    "fix_in_progress": "someone said they are working on a fix, but no PR "
    "is open yet",
    "resolved": "fixed (e.g. linked PR merged), answered, duplicate, or "
    "otherwise done (even if open)",
}


def main() -> None:
    """Records the week and writes the label template."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo")
    parser.add_argument("--since", required=True, help="YYYY-MM-DD (UTC)")
    args = parser.parse_args()
    if set(LABEL_HELP) != set(config.STATUS_LABELS):
        raise SystemExit("LABEL_HELP is out of sync with STATUS_LABELS.")

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    since = datetime.datetime.combine(
        datetime.date.fromisoformat(args.since),
        datetime.time(),
        tzinfo=datetime.timezone.utc,
    )
    client = GitHubClient(token=os.environ.get("GITHUB_TOKEN") or None)
    issues = client.list_changed_issues(args.repo, since)
    comments, linked_prs = {}, {}
    for issue in issues:
        thread, truncated = client.get_comments(args.repo, issue.number)
        if truncated:
            print(f"warning: #{issue.number} has over 500 comments; cut")
        comments[issue.number] = thread
        linked_prs[issue.number] = client.get_linked_prs(
            args.repo, issue.number
        )

    week = Week(
        repo=args.repo,
        since=since,
        recorded_at=datetime.datetime.now(datetime.timezone.utc),
        issues=issues,
        comments=comments,
        linked_prs=linked_prs,
    )
    fixture = (
        ROOT / "tests/fixtures" / (args.repo.replace("/", "__") + "_week.json")
    )
    save_week(week, fixture)
    print(
        f"Recorded {len(issues)} issues ({client.requests_made} requests) "
        f"to {fixture.relative_to(ROOT)}"
    )

    labels = ROOT / "eval/labels.json"
    if labels.exists():
        print(f"{labels.relative_to(ROOT)} exists; not overwritten.")
        return
    template = {
        "fixture": str(fixture.relative_to(ROOT)),
        "instructions": (
            "For each issue, read the thread on GitHub as it was at "
            f"{week.recorded_at:%Y-%m-%d %H:%M} UTC and set 'status' to one "
            "label from 'status_labels'. Label before looking at any model "
            "output. Leave status null to skip an issue."
        ),
        "status_labels": LABEL_HELP,
        "labels": [
            {
                "number": issue.number,
                "title": issue.title,
                "url": issue.url,
                "state": issue.state,
                "status": None,
                "note": "",
            }
            for issue in issues
        ],
    }
    labels.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote label template {labels.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
