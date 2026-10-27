"""Scores thread status labels and themes against hand labels.

Runs the real pipeline (real LLM) on the frozen week named in the labels
file, never on live GitHub, so scores are comparable across runs. The model
isn't deterministic, so --repeats runs it several times and reports the
spread.

Usage (from the repo root):
    python eval/run_eval.py                       # current prompt, 1 run
    python eval/run_eval.py --repeats 3 --name baseline
    python eval/run_eval.py --repeats 3 --name closed-rule \
        --prompt-file eval/prompts/thread_state_closed_rule.md

Each run's full output (statuses, summaries, themes) is saved under
eval/results/ for spot-checking summaries for invented facts.
"""

import argparse
import collections
import datetime
import json
import pathlib
from typing import Any

import dotenv
from week import Week, load_week

from monday_triage import config
from monday_triage.budget import BudgetGuard
from monday_triage.delta import classify
from monday_triage.llm import build_chat_model
from monday_triage.state import Snapshot
from monday_triage.themes import find_themes
from monday_triage.thread_state import summarize_threads

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    """Runs the eval and prints the scores."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", default="eval/labels.json")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--prompt-file", help="thread_state prompt override")
    parser.add_argument("--name", default="run", help="name for the results")
    args = parser.parse_args()

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    label_data = json.loads((ROOT / args.labels).read_text(encoding="utf-8"))
    expected = {
        row["number"]: row["status"]
        for row in label_data["labels"]
        if row["status"] is not None
    }
    if not expected:
        raise SystemExit(f"No labels yet: fill in 'status' in {args.labels}.")
    unknown = {s for s in expected.values() if s not in config.STATUS_LABELS}
    if unknown:
        raise SystemExit(f"Unknown status labels in {args.labels}: {unknown}")

    week = load_week(ROOT / label_data["fixture"])
    prompt = (
        (ROOT / args.prompt_file).read_text(encoding="utf-8")
        if args.prompt_file
        else None
    )
    settings = config.load_settings(week.repo)
    model_summary = build_chat_model(settings, config.MAX_OUTPUT_TOKENS_SUMMARY)
    model_themes = build_chat_model(settings, config.MAX_OUTPUT_TOKENS_THEMES)

    runs = []
    for i in range(args.repeats):
        print(f"Run {i + 1}/{args.repeats}...", flush=True)
        runs.append(
            _run_once(week, settings, model_summary, model_themes, prompt)
        )

    report = _score(runs, expected, week)
    _print(report, args, settings.llm_model, len(expected), len(week.issues))
    _save(report, runs, args, settings.llm_model)


def _run_once(
    week: Week, settings, model_summary, model_themes, prompt: str | None
) -> dict[str, Any]:
    guard = BudgetGuard(settings.budget_usd, settings.price_per_mtok)
    items = classify(week.issues, Snapshot(repo=week.repo), week.since)
    results = summarize_threads(
        items,
        lambda number: week.comments[number],
        week.since,
        model_summary,
        guard,
        prompt,
        lambda number: week.linked_prs.get(number, []),
    )
    summaries = {n: r.summary for n, r in results.items()}
    themes = find_themes(items, summaries, [], model_themes, guard)
    return {
        "statuses": {n: (s or {}).get("status") for n, s in summaries.items()},
        "summaries": summaries,
        "themes": [
            {"name": t.name, "description": t.description, "issues": t.issues}
            for t in themes.themes
        ],
        "themes_dropped": themes.dropped,
        "coverage": sum(len(t.issues) for t in themes.themes) / len(items),
        "cost_usd": guard.spent_usd,
        "calls": guard.calls,
    }


def _score(
    runs: list[dict[str, Any]], expected: dict[int, str], week: Week
) -> dict[str, Any]:
    states = {issue.number: issue.state for issue in week.issues}
    accuracies = []
    confusion: collections.Counter = collections.Counter()
    answers: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for run in runs:
        correct = 0
        for number, want in expected.items():
            got = run["statuses"].get(number) or "fallback"
            answers[number][got] += 1
            if got == want:
                correct += 1
            else:
                confusion[(want, got)] += 1
        accuracies.append(correct / len(expected))
    disagreements = [
        {
            "number": number,
            "state": states.get(number),
            "expected": want,
            "model": dict(answers[number]),
        }
        for number, want in sorted(expected.items())
        if answers[number][want] < len(runs)
    ]
    return {
        "accuracy_per_run": accuracies,
        "accuracy_mean": sum(accuracies) / len(accuracies),
        "coverage_per_run": [run["coverage"] for run in runs],
        "cost_usd_total": sum(run["cost_usd"] for run in runs),
        "confusion": [
            {"expected": want, "model": got, "count": count}
            for (want, got), count in confusion.most_common()
        ],
        "disagreements": disagreements,
    }


def _print(report, args, model: str, labeled: int, total: int) -> None:
    acc = report["accuracy_mean"]
    cov = report["coverage_per_run"]
    verdict = "PASS" if acc >= config.STATUS_ACCURACY_TARGET else "FAIL"
    print(f"\n== {args.name}: {model}, {args.repeats} run(s) ==")
    print(f"Labeled issues: {labeled} of {total}")
    print(
        f"Status accuracy: mean {acc:.0%} "
        f"(runs: {', '.join(f'{a:.0%}' for a in report['accuracy_per_run'])})"
        f"  target {config.STATUS_ACCURACY_TARGET:.0%}  {verdict}"
    )
    print(
        f"Theme coverage: {', '.join(f'{c:.0%}' for c in cov)}  "
        "(share of issues in a theme; for information, no target)"
    )
    print(f"Cost: ${report['cost_usd_total']:.4f}")
    if report["confusion"]:
        print("\nMistakes (your label -> model):")
        for row in report["confusion"]:
            print(f"  {row['expected']} -> {row['model']}: {row['count']}")
    if report["disagreements"]:
        print("\nIssues the model got wrong at least once:")
        for row in report["disagreements"]:
            got = ", ".join(f"{k} x{v}" for k, v in row["model"].items())
            print(
                f"  #{row['number']} ({row['state']}): "
                f"you {row['expected']} | model {got}"
            )


def _save(report, runs, args, model: str) -> None:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = ROOT / "eval/results" / f"{stamp}_{args.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "name": args.name,
        "model": model,
        "prompt_file": args.prompt_file
        or "src/monday_triage/prompts/thread_state.md",
        "repeats": args.repeats,
        "report": report,
        "runs": runs,
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
