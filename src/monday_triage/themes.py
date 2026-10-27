"""Groups this week's changed issues into a few themes, in one LLM call.

The model sees each issue's title, delta label, and thread summary, plus last
run's theme names so it can reuse them and names stay stable week to week.

Each proposed theme is checked on its own, and bad ones are dropped while
the rest are kept:

- every cited issue must be in this week's list (no invented numbers),
- an issue already in an earlier theme is removed from later ones,
- a theme needs at least MIN_ISSUES_PER_THEME issues after that,
- only the first MAX_THEMES themes are kept.

If the whole reply is invalid after one retry, there are no themes and the
report says so.
"""

import dataclasses
import importlib.resources

import pydantic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from monday_triage import config
from monday_triage.budget import BudgetGuard
from monday_triage.delta import DeltaItem
from monday_triage.llm import invoke_json


class ProposedTheme(pydantic.BaseModel):
    """One theme as the model proposes it, before the checks."""

    name: str = pydantic.Field(
        min_length=1, max_length=config.MAX_THEME_NAME_CHARS
    )
    description: str = pydantic.Field(
        min_length=1, max_length=config.MAX_SUMMARY_FIELD_CHARS
    )
    issues: list[int]


class ThemesReply(pydantic.BaseModel):
    """The model's whole reply."""

    themes: list[ProposedTheme]


@dataclasses.dataclass(frozen=True)
class Theme:
    """A theme that passed every check.

    Attributes:
        name: Short name, reused from last run when it matches.
        description: One sentence on what the issues share.
        issues: Issue numbers, each in no other theme.
    """

    name: str
    description: str
    issues: list[int]


@dataclasses.dataclass(frozen=True)
class ThemesResult:
    """Outcome of the themes step.

    Attributes:
        themes: Themes that passed the checks.
        dropped: Why each dropped theme was dropped, e.g. "'X': cites #9".
        error: Why the whole reply was unusable, or None.
        called: False if there were too few issues to need a call.
    """

    themes: list[Theme]
    dropped: list[str] = dataclasses.field(default_factory=list)
    error: str | None = None
    called: bool = True


def find_themes(
    items: list[DeltaItem],
    summaries: dict[int, dict[str, str] | None],
    previous_names: list[str],
    model: BaseChatModel,
    guard: BudgetGuard,
) -> ThemesResult:
    """Asks the model to group this week's issues, then checks each theme.

    Args:
        items: This week's changed issues.
        summaries: Thread summary per issue number (None if it fell back).
        previous_names: Theme names from the last run.
        model: Chat model, already capped at MAX_OUTPUT_TOKENS_THEMES.
        guard: Budget guard, checked before every call.

    Returns:
        The checked themes, with notes on anything dropped.

    Raises:
        BudgetExceeded: If the call could pass a cap.
    """
    if len(items) < config.MIN_ISSUES_PER_THEME:
        return ThemesResult(themes=[], called=False)
    messages = [
        SystemMessage(_system_prompt()),
        HumanMessage(render_issues(items, summaries, previous_names)),
    ]
    reply, error = invoke_json(
        model, messages, ThemesReply, guard, config.MAX_OUTPUT_TOKENS_THEMES
    )
    if reply is None:
        return ThemesResult(themes=[], error=error)
    themes, dropped = check_themes(
        reply.themes, {item.issue.number for item in items}
    )
    return ThemesResult(themes=themes, dropped=dropped)


def check_themes(
    proposed: list[ProposedTheme], known: set[int]
) -> tuple[list[Theme], list[str]]:
    """Keeps the proposed themes that pass every check, in order.

    Args:
        proposed: Themes as the model returned them.
        known: Issue numbers that were in the input.

    Returns:
        The kept themes, and one reason per dropped theme.
    """
    kept: list[Theme] = []
    dropped: list[str] = []
    assigned: set[int] = set()
    for theme in proposed:
        unknown = sorted(set(theme.issues) - known)
        if len(kept) == config.MAX_THEMES:
            dropped.append(
                f"'{theme.name}': over the {config.MAX_THEMES} limit"
            )
            continue
        if unknown:
            cited = ", ".join(f"#{n}" for n in unknown)
            dropped.append(f"'{theme.name}': cites {cited}, not in this week")
            continue
        members = [n for n in dict.fromkeys(theme.issues) if n not in assigned]
        if len(members) < config.MIN_ISSUES_PER_THEME:
            dropped.append(
                f"'{theme.name}': fewer than {config.MIN_ISSUES_PER_THEME} "
                "issues not already in another theme"
            )
            continue
        assigned.update(members)
        kept.append(Theme(theme.name, theme.description, members))
    return kept, dropped


def render_issues(
    items: list[DeltaItem],
    summaries: dict[int, dict[str, str] | None],
    previous_names: list[str],
) -> str:
    """Formats the issue list and last run's names as the user message.

    Args:
        items: This week's changed issues.
        summaries: Thread summary per issue number (None if it fell back).
        previous_names: Theme names from the last run.

    Returns:
        The message text; issue content sits inside <issues> tags.
    """
    names = "; ".join(previous_names) if previous_names else "(none)"
    lines = [f"Last run's theme names: {names}", "", "<issues>"]
    for item in items:
        summary = summaries.get(item.issue.number)
        state = (
            f"{summary['status']}: {summary['where_it_stands']}"
            if summary
            else config.FALLBACK_TEXT
        )
        lines.append(
            f"#{item.issue.number} [{item.label}] {item.issue.title} | {state}"
        )
    lines.append("</issues>")
    return "\n".join(lines)


def _system_prompt() -> str:
    path = importlib.resources.files("monday_triage") / "prompts"
    return (path / "themes.md").read_text(encoding="utf-8")
