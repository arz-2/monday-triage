"""Reads one changed thread and says where it stands.

Two halves:

- `trim_thread` (plain code) picks what the LLM sees: the opening post, the
  last KEEP_LAST_N_COMMENTS comments, and any earlier comment by a maintainer
  or linking to a fix. If that still exceeds MAX_TOKENS_PER_ISSUE, the opening
  post is cut to FIRST_POST_MAX_TOKENS and the newest kept comments fill the
  rest.
- `render_thread` adds the PRs linked to the issue (from its timeline), so the
  model can see a fix that is never mentioned in the comments.
- `summarize_thread` sends it to the model through `llm.invoke_json`, which
  checks the reply against `ThreadSummary` and retries once; a reply that is
  still invalid gives no summary (the report shows the fallback text).
"""

import concurrent.futures
import dataclasses
import datetime
import importlib.resources
from collections.abc import Callable, Sequence

import pydantic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from monday_triage import config
from monday_triage.budget import BudgetGuard, estimate_tokens
from monday_triage.delta import DeltaItem
from monday_triage.github_client import Comment, Issue, LinkedPR
from monday_triage.llm import invoke_json

_TRUNCATED = " [...truncated]"


class ThreadSummary(pydantic.BaseModel):
    """The LLM's answer for one thread.

    Attributes:
        status: One of config.STATUS_LABELS.
        where_it_stands: The issue's current state, one or two sentences.
        what_changed: What happened inside the window.
    """

    status: str
    where_it_stands: str = pydantic.Field(
        min_length=1, max_length=config.MAX_SUMMARY_FIELD_CHARS
    )
    what_changed: str = pydantic.Field(
        min_length=1, max_length=config.MAX_SUMMARY_FIELD_CHARS
    )

    @pydantic.field_validator("status")
    @classmethod
    def _known_status(cls, value: str) -> str:
        if value not in config.STATUS_LABELS:
            raise ValueError(
                f"status must be one of {', '.join(config.STATUS_LABELS)}"
            )
        return value


@dataclasses.dataclass(frozen=True)
class Post:
    """One post the LLM will see.

    Attributes:
        author: Login of the poster.
        role: GitHub author_association, or "REPORTER" for the opening post.
        created_at: When it was posted.
        body: Text, possibly truncated.
        position: 0 for the opening post, else the comment's 1-based index.
        in_window: True if posted at or after the window start.
    """

    author: str
    role: str
    created_at: datetime.datetime
    body: str
    position: int
    in_window: bool


@dataclasses.dataclass(frozen=True)
class TrimmedThread:
    """The posts chosen for the LLM.

    Attributes:
        posts: Opening post first, then kept comments, oldest first.
        total_comments: Comments in the full thread.
    """

    posts: list[Post]
    total_comments: int


@dataclasses.dataclass(frozen=True)
class ThreadResult:
    """Outcome of summarizing one thread.

    Attributes:
        number: Issue number.
        summary: The validated summary as a dict, or None on fallback.
        error: Why the fallback happened, or None.
    """

    number: int
    summary: dict[str, str] | None
    error: str | None = None


def trim_thread(
    issue: Issue, comments: list[Comment], since: datetime.datetime
) -> TrimmedThread:
    """Picks the posts the LLM sees, within MAX_TOKENS_PER_ISSUE.

    Args:
        issue: The issue (its body is the opening post).
        comments: All fetched comments, oldest first.
        since: Window start, used to mark new comments.

    Returns:
        The chosen posts and the thread's full comment count.
    """
    opening = Post(
        author=issue.author,
        role="REPORTER",
        created_at=issue.created_at,
        body=issue.body,
        position=0,
        in_window=issue.created_at >= since,
    )
    last_start = max(0, len(comments) - config.KEEP_LAST_N_COMMENTS)
    kept = [
        _post(comment, index, since)
        for index, comment in enumerate(comments, start=1)
        if index > last_start or _is_key_comment(comment)
    ]

    total = sum(estimate_tokens(p.body) for p in [opening, *kept])
    if total > config.MAX_TOKENS_PER_ISSUE:
        opening, kept = _fit_to_cap(opening, kept)
    return TrimmedThread(posts=[opening, *kept], total_comments=len(comments))


def summarize_thread(
    item: DeltaItem,
    comments: list[Comment],
    since: datetime.datetime,
    model: BaseChatModel,
    guard: BudgetGuard,
    system_prompt: str | None = None,
    linked_prs: Sequence[LinkedPR] = (),
) -> ThreadResult:
    """Asks the model where one thread stands, with one retry.

    Args:
        item: The issue and its delta label.
        comments: All fetched comments, oldest first.
        since: Window start.
        model: Chat model, already capped at MAX_OUTPUT_TOKENS_SUMMARY.
        guard: Budget guard, checked before every call.
        system_prompt: Prompt to use instead of prompts/thread_state.md
            (the eval uses this to compare prompt versions).
        linked_prs: Pull requests linked to the issue.

    Returns:
        The validated summary, or a fallback result with the reason.

    Raises:
        BudgetExceeded: If a call could pass a cap. Not caught here, so the
            run stops before overspending.
    """
    thread = trim_thread(item.issue, comments, since)
    messages: list[BaseMessage] = [
        SystemMessage(system_prompt or _system_prompt()),
        HumanMessage(render_thread(item, thread, since, linked_prs)),
    ]
    summary, error = invoke_json(
        model, messages, ThreadSummary, guard, config.MAX_OUTPUT_TOKENS_SUMMARY
    )
    if summary is None:
        return ThreadResult(item.issue.number, None, error)
    return ThreadResult(item.issue.number, summary.model_dump())


def summarize_threads(
    items: list[DeltaItem],
    get_comments: Callable[[int], list[Comment]],
    since: datetime.datetime,
    model: BaseChatModel,
    guard: BudgetGuard,
    system_prompt: str | None = None,
    get_linked_prs: Callable[[int], list[LinkedPR]] | None = None,
) -> dict[int, ThreadResult]:
    """Fetches and summarizes several threads, MAX_CONCURRENT_CALLS at once.

    The budget guard counts calls still in flight, so running in parallel
    can't overspend. If any thread raises (budget or provider error), no new
    threads start, calls already running finish, and the first error is
    re-raised.

    Args:
        items: Issues to summarize (cache hits already removed).
        get_comments: Fetches an issue's comments by number, oldest first.
        since: Window start.
        model: Chat model, already capped at MAX_OUTPUT_TOKENS_SUMMARY.
        guard: Budget guard shared by all calls.
        system_prompt: Optional prompt override, as in summarize_thread.
        get_linked_prs: Fetches an issue's linked PRs by number; None to
            skip them.

    Returns:
        One result per issue number.

    Raises:
        BudgetExceeded: If a call could pass a cap.
    """

    def one(item: DeltaItem) -> ThreadResult:
        number = item.issue.number
        comments = get_comments(number)
        linked = get_linked_prs(number) if get_linked_prs else []
        return summarize_thread(
            item, comments, since, model, guard, system_prompt, linked
        )

    results: dict[int, ThreadResult] = {}
    pool = concurrent.futures.ThreadPoolExecutor(config.MAX_CONCURRENT_CALLS)
    try:
        futures = {pool.submit(one, item): item for item in items}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results[result.number] = result
    finally:
        # On error: drop threads not yet started, let running ones finish.
        pool.shutdown(wait=True, cancel_futures=True)
    return results


def render_thread(
    item: DeltaItem,
    thread: TrimmedThread,
    since: datetime.datetime,
    linked_prs: Sequence[LinkedPR] = (),
) -> str:
    """Formats the trimmed thread as the user message.

    Args:
        item: The issue and its delta label.
        thread: The chosen posts.
        since: Window start.
        linked_prs: Pull requests linked to the issue.

    Returns:
        The message text. Thread content sits inside <thread> tags so the
        prompt can tell the model to treat it as data.
    """
    issue = item.issue
    lines = [
        f"Issue #{issue.number}: {issue.title}",
        f"State: {issue.state}. This week's change: {item.label}.",
        f"Reporting window starts {since:%Y-%m-%d %H:%M} UTC.",
        f"The thread has {thread.total_comments} comments.",
        "",
        "<thread>",
        "Linked pull requests:" + ("" if linked_prs else " none"),
    ]
    for pr in linked_prs:
        marker = " · NEW THIS WINDOW" if pr.linked_at >= since else ""
        state = "closed, not merged" if pr.state == "closed" else pr.state
        lines.append(f"- PR #{pr.number} ({state}): {pr.title}{marker}")
    previous = 0
    for post in thread.posts:
        skipped = post.position - previous - 1
        if post.position > 0 and skipped > 0:
            lines.append(f"[{skipped} comments omitted]")
        previous = post.position
        label = (
            "Opening post" if post.position == 0 else f"Comment {post.position}"
        )
        marker = " · NEW THIS WINDOW" if post.in_window else ""
        lines.append(
            f"--- {label} · @{post.author} ({post.role}) · "
            f"{post.created_at:%Y-%m-%d}{marker} ---"
        )
        lines.append(post.body or "(empty)")
    lines.append("</thread>")
    return "\n".join(lines)


def _system_prompt() -> str:
    path = importlib.resources.files("monday_triage") / "prompts"
    return (path / "thread_state.md").read_text(encoding="utf-8")


def _post(comment: Comment, index: int, since: datetime.datetime) -> Post:
    return Post(
        author=comment.author,
        role=comment.author_association,
        created_at=comment.created_at,
        body=comment.body,
        position=index,
        in_window=comment.created_at >= since,
    )


def _is_key_comment(comment: Comment) -> bool:
    return comment.author_association in config.MAINTAINER_ROLES or bool(
        config.FIX_LINK_PATTERN.search(comment.body)
    )


def _fit_to_cap(opening: Post, kept: list[Post]) -> tuple[Post, list[Post]]:
    """Cuts the opening post, then keeps the newest comments that fit."""
    opening = dataclasses.replace(
        opening, body=_truncate(opening.body, config.FIRST_POST_MAX_TOKENS)
    )
    room = config.MAX_TOKENS_PER_ISSUE - estimate_tokens(opening.body)
    fitted: list[Post] = []
    for post in reversed(kept):
        size = estimate_tokens(post.body)
        if size <= room:
            fitted.append(post)
            room -= size
        elif not fitted:
            # Always keep the newest comment, cut to the room left.
            body = _truncate(post.body, room)
            fitted.append(dataclasses.replace(post, body=body))
            room -= estimate_tokens(body)
    return opening, list(reversed(fitted))


def _truncate(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * config.CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - len(_TRUNCATED))] + _TRUNCATED
