"""Constants and run settings.

Every tunable value lives here as a named constant, so the pipeline code has
no magic numbers. `load_settings` merges CLI flags and environment variables
into one frozen `Settings` object and rejects bad values with a clear message.
"""

import dataclasses
import datetime
import os
import pathlib
import re
from collections.abc import Mapping

# --- Thread trimming (thread_state.py) ---
KEEP_FIRST_POST = 1
KEEP_LAST_N_COMMENTS = 5
# GitHub `author_association` values that mark a project team member.
MAINTAINER_ROLES = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
# A comment pointing at a fix: an issue/PR ref, a PR link, or a commit SHA.
FIX_LINK_PATTERN = re.compile(r"#\d+|/pull/\d+|\b[0-9a-f]{7,40}\b")

# When a trimmed thread still exceeds MAX_TOKENS_PER_ISSUE, the opening post
# is cut to this size and the newest kept comments fill the rest.
FIRST_POST_MAX_TOKENS = 1_000
# Longest allowed text field in an LLM summary.
MAX_SUMMARY_FIELD_CHARS = 400

# --- Token and spend caps (budget.py) ---
# Rough token estimate used before a call (the 8-week measurement used it
# too); real counts from the provider replace it after the call.
CHARS_PER_TOKEN = 4
# Measured over 8 weeks of huggingface/datasets: 153 of 156 threads fit.
MAX_TOKENS_PER_ISSUE = 4_000
# 30 issues x 4,000 + ~5,000 for the themes call, so the two caps agree.
MAX_TOTAL_TOKENS = 125_000
# Output caps, thinking included. First real run (17 issues, Haiku 5.5 at
# low effort): summaries max 613, themes ~1,040. Set to about 2.5x / 3x.
# Worst case for a full run: 30 x 1,500 + 3,000 = 48k output, ~$0.024.
MAX_OUTPUT_TOKENS_SUMMARY = 1_500
MAX_OUTPUT_TOKENS_THEMES = 3_000
# Thread summaries run this many LLM calls at once (17 issues took 48 s one
# at a time; a 30-issue week would miss the 60 s target).
MAX_CONCURRENT_CALLS = 5
DEFAULT_BUDGET_USD = 0.10

# --- Scope ---
DEFAULT_MAX_ISSUES = 30
DEFAULT_LOOKBACK_DAYS = 7

# --- GitHub (github_client.py) ---
GITHUB_API_URL = "https://api.github.com"
GITHUB_PER_PAGE = 100
GITHUB_TIMEOUT_S = 30
# Page caps keep a run bounded: 1,000 changed issues, 500 comments a thread,
# 300 timeline events (where linked pull requests come from).
MAX_ISSUE_PAGES = 10
MAX_COMMENT_PAGES = 5
MAX_TIMELINE_PAGES = 3

# --- Labels ---
DELTA_LABELS = ("new", "newly active", "closed", "quiet")
STATUS_LABELS = (
    "needs_triage",
    "waiting_on_maintainer",
    "waiting_on_reporter",
    "fix_in_progress",
    "resolved",
)

# --- LLM output checks ---
MAX_LLM_RETRIES = 1
FALLBACK_TEXT = "summary unavailable"
MAX_THEMES = 5
MIN_ISSUES_PER_THEME = 2
MAX_THEME_NAME_CHARS = 60

# --- Output and state ---
FORMATS = ("text", "json")
DEFAULT_FORMAT = "text"
DEFAULT_STATE_DIR = ".monday_triage"
# Bump when the snapshot layout changes; older files are then rejected.
STATE_VERSION = 1

# --- Eval targets (eval/run_eval.py) ---
STATUS_ACCURACY_TARGET = 0.80

# --- Models ---
# Providers with a default model. Every other provider requires LLM_MODEL.
DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5"}
# (input, output) USD per million tokens. Anthropic list prices, 2026-10-06.
# Models not listed need --price-per-mtok.
PRICES_PER_MTOK = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
# Extra model settings per provider. Anthropic: low effort keeps thinking
# (billed as output) short; this is a summarize-and-label task.
PROVIDER_MODEL_KWARGS = {"anthropic": {"effort": "low"}}
LLM_TIMEOUT_S = 60
# Retries for transient provider errors (rate limits, 5xx), done by LangChain.
# Separate from MAX_LLM_RETRIES, which retries invalid answers.
LLM_TRANSPORT_RETRIES = 2
# API key variables for common providers, checked up front so a missing key
# fails before any work. Other providers' packages report their own errors.
PROVIDER_KEY_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
}

_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class ConfigError(ValueError):
    """A setting is missing or invalid. The message says how to fix it."""


@dataclasses.dataclass(frozen=True)
class Settings:
    """Everything one run needs, fixed after loading.

    Attributes:
        repo: Repository as "owner/name".
        since: Start of the window, or None to use the last run (else the
            last DEFAULT_LOOKBACK_DAYS days).
        max_issues: Most changed issues to summarize.
        budget_usd: Hard spending cap for the run.
        price_per_mtok: (input, output) USD per million tokens.
        output_format: "text" or "json".
        llm_provider: Provider name for LangChain's init_chat_model.
        llm_model: Model name for that provider.
        github_token: Optional GitHub token. Hidden from repr.
        state_dir: Directory holding the last-run snapshots.
    """

    repo: str
    since: datetime.datetime | None
    max_issues: int
    budget_usd: float
    price_per_mtok: tuple[float, float]
    output_format: str
    llm_provider: str
    llm_model: str
    github_token: str | None = dataclasses.field(repr=False)
    state_dir: pathlib.Path


def load_settings(
    repo: str,
    since: str | None = None,
    max_issues: int | None = None,
    budget_usd: float | None = None,
    price_per_mtok: str | None = None,
    output_format: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Builds validated settings from CLI flag values and the environment.

    Flags win over defaults. Environment values come from `env`, which
    defaults to `os.environ` (the CLI loads `.env` into it first).

    Args:
        repo: Repository as "owner/name".
        since: Window start as YYYY-MM-DD, or None.
        max_issues: Flag value, or None for the default.
        budget_usd: Flag value, or None for the default.
        price_per_mtok: "IN,OUT" in USD per million tokens, or None to look
            the model up in PRICES_PER_MTOK.
        output_format: "text" or "json", or None for the default.
        env: Environment variables to read.

    Returns:
        The frozen settings for this run.

    Raises:
        ConfigError: If any value is missing or invalid.
    """
    env = os.environ if env is None else env

    provider = _env(env, "LLM_PROVIDER")
    if provider is None:
        raise ConfigError(
            "LLM_PROVIDER is not set. Set it to a LangChain provider such as "
            "'anthropic' (see .env.example)."
        )
    provider = provider.lower()
    model = _env(env, "LLM_MODEL") or DEFAULT_MODELS.get(provider)
    if model is None:
        raise ConfigError(
            f"LLM_MODEL is required for provider '{provider}'. "
            "Only 'anthropic' has a default model."
        )
    key_var = PROVIDER_KEY_VARS.get(provider)
    if key_var is not None and _env(env, key_var) is None:
        raise ConfigError(
            f"{key_var} is not set. Add it to .env or the environment."
        )

    state_dir = _env(env, "MONDAY_TRIAGE_STATE_DIR") or DEFAULT_STATE_DIR

    return Settings(
        repo=_parse_repo(repo),
        since=_parse_since(since),
        max_issues=_parse_max_issues(max_issues),
        budget_usd=_parse_budget(budget_usd),
        price_per_mtok=_parse_price(price_per_mtok, model),
        output_format=_parse_format(output_format),
        llm_provider=provider,
        llm_model=model,
        github_token=_env(env, "GITHUB_TOKEN"),
        state_dir=pathlib.Path(state_dir),
    )


def _env(env: Mapping[str, str], name: str) -> str | None:
    """Returns a stripped env value, treating empty strings as unset."""
    value = env.get(name, "").strip()
    return value or None


def _parse_repo(repo: str) -> str:
    repo = repo.strip().removesuffix("/")
    if not _REPO_PATTERN.match(repo):
        raise ConfigError(
            f"Repository '{repo}' must look like 'owner/name', "
            "e.g. huggingface/datasets."
        )
    return repo


def _parse_since(since: str | None) -> datetime.datetime | None:
    if since is None:
        return None
    try:
        day = datetime.date.fromisoformat(since)
    except ValueError:
        raise ConfigError(
            f"--since '{since}' must be a date like 2026-10-19."
        ) from None
    start = datetime.datetime.combine(
        day, datetime.time(), tzinfo=datetime.timezone.utc
    )
    if start > datetime.datetime.now(datetime.timezone.utc):
        raise ConfigError(f"--since {since} is in the future.")
    return start


def _parse_max_issues(max_issues: int | None) -> int:
    if max_issues is None:
        return DEFAULT_MAX_ISSUES
    if max_issues < 1:
        raise ConfigError(f"--max-issues must be at least 1, got {max_issues}.")
    return max_issues


def _parse_budget(budget_usd: float | None) -> float:
    if budget_usd is None:
        return DEFAULT_BUDGET_USD
    if budget_usd <= 0:
        raise ConfigError(f"--budget-usd must be positive, got {budget_usd}.")
    return budget_usd


def _parse_price(price: str | None, model: str) -> tuple[float, float]:
    if price is None:
        if model not in PRICES_PER_MTOK:
            raise ConfigError(
                f"No known price for model '{model}', so the spending cap "
                "can't be enforced. Pass --price-per-mtok IN,OUT "
                "(USD per million tokens; 0,0 for a local model)."
            )
        return PRICES_PER_MTOK[model]
    parts = price.split(",")
    try:
        if len(parts) != 2:
            raise ValueError
        input_price, output_price = (float(p) for p in parts)
    except ValueError:
        raise ConfigError(
            f"--price-per-mtok '{price}' must be two numbers like 0.10,0.50."
        ) from None
    if input_price < 0 or output_price < 0:
        raise ConfigError(f"--price-per-mtok '{price}' can't be negative.")
    return input_price, output_price


def _parse_format(output_format: str | None) -> str:
    if output_format is None:
        return DEFAULT_FORMAT
    if output_format not in FORMATS:
        raise ConfigError(
            f"--format must be one of {', '.join(FORMATS)}, "
            f"got '{output_format}'."
        )
    return output_format
