# monday-triage

A command-line tool for open-source maintainers. Point it at any public GitHub
repo and it reports the issues that changed since your last run:

1. **Header:** time window and counts (new, newly active, closed).
2. **Needs your attention:** first reply needed, PR ready to review, reply
   waiting, or check by hand.
3. **Themes:** up to 5 groups across this week's issues.
4. **Other issues:** the rest, by status.
5. **Run line:** LLM calls, cost, and GitHub requests.

An LLM reads each thread and its linked PRs to say where the issue stands and
what changed, and groups the themes. Plain code does everything else.

## Quick start

Requires Python 3.10+.

```bash
python3 -m venv .venv && source .venv/bin/activate   # python3 must be 3.10+
pip install --upgrade pip
pip install -e . -c requirements-lock.txt   # exact versions we tested
cp .env.example .env        # then add your ANTHROPIC_API_KEY
monday-triage huggingface/datasets
```

Run it from the repo folder: `.env` and the state folder (`.monday_triage/`)
are looked up in the current directory.

**Windows:** create the venv with `py -3.10 -m venv .venv`, activate it with
`.venv\Scripts\activate`, and use `copy` instead of `cp`. Set
`PYTHONUTF8=1` if you redirect the output to a file.

The first run covers the last 7 days; each later run covers everything since
the previous one.

## Configuration

Settings come from environment variables, read from `.env` if present.

| Variable | Required | Meaning |
|---|---|---|
| `LLM_PROVIDER` | yes | Any provider LangChain's `init_chat_model` supports, e.g. `anthropic`, `openai`, `ollama` |
| `LLM_MODEL` | not for anthropic | Model name. Anthropic default: `claude-haiku-5-5` |
| Provider API key | yes, except local models | e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` |
| `GITHUB_TOKEN` | no, but recommended | Raises GitHub's limit from 60 to 5,000 requests/hour |
| `MONDAY_TRIAGE_STATE_DIR` | no | Where last-run snapshots go. Default `./.monday_triage/` |

### Switching providers

Install the provider's LangChain package and set the variables; no code
changes:

```bash
pip install -e ".[openai]"     # or [google], [mistral], [ollama], [groq], [bedrock]
LLM_PROVIDER=openai LLM_MODEL=<model> OPENAI_API_KEY=... \
  monday-triage huggingface/datasets --price-per-mtok <in>,<out>
```

Only Anthropic models have built-in prices, so other models need
`--price-per-mtok` (use `0,0` for a local model).

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--since YYYY-MM-DD` | last run, else 7 days ago | Start of the window |
| `--max-issues N` | 30 | Most issues to summarize; the rest are listed with a warning |
| `--budget-usd X` | 0.10 | Hard spending cap for one run |
| `--price-per-mtok IN,OUT` | built-in table | Model price in $ per million tokens |
| `--format text\|json` | `text` | `json` for machine-readable output |

## Limits and failures

Cheap by design, and it fails loudly, never silently.

- **Spend:** the worst-case cost is checked before every LLM call. If it
  would pass `--budget-usd`, the run stops with a non-zero exit code.
- **GitHub requests:** one search plus two per summarized issue (comments and
  linked PRs), so about 25 issues an hour without `GITHUB_TOKEN`.
- **Size:** at most 4,000 input tokens per issue and 125,000 per run. Long
  threads keep the first post, the 5 newest comments, and maintainer or
  fix-link comments.
- **Bad LLM answers:** each answer is schema-checked and retried once, then
  shown as "summary unavailable". Invalid themes are dropped.
- **Reruns:** summaries are cached per issue; a rerun with nothing new makes
  0 LLM calls and reprints the last report.

## Development

```bash
pip install -e ".[dev]" -c requirements-lock.txt
pytest                                # offline: recorded GitHub data, fake LLM
ruff check .                          # Google Python Style Guide rules
python eval/run_eval.py --repeats 3   # real LLM on a frozen week
python eval/measure_tokens.py huggingface/datasets --weeks 8
```

The eval scores status labels against hand labels in `eval/labels.json`, on a
week of issues frozen in `tests/fixtures/` (made with `eval/record_week.py`
and `eval/backfill_linked_prs.py`). Try other prompts with
`--prompt-file eval/prompts/<file>.md`.

