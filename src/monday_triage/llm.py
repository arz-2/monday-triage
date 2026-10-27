"""Builds the chat model and runs checked JSON calls on it.

`build_chat_model` uses LangChain's `init_chat_model`, which picks the
integration class from LLM_PROVIDER, so swapping providers needs no code
change, only env vars and the provider's package.

`invoke_json` is the one place LLM calls happen: it checks the budget, calls
the model, records real token use, validates the JSON reply against a
pydantic schema, and retries once with the error. Replies are JSON in plain
text rather than tool calls, so this works with every chat model.
"""

import json
import re
from typing import TypeVar

import pydantic
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from monday_triage import config
from monday_triage.budget import BudgetGuard, estimate_tokens
from monday_triage.config import Settings

SchemaT = TypeVar("SchemaT", bound=pydantic.BaseModel)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class LLMSetupError(RuntimeError):
    """The chat model can't be built. The message says how to fix it."""


def build_chat_model(settings: Settings, max_tokens: int) -> BaseChatModel:
    """Builds the configured chat model with an output token cap.

    Args:
        settings: Run settings (provider and model name).
        max_tokens: Output token cap for every call made with this model.
            Thread summaries and themes use separate caps.

    Returns:
        A LangChain chat model. API keys are read by the provider's own
        package from its usual environment variable.

    Raises:
        LLMSetupError: If the provider is unknown to LangChain or its
            package isn't installed.
    """
    extra = config.PROVIDER_MODEL_KWARGS.get(settings.llm_provider, {})
    try:
        return init_chat_model(
            settings.llm_model,
            model_provider=settings.llm_provider,
            max_tokens=max_tokens,
            timeout=config.LLM_TIMEOUT_S,
            max_retries=config.LLM_TRANSPORT_RETRIES,
            **extra,
        )
    except ImportError as e:
        raise LLMSetupError(
            f"Provider '{settings.llm_provider}' needs its LangChain package. "
            f'{e}. Common ones have extras, e.g. pip install -e ".[openai]".'
        ) from e
    except ValueError as e:
        raise LLMSetupError(
            f"Can't build model '{settings.llm_model}' for provider "
            f"'{settings.llm_provider}': {e}"
        ) from e


def invoke_json(
    model: BaseChatModel,
    messages: list[BaseMessage],
    schema: type[SchemaT],
    guard: BudgetGuard,
    max_output_tokens: int,
) -> tuple[SchemaT | None, str | None]:
    """Calls the model for a JSON answer, with one retry on a bad reply.

    Args:
        model: Chat model, built with `max_output_tokens` as its cap.
        messages: Prompt messages. Not modified.
        schema: Pydantic model the reply must match.
        guard: Budget guard, checked before every call (retries included).
        max_output_tokens: The model's output cap, for the worst-case check.

    Returns:
        (parsed answer, None) on success, or (None, reason) if the reply was
        still invalid after MAX_LLM_RETRIES retries.

    Raises:
        BudgetExceeded: If a call could pass a cap. Provider errors (auth,
            network) also propagate, so a broken setup stops the run.
    """
    messages = list(messages)
    error = None
    for _ in range(1 + config.MAX_LLM_RETRIES):
        input_estimate = sum(estimate_tokens(str(m.content)) for m in messages)
        reservation = guard.reserve(input_estimate, max_output_tokens)
        try:
            reply = model.invoke(messages)
        except Exception:
            guard.release(reservation)
            raise
        # `.text` joins the text blocks and skips thinking blocks, which some
        # providers (e.g. Anthropic with effort set) put in `content`.
        text = reply.text
        guard.record(
            reservation,
            getattr(reply, "usage_metadata", None),
            estimate_tokens(text),
        )
        try:
            return parse_json(text, schema), None
        except ValueError as e:
            error = str(e)
            messages += [
                AIMessage(text),
                HumanMessage(
                    f"That reply was invalid: {error}\n"
                    "Reply again with only the JSON object."
                ),
            ]
    return None, error


def parse_json(text: str, schema: type[SchemaT]) -> SchemaT:
    """Pulls the JSON object out of a reply and validates it.

    Tolerates code fences or a sentence around the object.

    Args:
        text: The model's reply.
        schema: Pydantic model to validate against.

    Returns:
        The validated object.

    Raises:
        ValueError: If there is no JSON object or it fails validation. The
            message is short enough to send back to the model.
    """
    match = _JSON_OBJECT.search(text)
    if match is None:
        raise ValueError("no JSON object found")
    try:
        return schema.model_validate(json.loads(match.group()))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON ({e.msg})") from None
    except pydantic.ValidationError as e:
        raise ValueError(
            "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                for err in e.errors()
            )
        ) from None
