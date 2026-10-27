"""Token and spend guard for LLM calls.

Before every call, `reserve` checks the worst case: the estimated input plus
the full output cap, on top of what is already spent and what calls still in
flight could spend. If that could pass the run's token cap or dollar budget,
it raises `BudgetExceeded` and the call is never made. After the call,
`record` swaps the reservation for the real token counts the provider
reported. The guard is thread-safe, so concurrent calls can't jointly
overspend.
"""

import dataclasses
import math
import threading
from typing import Any

from monday_triage import config


class BudgetExceeded(RuntimeError):
    """The next LLM call could pass a cap. The message names the cap."""


@dataclasses.dataclass(frozen=True)
class Reservation:
    """Worst-case size of one call in flight.

    Attributes:
        input_tokens: Estimated input tokens.
        max_output_tokens: The call's output cap.
    """

    input_tokens: int
    max_output_tokens: int


def estimate_tokens(text: str) -> int:
    """Estimates tokens as characters / CHARS_PER_TOKEN, rounded up."""
    return math.ceil(len(text) / config.CHARS_PER_TOKEN)


class BudgetGuard:
    """Tracks spend for one run and refuses calls that could overrun it.

    Attributes:
        calls: LLM calls made.
        input_tokens: Input tokens used so far.
        output_tokens: Output tokens used so far.
    """

    def __init__(
        self,
        budget_usd: float,
        price_per_mtok: tuple[float, float],
        max_total_tokens: int = config.MAX_TOTAL_TOKENS,
    ) -> None:
        """Creates a guard.

        Args:
            budget_usd: Hard spending cap for the run.
            price_per_mtok: (input, output) USD per million tokens.
            max_total_tokens: Hard cap on input tokens for the run.
        """
        self._budget_usd = budget_usd
        self._input_price, self._output_price = price_per_mtok
        self._max_total_tokens = max_total_tokens
        self._lock = threading.Lock()
        self._pending_input = 0
        self._pending_usd = 0.0
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    @property
    def spent_usd(self) -> float:
        """Dollars spent so far, from recorded token counts."""
        return self._cost(self.input_tokens, self.output_tokens)

    def reserve(self, input_tokens: int, max_output_tokens: int) -> Reservation:
        """Reserves room for one call if its worst case fits both caps.

        Args:
            input_tokens: Estimated input tokens for the call.
            max_output_tokens: The call's output token cap.

        Returns:
            The reservation, to pass to `record` or `release`.

        Raises:
            BudgetExceeded: If the worst case would pass a cap.
        """
        worst_usd = self._cost(input_tokens, max_output_tokens)
        with self._lock:
            used = self.input_tokens + self._pending_input
            if used + input_tokens > self._max_total_tokens:
                raise BudgetExceeded(
                    f"Stopped before an LLM call: it would pass the run's "
                    f"token cap of {self._max_total_tokens:,} input tokens "
                    f"({used:,} used or in flight, {input_tokens:,} needed)."
                )
            committed = self.spent_usd + self._pending_usd
            if committed + worst_usd > self._budget_usd:
                raise BudgetExceeded(
                    f"Stopped before an LLM call: it could pass --budget-usd "
                    f"${self._budget_usd:.2f} (${committed:.4f} spent or in "
                    f"flight, worst case after this call "
                    f"${committed + worst_usd:.4f})."
                )
            self._pending_input += input_tokens
            self._pending_usd += worst_usd
        return Reservation(input_tokens, max_output_tokens)

    def record(
        self,
        reservation: Reservation,
        usage: dict[str, Any] | None,
        estimated_output: int,
    ) -> None:
        """Replaces a reservation with the finished call's token counts.

        Args:
            reservation: What `reserve` returned for this call.
            usage: The reply's `usage_metadata` (input_tokens, output_tokens),
                or None if the provider didn't report it.
            estimated_output: Fallback output count if usage is missing. The
                input fallback is the reservation's estimate.
        """
        usage = usage or {}
        with self._lock:
            self._release(reservation)
            self.calls += 1
            self.input_tokens += (
                usage.get("input_tokens") or reservation.input_tokens
            )
            self.output_tokens += usage.get("output_tokens") or estimated_output

    def release(self, reservation: Reservation) -> None:
        """Drops a reservation for a call that failed before replying."""
        with self._lock:
            self._release(reservation)

    def _release(self, reservation: Reservation) -> None:
        self._pending_input -= reservation.input_tokens
        self._pending_usd -= self._cost(
            reservation.input_tokens, reservation.max_output_tokens
        )

    def _cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self._input_price
            + output_tokens * self._output_price
        ) / 1_000_000
