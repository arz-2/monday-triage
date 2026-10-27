import pytest

from monday_triage.budget import BudgetExceeded, BudgetGuard, estimate_tokens

PRICE = (1.00, 5.00)  # USD per million input / output tokens


def test_estimate_tokens_rounds_up():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_reserve_allows_call_within_both_caps():
    BudgetGuard(0.10, PRICE, max_total_tokens=10_000).reserve(5_000, 1_000)


def test_reserve_refuses_call_that_could_pass_budget():
    guard = BudgetGuard(0.01, PRICE)
    # Worst case: 5,000 in ($0.005) + 2,000 out ($0.010) = $0.015 > $0.01.
    with pytest.raises(BudgetExceeded, match=r"--budget-usd \$0\.01"):
        guard.reserve(5_000, 2_000)


def test_reserve_refuses_call_that_passes_token_cap():
    guard = BudgetGuard(100.0, PRICE, max_total_tokens=10_000)
    guard.record(guard.reserve(8_000, 0), None, 0)
    with pytest.raises(BudgetExceeded, match="token cap of 10,000"):
        guard.reserve(3_000, 100)


def test_record_uses_reported_usage_then_estimates():
    guard = BudgetGuard(1.0, PRICE)
    reservation = guard.reserve(9, 9)
    guard.record(reservation, {"input_tokens": 1_000, "output_tokens": 200}, 9)
    guard.record(guard.reserve(500, 9), None, 100)
    assert (guard.calls, guard.input_tokens, guard.output_tokens) == (
        2,
        1_500,
        300,
    )
    assert guard.spent_usd == pytest.approx(0.0015 + 0.0015)


def test_calls_in_flight_count_against_the_budget():
    guard = BudgetGuard(0.01, PRICE)
    # Each worst case is 1,000 in + 1,000 out = $0.006; two can't both fit.
    first = guard.reserve(1_000, 1_000)
    with pytest.raises(BudgetExceeded, match="in flight"):
        guard.reserve(1_000, 1_000)
    guard.record(first, {"input_tokens": 1_000, "output_tokens": 100}, 0)
    guard.reserve(1_000, 1_000)  # Real spend was small, so this now fits.


def test_release_frees_a_failed_call():
    guard = BudgetGuard(0.01, PRICE)
    guard.release(guard.reserve(1_000, 1_000))
    guard.reserve(1_000, 1_000)
    assert guard.calls == 0
