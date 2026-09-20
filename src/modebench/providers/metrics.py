"""Numbers that come from an outcome: answer tokens, speed and cost."""

from dataclasses import dataclass

from modebench.config import Price, ProviderConfig
from modebench.providers.base import StreamOutcome

# Failures that come after the provider started to make tokens.
BILLABLE_FAILURES = frozenset({"timeout", "stream_error", "empty_output"})


@dataclass(frozen=True, slots=True)
class DerivedMetrics:
    """What the usage block and the times of one outcome add up to."""

    answer_tokens: int | None
    reasoning_tokens: int | None
    tok_per_s: float | None
    cost_usd: float | None
    cost_source: str


def split_output_tokens(outcome: StreamOutcome) -> tuple[int | None, int | None, int]:
    """Return the answer tokens, the reasoning tokens and the billable output tokens.

    Two conventions exist. In the first, `completion_tokens` includes the
    reasoning, and the details block says how much of it is reasoning. In the
    second, `completion_tokens` is the visible answer only, and the reasoning
    shows only as the part of `total_tokens` that nothing else explains.
    """
    completion = outcome.completion_tokens
    reasoning = outcome.reasoning_tokens
    if completion is None:
        return None, reasoning, 0
    hidden = 0
    if outcome.total_tokens is not None and outcome.prompt_tokens is not None:
        hidden = max(0, outcome.total_tokens - outcome.prompt_tokens - completion)
    if hidden > 0:
        return completion, reasoning if reasoning is not None else hidden, completion + hidden
    if reasoning is None:
        return completion, None, completion
    return max(0, completion - reasoning), reasoning, completion


def price_cost(outcome: StreamOutcome, price: Price, billable_output: int) -> float | None:
    """Return the cost from the price table, or None if the usage is absent."""
    if outcome.prompt_tokens is None:
        return None
    cached = outcome.cached_tokens or 0
    cached = min(cached, outcome.prompt_tokens)
    cached_price = price.usd_per_mtok_cached_in
    if cached_price is None:
        cached_price = price.usd_per_mtok_in
    fresh = outcome.prompt_tokens - cached
    total = fresh * price.usd_per_mtok_in + cached * cached_price
    total += billable_output * price.usd_per_mtok_out
    return total / 1_000_000


def assumed_cost(outcome: StreamOutcome, metrics: DerivedMetrics, estimate_usd: float) -> float:
    """Return the cost that the ceiling assumes for a request with no known cost.

    A request with no usage block can still be billed: a timeout or a broken
    stream comes after the provider made tokens. Such a request counts as the
    estimate of one request, so the ceiling stays a ceiling. A request that
    the provider refused (an HTTP error or a transport error) made no tokens,
    and it counts as zero. A request with a known cost has nothing to assume.
    """
    if metrics.cost_usd is not None:
        return 0.0
    if outcome.ok or outcome.error_kind in BILLABLE_FAILURES:
        return max(0.0, estimate_usd)
    return 0.0


def derive_metrics(
    outcome: StreamOutcome, provider: ProviderConfig, price: Price | None
) -> DerivedMetrics:
    """Return the derived numbers of one outcome."""
    answer_tokens, reasoning_tokens, billable_output = split_output_tokens(outcome)
    tok_per_s: float | None = None
    if answer_tokens is not None and outcome.ttfat_ms is not None:
        seconds = (outcome.total_ms - outcome.ttfat_ms) / 1000.0
        if seconds > 0:
            tok_per_s = answer_tokens / seconds
    cost_usd: float | None = None
    cost_source = "unknown"
    if provider.cost_source == "usage" and outcome.reported_cost_usd is not None:
        cost_usd = outcome.reported_cost_usd
        cost_source = "usage"
    elif price is not None:
        cost_usd = price_cost(outcome, price, billable_output)
        if cost_usd is not None:
            cost_source = "price_table"
    return DerivedMetrics(
        answer_tokens=answer_tokens,
        reasoning_tokens=reasoning_tokens,
        tok_per_s=tok_per_s,
        cost_usd=cost_usd,
        cost_source=cost_source,
    )
