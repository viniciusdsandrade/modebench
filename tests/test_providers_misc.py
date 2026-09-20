"""Derived metrics, the fake provider and the provider registry."""

import pytest

from helpers import fake_mode, local_modes_file, openrouter_mode
from modebench.config import Price, ProviderConfig
from modebench.errors import PreflightError
from modebench.providers.base import ChatRequest, StreamOutcome
from modebench.providers.fake import REFUSAL_TEXT, FakeProvider, FakeSpec, fake_params
from modebench.providers.metrics import derive_metrics, price_cost, split_output_tokens
from modebench.providers.openai_compat import OpenAICompatProvider
from modebench.providers.registry import build_providers, effective_provider_config

USAGE_PROVIDER = ProviderConfig(
    kind="openai_compat", base_url="https://x.test", api_key_env="K", cost_source="usage"
)
TABLE_PROVIDER = ProviderConfig(kind="openai_compat", base_url="https://x.test", api_key_env="K")
PRICE = Price(usd_per_mtok_in=1.0, usd_per_mtok_out=4.0, usd_per_mtok_cached_in=0.1)


def outcome(**fields: object) -> StreamOutcome:
    values: dict[str, object] = {"ok": True, "total_ms": 3000.0, "ttfat_ms": 1000.0}
    values.update(fields)
    return StreamOutcome(**values)


def test_completion_tokens_that_include_the_reasoning_are_split() -> None:
    result = outcome(
        prompt_tokens=100, completion_tokens=300, reasoning_tokens=250, total_tokens=400
    )
    assert split_output_tokens(result) == (50, 250, 300)


def test_completion_tokens_that_exclude_the_reasoning_are_found_in_the_total() -> None:
    result = outcome(prompt_tokens=100, completion_tokens=50, total_tokens=400)
    assert split_output_tokens(result) == (50, 250, 300)


def test_no_usage_gives_no_tokens() -> None:
    assert split_output_tokens(outcome()) == (None, None, 0)
    assert split_output_tokens(outcome(completion_tokens=40)) == (40, None, 40)


def test_the_price_table_charges_cached_tokens_at_their_own_price() -> None:
    result = outcome(prompt_tokens=1_000_000, cached_tokens=400_000, completion_tokens=100_000)
    assert price_cost(result, PRICE, 100_000) == pytest.approx(0.6 + 0.04 + 0.4)
    no_cached_price = Price(usd_per_mtok_in=1.0, usd_per_mtok_out=4.0)
    assert price_cost(result, no_cached_price, 100_000) == pytest.approx(1.0 + 0.4)
    assert price_cost(outcome(), PRICE, 0) is None


def test_reported_cost_wins_when_the_provider_reports_it() -> None:
    result = outcome(prompt_tokens=10, completion_tokens=200, reported_cost_usd=0.5)
    metrics = derive_metrics(result, USAGE_PROVIDER, PRICE)
    assert metrics.cost_usd == 0.5
    assert metrics.cost_source == "usage"
    assert metrics.tok_per_s == pytest.approx(100.0)


def test_the_price_table_is_the_fallback_and_unknown_is_explicit() -> None:
    result = outcome(prompt_tokens=1000, completion_tokens=100)
    table = derive_metrics(result, TABLE_PROVIDER, PRICE)
    assert table.cost_source == "price_table"
    assert table.cost_usd == pytest.approx((1000 * 1.0 + 100 * 4.0) / 1_000_000)
    unknown = derive_metrics(result, TABLE_PROVIDER, None)
    assert unknown.cost_usd is None
    assert unknown.cost_source == "unknown"
    no_usage = derive_metrics(outcome(), USAGE_PROVIDER, PRICE)
    assert no_usage.cost_source == "unknown"
    assert no_usage.tok_per_s is None


def request(**metadata: str) -> ChatRequest:
    base = {
        "expect_refusal": "0",
        "gold_question": "Qual é o prazo?",
        "key_points": "seis semanas\ninício em outubro",
        "truncation_pct": "100",
        "repetition": "1",
    }
    base.update(metadata)
    return ChatRequest(system="system prompt", user="CHAMADA: qual é o prazo", metadata=base)


def test_the_fake_provider_is_deterministic() -> None:
    mode = fake_mode("steady", ttft_ms=200, reasoning_ms=500, quality=1.0)
    first = FakeProvider().stream_chat(mode, request(), 60.0)
    second = FakeProvider().stream_chat(mode, request(), 60.0)
    assert first == second
    assert first.ok
    assert first.ttft_ms is not None and first.ttfat_ms is not None
    assert first.ttfat_ms > first.ttft_ms
    assert first.first_reasoning_ms == first.ttft_ms
    assert "Qual é o prazo?" in first.answer
    assert "seis semanas" in first.answer
    other = FakeProvider().stream_chat(mode, request(repetition="2"), 60.0)
    assert other.ttft_ms != first.ttft_ms


def test_the_fake_provider_refuses_noise_when_its_quality_is_perfect() -> None:
    mode = fake_mode("refuser", quality=1.0, reasoning_ms=0)
    result = FakeProvider().stream_chat(mode, request(expect_refusal="1"), 60.0)
    assert result.answer == REFUSAL_TEXT
    assert result.first_reasoning_ms is None


def test_the_fake_provider_can_fail_time_out_and_add_a_preamble() -> None:
    failing = FakeProvider().stream_chat(fake_mode("broken", error_rate=1.0), request(), 60.0)
    assert not failing.ok
    assert failing.error_kind == "fake_error"
    slow = FakeProvider().stream_chat(fake_mode("slow", reasoning_ms=500_000), request(), 60.0)
    assert not slow.ok
    assert slow.error_kind == "timeout"
    assert slow.total_ms == 60_000.0
    polite = FakeProvider().stream_chat(fake_mode("polite", preamble_rate=1.0), request(), 60.0)
    assert polite.answer.startswith("Claro!")
    wrong = FakeProvider().stream_chat(fake_mode("wrong", quality=0.0), request(), 60.0)
    assert "Qual é o prazo?" not in wrong.answer


def test_a_warm_cache_makes_the_fake_provider_faster_and_reports_cached_tokens() -> None:
    mode = fake_mode("cache", ttft_ms=1000, quality=1.0)
    cold = FakeProvider().stream_chat(mode, request(cache_state="cold"), 60.0)
    warm = FakeProvider().stream_chat(mode, request(cache_state="warm"), 60.0)
    assert cold.ttft_ms is not None and warm.ttft_ms is not None
    assert warm.ttft_ms < cold.ttft_ms
    assert cold.cached_tokens == 0
    assert warm.cached_tokens is not None and warm.cached_tokens > 0


def test_the_fake_spec_follows_the_reasoning_level_and_ignores_unknown_keys() -> None:
    low = FakeSpec.for_mode(openrouter_mode("a", reasoning={"effort": "low"}))
    none = FakeSpec.for_mode(openrouter_mode("b", reasoning={"effort": "none"}))
    assert low.reasoning_ms > none.reasoning_ms == 0.0
    odd = FakeSpec.for_mode(
        openrouter_mode("c", fake={"ttft_ms": 50, "nonsense": 1, "quality": True})
    )
    assert odd.ttft_ms == 50.0
    assert fake_params(ttft_ms=5) == {"fake": {"ttft_ms": 5}}


def test_a_dry_run_gives_each_mode_the_fake_provider() -> None:
    modes = [openrouter_mode("a"), openrouter_mode("b")]
    modes_file = local_modes_file(*modes)
    providers = build_providers(modes_file, modes, {}, dry_run=True)
    assert isinstance(providers["a"], FakeProvider)
    assert providers["a"] is providers["b"]
    assert effective_provider_config(modes_file, modes[0], dry_run=True).kind == "fake"
    assert effective_provider_config(modes_file, modes[0], dry_run=False).kind == "openai_compat"


def test_a_real_run_needs_the_key_of_each_provider() -> None:
    modes = [openrouter_mode("a"), fake_mode("local-mode")]
    modes_file = local_modes_file(*modes)
    with pytest.raises(PreflightError, match="OPENROUTER_API_KEY"):
        build_providers(modes_file, modes, {}, dry_run=False)
    providers = build_providers(modes_file, modes, {"OPENROUTER_API_KEY": "sk-1"}, dry_run=False)
    assert isinstance(providers["a"], OpenAICompatProvider)
    assert isinstance(providers["local-mode"], FakeProvider)
