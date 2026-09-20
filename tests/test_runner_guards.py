"""Tests for privacy guards, cost ceiling checks, cost estimation, and the round-robin plan builder."""

import pytest

from helpers import fake_mode
from modebench.config import (
    CostConfig,
    MeetingConfig,
    Mode,
    Price,
    Profile,
    ProviderConfig,
    TranscriptConfig,
)
from modebench.dataset.schema import FillerBlock, FillerFile, Line
from modebench.dataset.transcript import Labels
from modebench.dataset.variants import RenderLine, Variant
from modebench.errors import ConfigError, CostCeilingExceeded, PrivacyViolation
from modebench.providers.base import StreamOutcome
from modebench.providers.metrics import DerivedMetrics, assumed_cost
from modebench.runner.cost import CostEstimate, ModeEstimate, estimate_cost
from modebench.runner.guard import (
    check_running_cost,
    enforce_cost_ceiling,
    enforce_privacy,
    is_openrouter,
    privacy_problems,
)
from modebench.runner.plan import build_items, build_plan


def test_privacy_problems_detects_free_models_and_missing_denial() -> None:
    openrouter_provider = ProviderConfig(
        kind="openai_compat",
        base_url="https://openrouter.test/api/v1",
        api_key_env="K",
        privacy="openrouter",
    )
    direct_provider = ProviderConfig(
        kind="openai_compat",
        base_url="https://google.test/v1beta/openai",
        api_key_env="K",
        privacy="attested",
    )

    # 1. Free variant
    mode_free = Mode(
        id="free-model",
        provider="openrouter",
        model="meta-llama/llama-3-8b-instruct:free",
        params={"provider": {"data_collection": "deny"}},
    )
    probs = privacy_problems(mode_free, openrouter_provider)
    assert any(":free" in p for p in probs)

    # 2. OpenRouter without data_collection: deny
    mode_open = Mode(
        id="open-model",
        provider="openrouter",
        model="google/gemini-flash",
        params={},
    )
    probs = privacy_problems(mode_open, openrouter_provider)
    assert any("data_collection" in p for p in probs)

    # 3. Direct API without private_data_ok = true
    mode_direct = Mode(
        id="direct-model",
        provider="google",
        model="gemini-2.5-flash",
        private_data_ok=False,
    )
    probs = privacy_problems(mode_direct, direct_provider)
    assert any("private_data_ok" in p for p in probs)

    # 4. Safe direct mode
    mode_direct_safe = Mode(
        id="direct-model-safe",
        provider="google",
        model="gemini-2.5-flash",
        private_data_ok=True,
    )
    assert privacy_problems(mode_direct_safe, direct_provider) == []


def test_enforce_privacy_blocks_unsafe_routes_on_private_data() -> None:
    provider = ProviderConfig(
        kind="openai_compat",
        base_url="https://openrouter.test/api/v1",
        api_key_env="K",
        privacy="openrouter",
    )
    unsafe_mode = Mode(id="unsafe", provider="openrouter", model="model-x", params={})

    # When no private data is used, it should not raise
    enforce_privacy(has_private_data=False, routes=[(unsafe_mode, provider)])

    # When private data is present, it must raise PrivacyViolation
    with pytest.raises(PrivacyViolation, match="A private dataset is selected"):
        enforce_privacy(has_private_data=True, routes=[(unsafe_mode, provider)])


def test_enforce_cost_ceiling_stops_excessive_or_unknown_spending() -> None:
    # Within ceiling
    enforce_cost_ceiling(estimate_usd=5.0, ceiling_usd=10.0)

    # Unknown price modes
    with pytest.raises(CostCeilingExceeded, match="These modes have no price"):
        enforce_cost_ceiling(
            estimate_usd=5.0, ceiling_usd=10.0, unknown_price_modes=["mystery-mode"]
        )

    # Over budget
    with pytest.raises(CostCeilingExceeded, match="The estimate is 15.00 USD"):
        enforce_cost_ceiling(estimate_usd=15.0, ceiling_usd=10.0)


def test_check_running_cost_trips_when_spent_exceeds_ceiling() -> None:
    assert not check_running_cost(spent_usd=9.99, ceiling_usd=10.0)
    assert check_running_cost(spent_usd=10.01, ceiling_usd=10.0)


def test_build_plan_interleaves_round_robin_with_warmup() -> None:
    variant = Variant(
        variant_id="var-1",
        case_id="case-1",
        kind="truncation",
        truncation_pct=100,
        asr_wer=0.0,
        noise_kind=None,
        expect_refusal=False,
        private=False,
        earlier=(),
        fresh=(RenderLine(speaker="mic", text="Qual o prazo?"),),
        gold_question="Qual o prazo?",
        key_points=("duas semanas",),
        reference_answer="Duas semanas.",
    )

    profile = Profile(
        design="star",
        durations_min=[5],
        baseline_duration_min=5,
        truncations=[100],
        noise_kinds=[],
        warmup=1,
        repetitions=2,
        max_cost_usd=10.0,
    )

    filler = FillerFile(
        blocks=[
            FillerBlock(
                topic="geral",
                lines=[Line(speaker="system", text="Discussao geral.")],
            )
        ]
    )
    transcript_cfg = TranscriptConfig(words_per_minute=120)
    meeting_cfg = MeetingConfig(click_minutes=[5])

    items = build_items(
        variants=[variant],
        profile=profile,
        filler=filler,
        transcript=transcript_cfg,
        meeting=meeting_cfg,
        seed=42,
    )
    assert len(items) == 1

    modes = [
        fake_mode("mode-A"),
        fake_mode("mode-B"),
    ]

    plan = build_plan(items, [m.id for m in modes], profile.repetitions, profile.warmup)
    # Warmup + 2 repetitions = 3 cycles for 2 modes = 6 requests
    assert len(plan) == 6

    # Verify warmup requests are scheduled first and marked warmup=True
    warmups = [req for req in plan if req.warmup]
    measured = [req for req in plan if not req.warmup]

    assert len(warmups) == 2
    assert len(measured) == 4
    assert all(w.seq < m.seq for w in warmups for m in measured)


def test_estimate_plan_cost_uses_prices() -> None:
    variant = Variant(
        variant_id="var-1",
        case_id="case-1",
        kind="truncation",
        truncation_pct=100,
        asr_wer=0.0,
        noise_kind=None,
        expect_refusal=False,
        private=False,
        earlier=(),
        fresh=(RenderLine(speaker="mic", text="Pergunta?"),),
        gold_question="Q",
        key_points=(),
        reference_answer="A",
    )

    filler = FillerFile(
        blocks=[
            FillerBlock(
                topic="geral",
                lines=[Line(speaker="system", text="Discussao.")],
            )
        ]
    )
    profile = Profile(
        design="star",
        durations_min=[5],
        baseline_duration_min=5,
        truncations=[100],
        noise_kinds=[],
        repetitions=1,
        warmup=0,
        max_cost_usd=10.0,
    )

    items = build_items(
        variants=[variant],
        profile=profile,
        filler=filler,
        transcript=TranscriptConfig(words_per_minute=100),
        meeting=MeetingConfig(),
        seed=1,
    )

    mode = Mode(
        id="priced-mode",
        provider="local",
        model="fake/priced",
        price=Price(usd_per_mtok_in=2.0, usd_per_mtok_out=5.0),
    )

    plan = build_plan(items, [mode.id], profile.repetitions, profile.warmup)
    labels = Labels(mic="MIC", system="SYSTEM", partial_marker="*")
    cost_cfg = CostConfig()
    estimate = estimate_cost(
        plan=plan,
        modes={mode.id: mode},
        prices={mode.id: mode.price},
        preprompt="Pre-prompt system",
        labels=labels,
        config=cost_cfg,
    )
    assert estimate.unknown_price_modes == []
    assert estimate.total_usd > 0.0


def test_the_host_of_the_endpoint_decides_that_a_route_is_openrouter() -> None:
    mislabelled = ProviderConfig(
        kind="openai_compat",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="K",
        privacy="attested",
    )
    attested = Mode(id="attested", provider="or", model="google/gemini", private_data_ok=True)
    assert is_openrouter(mislabelled)
    # The statement of the operator is not sufficient on OpenRouter: the route must deny.
    assert any("data_collection" in item for item in privacy_problems(attested, mislabelled))
    denies = attested.model_copy(update={"params": {"provider": {"data_collection": "deny"}}})
    assert privacy_problems(denies, mislabelled) == []

    local_label = mislabelled.model_copy(update={"privacy": "local"})
    assert any("data_collection" in item for item in privacy_problems(attested, local_label))
    subdomain = mislabelled.model_copy(update={"base_url": "https://eu.OpenRouter.ai/api/v1"})
    assert is_openrouter(subdomain)
    lookalike = mislabelled.model_copy(update={"base_url": "https://notopenrouter.ai/v1"})
    assert not is_openrouter(lookalike)
    true_local = ProviderConfig(kind="fake", privacy="local")
    assert privacy_problems(attested, true_local) == []


def test_the_estimate_of_one_request_and_the_cost_that_the_ceiling_assumes() -> None:
    estimate = CostEstimate(
        modes=[
            ModeEstimate("priced", requests=4, input_tokens=0, output_tokens=0, cost_usd=2.0),
            ModeEstimate("no-price", requests=4, input_tokens=0, output_tokens=0, cost_usd=None),
            ModeEstimate("idle", requests=0, input_tokens=0, output_tokens=0, cost_usd=0.0),
        ]
    )
    assert estimate.per_request_usd("priced") == 0.5
    assert estimate.per_request_usd("no-price") == 0.0
    assert estimate.per_request_usd("idle") == 0.0
    assert estimate.per_request_usd("absent") == 0.0
    assert estimate.unknown_price_modes == ["no-price"]
    assert estimate.requests == 8

    unknown = DerivedMetrics(None, None, None, None, "unknown")
    known = DerivedMetrics(None, None, None, 0.2, "usage")
    answered = StreamOutcome(ok=True, total_ms=10.0)
    assert assumed_cost(answered, unknown, 0.5) == 0.5
    assert assumed_cost(answered, known, 0.5) == 0.0
    for kind in ("timeout", "stream_error", "empty_output"):
        billed = StreamOutcome(ok=False, total_ms=10.0, error_kind=kind)
        assert assumed_cost(billed, unknown, 0.5) == 0.5
    for kind in ("http_error", "transport_error"):
        refused = StreamOutcome(ok=False, total_ms=10.0, error_kind=kind)
        assert assumed_cost(refused, unknown, 0.5) == 0.0


def test_the_estimate_counts_one_judge_request_for_each_measured_request() -> None:
    variant = Variant(
        variant_id="var-1",
        case_id="case-1",
        kind="truncation",
        truncation_pct=100,
        asr_wer=0.0,
        noise_kind=None,
        expect_refusal=False,
        private=False,
        earlier=(),
        fresh=(RenderLine(speaker="mic", text="Qual o prazo?"),),
        gold_question="Qual o prazo?",
        key_points=(),
        reference_answer="",
    )
    filler = FillerFile(
        blocks=[FillerBlock(topic="geral", lines=[Line(speaker="system", text="Discussao.")])]
    )
    profile = Profile(
        durations_min=[5], baseline_duration_min=5, truncations=[100], max_cost_usd=10.0
    )
    items = build_items([variant], profile, filler, TranscriptConfig(), MeetingConfig(), seed=1)
    plan = build_plan(items, ["m"], repetitions=2, warmup=1)
    mode = Mode(id="m", provider="local", model="fake/m", params={"reasoning": {"effort": "high"}})
    estimate = estimate_cost(
        plan,
        {"m": mode},
        {"m": Price(usd_per_mtok_in=1.0, usd_per_mtok_out=1.0), "judge": None},
        "preprompt",
        Labels(mic="MIC", system="SYSTEM", partial_marker="*"),
        CostConfig(),
        judge_prompt_chars=700,
    )
    by_id = {item.mode_id: item for item in estimate.modes}
    assert by_id["m"].requests == 3
    # The warm-up request is not graded.
    assert by_id["judge"].requests == 2
    assert by_id["m"].output_tokens == 3 * (220 + 4096)
    assert estimate.unknown_price_modes == ["judge"]
    with pytest.raises(ConfigError, match="no item"):
        build_plan([], ["m"], 1, 0)
    with pytest.raises(ConfigError, match="no mode"):
        build_plan(items, [], 1, 0)
