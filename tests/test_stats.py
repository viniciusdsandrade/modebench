"""Tests for statistical routines: percentiles, bootstrap intervals, Pareto fronts, and aggregation."""

from dataclasses import replace
from pathlib import Path

import pytest

from modebench.config import StatsConfig
from modebench.errors import StorageError
from modebench.stats.aggregate import (
    effective_latency,
    load_summary,
    summarize_mode,
)
from modebench.stats.pareto import ParetoPoint, dominates, pareto_front
from modebench.stats.percentiles import (
    Estimate,
    bootstrap_mean,
    bootstrap_percentile,
    percentile,
)
from modebench.storage.records import RequestRecord, ScoreRecord, StoredRequest


def test_percentile_computes_linear_interpolation() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 0.0) == 10.0
    assert percentile(values, 50.0) == 30.0
    assert percentile(values, 100.0) == 50.0


def test_percentile_empty_raises_value_error() -> None:
    with pytest.raises(ValueError, match="one value or more"):
        percentile([], 50.0)


def test_estimate_overlaps_detects_intersection() -> None:
    est1 = Estimate(value=10.0, ci_low=8.0, ci_high=12.0, n=50)
    est2 = Estimate(value=13.0, ci_low=11.5, ci_high=15.0, n=50)
    est3 = Estimate(value=20.0, ci_low=18.0, ci_high=22.0, n=50)

    assert est1.overlaps(est2)
    assert est2.overlaps(est1)
    assert not est1.overlaps(est3)
    assert not est3.overlaps(est1)


def test_bootstrap_percentile_returns_bounded_interval() -> None:
    values = [100.0 + float(i) for i in range(50)]
    est = bootstrap_percentile(values, 95.0, resamples=100, confidence=0.95, seed=42)
    assert est.n == 50
    assert est.ci_low <= est.value <= est.ci_high


def test_bootstrap_percentile_single_value_gives_exact_point() -> None:
    est = bootstrap_percentile([42.0], 50.0, resamples=100, confidence=0.95, seed=7)
    assert est.value == 42.0
    assert est.ci_low == 42.0
    assert est.ci_high == 42.0
    assert est.n == 1


def test_bootstrap_empty_raises_value_error() -> None:
    with pytest.raises(ValueError, match="one value or more"):
        bootstrap_percentile([], 50.0, resamples=50, confidence=0.95, seed=1)


def test_bootstrap_mean_calculates_ci() -> None:
    values = [10.0, 12.0, 11.0, 10.5, 11.5, 12.5, 9.5]
    est = bootstrap_mean(values, resamples=100, confidence=0.95, seed=123)
    assert est.n == len(values)
    assert est.ci_low <= est.value <= est.ci_high


def test_pareto_dominance() -> None:
    # High quality, low latency is best
    better = ParetoPoint(key="fast_and_good", quality=0.9, latency_ms=1000.0)
    worse = ParetoPoint(key="slow_and_poor", quality=0.7, latency_ms=2000.0)
    equal_qual_slower = ParetoPoint(key="equal_qual_slower", quality=0.9, latency_ms=1500.0)
    equal_lat_poorer = ParetoPoint(key="equal_lat_poorer", quality=0.8, latency_ms=1000.0)

    assert dominates(better, worse)
    assert not dominates(worse, better)
    assert dominates(better, equal_qual_slower)
    assert dominates(better, equal_lat_poorer)
    assert not dominates(equal_qual_slower, equal_lat_poorer)


def test_pareto_front_filters_dominated_and_sorts() -> None:
    p1 = ParetoPoint(key="fast", quality=0.7, latency_ms=500.0)
    p2 = ParetoPoint(key="balanced", quality=0.85, latency_ms=1200.0)
    p3 = ParetoPoint(key="accurate", quality=0.95, latency_ms=2500.0)
    p_dominated = ParetoPoint(key="dominated", quality=0.6, latency_ms=3000.0)

    front = pareto_front([p1, p2, p3, p_dominated])
    keys = [p.key for p in front]
    assert "dominated" not in keys
    assert keys == ["fast", "balanced", "accurate"]


def make_req(*, ok: bool, ttfat_ms: float | None = None) -> RequestRecord:
    return RequestRecord(
        run_id="r1",
        seq=1,
        mode_id="m1",
        suite="analyze",
        item_id="it1",
        case_id="c1",
        variant_id="v1",
        variant_kind="truncation",
        truncation_pct=100,
        asr_wer=0.0,
        noise_kind=None,
        duration_min=5,
        repetition=1,
        warmup=False,
        scenario_id=None,
        click_index=None,
        cache_state="cold",
        expect_refusal=False,
        private=False,
        started_at="2026-09-01T12:00:00Z",
        ok=ok,
        error_kind=None if ok else "error",
        error_message=None,
        http_status=200 if ok else 500,
        first_chunk_ms=None,
        ttft_ms=None,
        first_reasoning_ms=None,
        ttfat_ms=ttfat_ms,
        total_ms=1000.0,
        prompt_tokens=None,
        completion_tokens=None,
        reasoning_tokens=None,
        answer_tokens=None,
        cached_tokens=None,
        tok_per_s=None,
        served_by=None,
        cost_usd=None,
        cost_source="usage",
        prompt_chars=100,
        prompt_sha="sha",
        answer="ans",
    )


def test_effective_latency_counts_failure_as_timeout() -> None:
    timeout_ms = 60000.0
    rec_ok = make_req(ok=True, ttfat_ms=800.0)
    assert effective_latency(rec_ok, rec_ok.ttfat_ms, timeout_ms) == 800.0

    rec_fail = make_req(ok=False, ttfat_ms=None)
    assert effective_latency(rec_fail, rec_fail.ttfat_ms, timeout_ms) == timeout_ms

    # Even if an erroneous request has a partial ttfat_ms, ok=False forces timeout
    rec_fail_with_ttfat = make_req(ok=False, ttfat_ms=500.0)
    assert (
        effective_latency(rec_fail_with_ttfat, rec_fail_with_ttfat.ttfat_ms, timeout_ms)
        == timeout_ms
    )


def stored(request_id: int, **fields: object) -> StoredRequest:
    base = make_req(ok=True, ttfat_ms=500.0)
    values: dict[str, object] = {"suite": "single", "ttft_ms": 200.0, **fields}
    return StoredRequest(id=request_id, record=replace(base, **values))


def rows(**values: float | None) -> dict[str, ScoreRecord]:
    result: dict[str, ScoreRecord] = {}
    for key, value in values.items():
        scorer, _, name = key.partition("__")
        result[f"{scorer}.{name}"] = ScoreRecord(scorer, name, value)
    return result


CONFIG = StatsConfig(bootstrap_resamples=50)


def test_summarize_mode_counts_the_answers_with_and_with_no_verdict() -> None:
    requests = [
        stored(1, cost_usd=0.002, tok_per_s=80.0, reasoning_tokens=10, answer_tokens=40),
        stored(2, served_by="upstream-a"),
        stored(3, ok=False, error_kind="empty_output", ttfat_ms=None),
        stored(4, expect_refusal=True),
    ]
    scores = {
        1: rows(
            derived__quality=0.9,
            derived__question_correct=1.0,
            derived__false_refusal=0.0,
            judge__utility=5.0,
            deterministic__no_preamble=1.0,
        ),
        # A judge failure: the answer needs a verdict and has none.
        2: rows(derived__quality=None, derived__question_correct=None, judge__failure=1.0),
        3: rows(derived__quality=0.0, derived__question_correct=0.0),
        4: rows(
            derived__quality=1.0,
            derived__false_acceptance=0.0,
            judge__utility=5.0,
            deterministic__no_preamble=0.0,
        ),
    }
    summary = summarize_mode("m1", requests, scores, 60_000.0, CONFIG)
    assert summary.requests == 4
    assert summary.errors == 1
    assert summary.empty_rate == 0.25
    assert (summary.judged, summary.unjudged, summary.judge_failures) == (2, 1, 1)
    assert summary.quality is not None
    assert summary.quality.value == pytest.approx((0.9 + 0.0 + 1.0) / 3)
    assert summary.question_accuracy is not None
    assert summary.question_accuracy.value == pytest.approx(0.5)
    assert summary.utility_mean == pytest.approx(1.0)
    assert summary.false_refusal_rate == 0.0
    assert summary.false_acceptance_rate == 0.0
    assert summary.preamble_rate == pytest.approx(0.5)
    assert summary.ttfat_p95_ms.value > 500.0
    assert summary.cost_mean_usd == pytest.approx(0.002)
    assert summary.tok_per_s_p50 == 80.0
    assert summary.served_by == {"upstream-a": 1}
    assert summary.meeting_cold_ttfat_p50_ms is None


def test_a_mode_with_no_verdict_at_all_has_no_quality_estimate() -> None:
    requests = [stored(1), stored(2), stored(3, ok=False, ttfat_ms=None)]
    scores = {
        1: rows(derived__quality=None, derived__question_correct=None),
        2: rows(derived__quality=None, derived__question_correct=None),
        # The failure has a quality of zero with no judge. It alone is no sample of the mode.
        3: rows(derived__quality=0.0, derived__question_correct=0.0),
    }
    summary = summarize_mode("m1", requests, scores, 60_000.0, CONFIG)
    assert (summary.judged, summary.unjudged) == (0, 2)
    assert summary.quality is None
    assert summary.question_accuracy is None
    # A mode whose answers all have a value with no judge keeps its estimate.
    only_failures = summarize_mode("m1", requests[2:], {3: scores[3]}, 60_000.0, CONFIG)
    assert only_failures.quality is not None and only_failures.quality.value == 0.0


def test_summarize_mode_keeps_the_meeting_clicks_apart() -> None:
    requests = [
        stored(1),
        stored(2, suite="meeting", cache_state="cold", ttfat_ms=900.0),
        stored(
            3,
            suite="meeting",
            cache_state="warm",
            ttfat_ms=400.0,
            cached_tokens=80,
            prompt_tokens=100,
        ),
    ]
    summary = summarize_mode("m1", requests, {}, 60_000.0, CONFIG)
    assert summary.requests == 1
    assert summary.meeting_cold_ttfat_p50_ms is not None
    assert summary.meeting_cold_ttfat_p50_ms.value == 900.0
    assert summary.meeting_warm_ttfat_p50_ms is not None
    assert summary.meeting_warm_ttfat_p50_ms.value == 400.0
    assert summary.meeting_warm_cached_share == pytest.approx(0.8)


def test_a_summary_file_that_is_absent_or_not_valid_is_a_storage_error(tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="not found"):
        load_summary(tmp_path / "absent.json")
    (tmp_path / "bad.json").write_text('{"run_id": 1}', encoding="utf-8")
    with pytest.raises(StorageError, match="not a valid summary"):
        load_summary(tmp_path / "bad.json")
