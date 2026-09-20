"""Tests for statistical routines: percentiles, bootstrap intervals, Pareto fronts, and aggregation."""

import pytest

from modebench.stats.aggregate import (
    effective_latency,
)
from modebench.stats.pareto import ParetoPoint, dominates, pareto_front
from modebench.stats.percentiles import (
    Estimate,
    bootstrap_mean,
    bootstrap_percentile,
    percentile,
)
from modebench.storage.records import RequestRecord


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


def test_effective_latency_counts_failure_as_timeout() -> None:
    timeout_ms = 60000.0
    rec_ok = RequestRecord(
        seq=1,
        run_id="r1",
        suite="analyze",
        mode_id="m1",
        item_id="it1",
        repetition=1,
        warmup=False,
        cache_state="cold",
        ok=True,
        total_ms=1500.0,
        ttfat_ms=800.0,
    )
    assert effective_latency(rec_ok, rec_ok.ttfat_ms, timeout_ms) == 800.0

    rec_fail = RequestRecord(
        seq=2,
        run_id="r1",
        suite="analyze",
        mode_id="m1",
        item_id="it2",
        repetition=1,
        warmup=False,
        cache_state="cold",
        ok=False,
        error_kind="timeout",
        total_ms=timeout_ms,
        ttfat_ms=None,
    )
    assert effective_latency(rec_fail, rec_fail.ttfat_ms, timeout_ms) == timeout_ms

    # Even if an erroneous request has a partial ttfat_ms, ok=False forces timeout
    rec_fail_with_ttfat = RequestRecord(
        seq=3,
        run_id="r1",
        suite="analyze",
        mode_id="m1",
        item_id="it3",
        repetition=1,
        warmup=False,
        cache_state="cold",
        ok=False,
        error_kind="network_error",
        total_ms=2000.0,
        ttfat_ms=500.0,
    )
    assert (
        effective_latency(rec_fail_with_ttfat, rec_fail_with_ttfat.ttfat_ms, timeout_ms)
        == timeout_ms
    )
