"""Tests for role decision rules, tie-breaking, SLO gating, and regression detection."""

from pathlib import Path

from helpers import fake_mode
from modebench.config import RoleSlo
from modebench.decision.decide import (
    STATUS_NONE_WITHIN_SLO,
    STATUS_OK,
    decide,
    decide_role,
    recommended_document,
    slo_problem,
    write_recommended,
)
from modebench.decision.regression import compare_summaries
from modebench.stats.aggregate import ModeSummary, RunSummary
from modebench.stats.percentiles import Estimate


def make_mode_summary(
    mode_id: str,
    *,
    ttfat_p95: float = 1500.0,
    ttfat_ci: tuple[float, float] = (1400.0, 1600.0),
    quality: float = 0.85,
    quality_ci: tuple[float, float] = (0.80, 0.90),
    accuracy: float = 0.9,
    error_rate: float = 0.0,
    cost_usd: float = 0.005,
) -> ModeSummary:
    return ModeSummary(
        mode_id=mode_id,
        requests=50,
        errors=int(error_rate * 50),
        error_rate=error_rate,
        ttft_p50_ms=Estimate(value=300.0, ci_low=250.0, ci_high=350.0, n=50),
        ttft_p95_ms=Estimate(value=500.0, ci_low=450.0, ci_high=550.0, n=50),
        ttfat_p50_ms=Estimate(value=1000.0, ci_low=900.0, ci_high=1100.0, n=50),
        ttfat_p95_ms=Estimate(value=ttfat_p95, ci_low=ttfat_ci[0], ci_high=ttfat_ci[1], n=50),
        total_p50_ms=Estimate(value=2000.0, ci_low=1800.0, ci_high=2200.0, n=50),
        total_p95_ms=Estimate(value=3000.0, ci_low=2800.0, ci_high=3200.0, n=50),
        quality=Estimate(value=quality, ci_low=quality_ci[0], ci_high=quality_ci[1], n=50),
        question_accuracy=Estimate(
            value=accuracy, ci_low=accuracy - 0.05, ci_high=accuracy + 0.05, n=50
        ),
        cost_mean_usd=cost_usd,
    )


def test_slo_problem_checks_latency_accuracy_and_errors() -> None:
    slo = RoleSlo(ttfat_p95_ms_max=2000.0, question_accuracy_min=0.8, error_rate_max=0.05)

    ok_summary = make_mode_summary("ok", ttfat_p95=1800.0, accuracy=0.85, error_rate=0.02)
    assert slo_problem(ok_summary, slo) is None

    slow_summary = make_mode_summary("slow", ttfat_p95=2500.0, accuracy=0.85)
    problem = slo_problem(slow_summary, slo)
    assert problem is not None
    assert "TTFAT p95 2500 ms is above 2000 ms" in problem

    inaccurate_summary = make_mode_summary("inaccurate", ttfat_p95=1000.0, accuracy=0.7)
    problem = slo_problem(inaccurate_summary, slo)
    assert problem is not None
    assert "question accuracy 0.70 is below 0.80" in problem

    error_summary = make_mode_summary("errored", error_rate=0.10)
    problem = slo_problem(error_summary, slo)
    assert problem is not None
    assert "error rate 0.100 is above 0.050" in problem


def test_decide_role_selects_clear_highest_quality() -> None:
    slo = RoleSlo(ttfat_p95_ms_max=5000.0, question_accuracy_min=0.7)
    modes = {"m_good": fake_mode("m_good"), "m_poor": fake_mode("m_poor")}

    good = make_mode_summary("m_good", quality=0.95, quality_ci=(0.92, 0.98))
    poor = make_mode_summary("m_poor", quality=0.75, quality_ci=(0.70, 0.80))

    dec = decide_role("fast", slo, [good, poor], modes)
    assert dec.status == STATUS_OK
    assert dec.winner == "m_good"
    assert "highest quality within the objective" in dec.reason


def test_decide_role_breaks_quality_tie_by_latency() -> None:
    slo = RoleSlo(ttfat_p95_ms_max=5000.0, question_accuracy_min=0.7)
    modes = {"m_fast": fake_mode("m_fast"), "m_slow": fake_mode("m_slow")}

    # Quality intervals overlap (0.85 vs 0.84 with overlapping bounds)
    fast = make_mode_summary(
        "m_fast",
        quality=0.84,
        quality_ci=(0.80, 0.88),
        ttfat_p95=1200.0,
        ttfat_ci=(1100.0, 1300.0),
    )
    slow = make_mode_summary(
        "m_slow",
        quality=0.85,
        quality_ci=(0.81, 0.89),
        ttfat_p95=3000.0,
        ttfat_ci=(2800.0, 3200.0),
    )

    dec = decide_role("fast", slo, [fast, slow], modes)
    assert dec.status == STATUS_OK
    assert dec.winner == "m_fast"
    assert "lowest TTFAT p95" in dec.reason


def test_decide_role_breaks_latency_tie_by_cost() -> None:
    slo = RoleSlo(ttfat_p95_ms_max=5000.0, question_accuracy_min=0.7)
    modes = {"m_cheap": fake_mode("m_cheap"), "m_costly": fake_mode("m_costly")}

    # Quality and latency intervals overlap
    cheap = make_mode_summary(
        "m_cheap",
        quality=0.85,
        quality_ci=(0.80, 0.90),
        ttfat_p95=1500.0,
        ttfat_ci=(1400.0, 1600.0),
        cost_usd=0.001,
    )
    costly = make_mode_summary(
        "m_costly",
        quality=0.86,
        quality_ci=(0.81, 0.91),
        ttfat_p95=1520.0,
        ttfat_ci=(1420.0, 1620.0),
        cost_usd=0.010,
    )

    dec = decide_role("fast", slo, [cheap, costly], modes)
    assert dec.status == STATUS_OK
    assert dec.winner == "m_cheap"
    assert "lowest cost" in dec.reason


def test_decide_role_handles_no_mode_within_slo() -> None:
    slo = RoleSlo(ttfat_p95_ms_max=1000.0, question_accuracy_min=0.95)
    modes = {"m_slow": fake_mode("m_slow")}
    slow = make_mode_summary("m_slow", ttfat_p95=2000.0, accuracy=0.8)

    dec = decide_role("fast", slo, [slow], modes)
    assert dec.status == STATUS_NONE_WITHIN_SLO
    assert dec.winner is None
    assert "m_slow" in dec.rejected


def test_compare_summaries_detects_latency_and_quality_regressions() -> None:
    base_m1 = make_mode_summary(
        "m1",
        ttfat_p95=1000.0,
        quality=0.85,
        quality_ci=(0.80, 0.90),
    )
    base_summary = RunSummary(
        run_id="run-base",
        suite="analyze",
        profile="smoke",
        started_at="2026-09-01T00:00:00Z",
        status="completed",
        dry_run=False,
        git_sha="abcdef1",
        git_dirty=False,
        config_hash="cfg1",
        dataset_hash="ds1",
        prompt_hash="pr1",
        timeout_s=60.0,
        judge_model="fake-judge",
        modes=[base_m1],
    )

    # 1. Healthy run: within 20% latency and within quality interval
    curr_ok = make_mode_summary(
        "m1",
        ttfat_p95=1150.0,  # +15% <= 20%
        quality=0.82,  # >= 0.80 (ci_low)
        quality_ci=(0.78, 0.86),
    )
    run_ok = base_summary.model_copy(update={"run_id": "run-ok", "modes": [curr_ok]})
    cmp_ok = compare_summaries(run_ok, base_summary, p95_increase_ratio=0.20)
    assert not cmp_ok.failed
    assert len(cmp_ok.regressions) == 0

    # 2. Regressed latency: +30% > 20%
    curr_slow = make_mode_summary("m1", ttfat_p95=1350.0, quality=0.85)
    run_slow = base_summary.model_copy(update={"run_id": "run-slow", "modes": [curr_slow]})
    cmp_slow = compare_summaries(run_slow, base_summary, p95_increase_ratio=0.20)
    assert cmp_slow.failed
    assert any(r.metric == "ttfat_p95_ms" for r in cmp_slow.regressions)

    # 3. Regressed quality: below baseline ci_low (0.75 < 0.80)
    curr_poor = make_mode_summary("m1", ttfat_p95=1000.0, quality=0.75)
    run_poor = base_summary.model_copy(update={"run_id": "run-poor", "modes": [curr_poor]})
    cmp_poor = compare_summaries(run_poor, base_summary, p95_increase_ratio=0.20)
    assert cmp_poor.failed
    assert any(r.metric == "quality" for r in cmp_poor.regressions)


def test_recommended_document_and_write(tmp_path: Path) -> None:
    slo = RoleSlo(ttfat_p95_ms_max=3000.0, question_accuracy_min=0.7)
    modes = {"m1": fake_mode("m1")}
    summary = RunSummary(
        run_id="run-1",
        suite="analyze",
        profile="smoke",
        started_at="2026-09-01T00:00:00Z",
        status="completed",
        dry_run=False,
        git_sha="abcdef1",
        git_dirty=False,
        config_hash="cfg1",
        dataset_hash="ds1",
        prompt_hash="pr1",
        timeout_s=60.0,
        judge_model="fake-judge",
        modes=[make_mode_summary("m1")],
    )

    decisions = decide({"fast": slo}, summary, modes)
    doc = recommended_document(decisions, summary, modes, None, "2026-09-01T01:00:00Z")
    assert doc["run_id"] == "run-1"
    assert "fast" in doc["roles"]
    assert doc["roles"]["fast"]["mode"] == "m1"

    out_path = tmp_path / "recommended_modes.toml"
    written = write_recommended(doc, out_path)
    assert written.is_file()
    content = written.read_text(encoding="utf-8")
    assert 'mode = "m1"' in content
