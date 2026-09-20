"""The Markdown report: its sections, the order of its rows and the text of other parties."""

from dataclasses import replace

from helpers import fake_mode
from modebench.config import RoleSlo
from modebench.decision.decide import decide
from modebench.decision.regression import compare_summaries
from modebench.report.markdown import _natural, cell, mode_rows, render_report
from modebench.stats.aggregate import ModeSummary, RunSummary
from modebench.stats.percentiles import Estimate
from modebench.storage.records import RequestRecord, ScoreRecord, StoredRequest

ROLES = {"fast": RoleSlo(ttfat_p95_ms_max=2000.0, question_accuracy_min=0.8)}


def estimate(value: float, spread: float = 0.05) -> Estimate:
    return Estimate(value=value, ci_low=value * (1 - spread), ci_high=value * (1 + spread), n=40)


def mode_summary(mode_id: str, *, p95: float = 1500.0, quality: float | None = 0.9) -> ModeSummary:
    return ModeSummary(
        mode_id=mode_id,
        requests=40,
        errors=1,
        error_rate=0.025,
        ttft_p50_ms=estimate(300.0),
        ttft_p95_ms=estimate(500.0),
        ttfat_p50_ms=estimate(900.0),
        ttfat_p95_ms=estimate(p95),
        total_p50_ms=estimate(2000.0),
        total_p95_ms=estimate(3000.0),
        quality=None if quality is None else estimate(quality),
        question_accuracy=None if quality is None else estimate(0.9),
        judged=0 if quality is None else 38,
        unjudged=39 if quality is None else 1,
        judge_failures=0 if quality is None else 1,
        cost_mean_usd=0.001,
        served_by={"Upstream | [evil](http://x)": 40},
        meeting_cold_ttfat_p50_ms=estimate(1200.0),
        meeting_warm_ttfat_p50_ms=estimate(700.0),
        meeting_warm_ttfat_p95_ms=estimate(900.0),
        meeting_warm_cached_share=0.8,
    )


def run_summary(*modes: ModeSummary, **fields: object) -> RunSummary:
    base = RunSummary(
        run_id="20260901T000000Z-abc123",
        suite="analyze",
        profile="quick",
        started_at="2026-09-01T00:00:00+00:00",
        status="completed",
        dry_run=False,
        git_sha="abcdef1",
        git_dirty=True,
        config_hash="c" * 64,
        dataset_hash="d" * 64,
        prompt_hash="p" * 64,
        timeout_s=60.0,
        judge_model="openrouter:judge",
        modes=list(modes),
    )
    return base.model_copy(update=fields)


def request(request_id: int, mode_id: str, **fields: object) -> StoredRequest:
    record = RequestRecord(
        run_id="r",
        seq=request_id,
        mode_id=mode_id,
        suite="single",
        item_id="case-01|trunc100|5m",
        case_id="case-01",
        variant_id="case-01|trunc100",
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
        started_at="2026-09-01T00:00:01+00:00",
        ok=True,
        error_kind=None,
        error_message=None,
        http_status=200,
        first_chunk_ms=100.0,
        ttft_ms=200.0,
        first_reasoning_ms=None,
        ttfat_ms=800.0,
        total_ms=1500.0,
        prompt_tokens=None,
        completion_tokens=None,
        reasoning_tokens=None,
        answer_tokens=None,
        cached_tokens=None,
        tok_per_s=None,
        served_by=None,
        cost_usd=None,
        cost_source="unknown",
        prompt_chars=10,
        prompt_sha="sha",
        answer="answer text that must never be in a report",
    )
    return StoredRequest(id=request_id, record=replace(record, **fields))


def quality_row(value: float) -> dict[str, ScoreRecord]:
    return {"derived.quality": ScoreRecord("derived", "quality", value)}


def test_labels_sort_by_their_number_and_not_by_their_digits() -> None:
    labels = ["WER 0.2", "WER 0.05", "WER 0.15", "WER 0.1"]
    assert sorted(labels, key=_natural) == ["WER 0.05", "WER 0.1", "WER 0.15", "WER 0.2"]
    assert sorted(["100%", "50%", "80%"], key=_natural) == ["50%", "80%", "100%"]
    assert sorted(["30 min", "5 min"], key=_natural) == ["5 min", "30 min"]
    assert _natural("no number") == (0.0, "no number")


def test_a_cell_cannot_break_a_table_or_make_a_link() -> None:
    assert cell("Upstream | [evil](http://x)\n<img>") == r"Upstream \| \[evil\](http://x) \<img\>"
    assert cell("plain text") == "plain text"


def test_the_report_has_each_section_and_no_answer_text() -> None:
    modes = {
        "routed": fake_mode("routed").model_copy(update={"overhead_of": "direct"}),
        "direct": fake_mode("direct"),
        "blind": fake_mode("blind"),
    }
    summary = run_summary(
        mode_summary("routed", p95=1800.0),
        mode_summary("direct", p95=1500.0),
        mode_summary("blind", quality=None),
    )
    requests = [
        request(1, "routed"),
        request(2, "direct", variant_kind="asr_noise", asr_wer=0.15, duration_min=30),
        request(3, "direct", variant_kind="asr_noise", asr_wer=0.05),
        request(4, "blind", ok=False, error_kind="timeout", ttfat_ms=None),
    ]
    scores = {1: quality_row(0.9), 2: quality_row(0.7), 3: quality_row(0.8)}
    baseline = run_summary(
        mode_summary("routed", p95=900.0), mode_summary("retired"), profile="full"
    )
    text = render_report(
        summary,
        decide(ROLES, summary, modes),
        ROLES,
        modes,
        requests,
        scores,
        images={"pareto": "pareto.png", "latency": "latency.png"},
        baseline=baseline,
        comparison=compare_summaries(summary, baseline, 0.20),
    )
    for heading in (
        "## Decision",
        "## Modes",
        "## Pareto front",
        "## Route overhead",
        "## Meeting scenario",
        "## Drill-down",
        "### Mean quality for each synthetic word error rate",
        "## Difference from the baseline",
        "## Failures",
    ):
        assert heading in text
    assert "with changes that are not committed" in text
    assert "`blind` is not eligible: no quality data" in text
    assert "| `routed` | `direct` | +0 | +300 |" in text
    assert r"Upstream \| \[evil\](http://x) (40)" in text
    assert text.index("WER 0.05") < text.index("WER 0.15")
    assert "Warning: the profile is quick in this run and full in the baseline." in text
    assert "`routed`, ttfat_p95_ms" in text
    assert "`retired` is in the baseline only." in text
    assert "`direct` is in this run only." in text
    assert "| `blind` | timeout | 1 |" in text
    assert "Answers that need a verdict and have none" in text
    assert "`blind`: 39 of 39" in text
    assert "Run not complete" not in text
    assert "answer text that must never be in a report" not in text
    assert text.endswith("\n")


def test_the_report_flags_a_dry_run_and_a_run_that_is_not_complete() -> None:
    summary = run_summary(
        mode_summary("blind", quality=None), dry_run=True, status="failed", judge_model=""
    )
    text = render_report(summary, [], ROLES, {}, [], {})
    assert "**Dry run.**" in text
    assert "**Run not complete** (status `failed`)" in text
    assert "No mode has a quality estimate" in text
    assert "No request failed." in text
    assert "| Judge | none |" in text
    assert mode_rows(summary.modes) == ["blind: TTFAT p95 1500 ms, quality n/a, errors 2.5%"]


def test_a_baseline_of_zero_gives_no_division() -> None:
    summary = run_summary(mode_summary("m"))
    zero = mode_summary("m").model_copy(
        update={"ttfat_p95_ms": Estimate(value=0.0, ci_low=0.0, ci_high=0.0, n=1)}
    )
    baseline = run_summary(zero)
    text = render_report(
        summary,
        [],
        ROLES,
        {},
        [],
        {},
        baseline=baseline,
        comparison=compare_summaries(summary, baseline, 0.20),
    )
    assert "| `m` | 0 | 1500 | n/a |" in text
