"""Writes the Markdown report of a run.

The report goes from the decision down to the detail: the mode of each role,
the table of the modes, the Pareto front, the cost of the gateway route, the
cold and the warm cache, the drill-down for each case, and the difference from
the baseline. It holds identifiers and numbers. It holds no transcript text,
so a report of a private dataset shows no meeting content.
"""

from collections.abc import Mapping, Sequence

from modebench.config import Mode, RoleSlo
from modebench.decision.decide import RoleDecision
from modebench.decision.regression import Comparison
from modebench.stats.aggregate import ModeSummary, RunSummary, effective_latency
from modebench.stats.pareto import ParetoPoint, pareto_front
from modebench.stats.percentiles import Estimate, percentile
from modebench.storage.records import ScoreRecord, StoredRequest

REPORT_NAME = "report.md"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _num(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _ms(estimate: Estimate | None) -> str:
    if estimate is None:
        return "n/a"
    return f"{estimate.value:.0f} ({estimate.ci_low:.0f} to {estimate.ci_high:.0f})"


def _unit(estimate: Estimate | None) -> str:
    if estimate is None:
        return "n/a"
    return f"{estimate.value:.3f} ({estimate.ci_low:.3f} to {estimate.ci_high:.3f})"


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.5f}"


def _header(summary: RunSummary) -> list[str]:
    dirty = " (with changes that are not committed)" if summary.git_dirty else ""
    lines = [f"# modebench report: {summary.run_id}", ""]
    if summary.dry_run:
        lines += [
            "> **Dry run.** The fake provider made these numbers. Do not use them for a decision.",
            "",
        ]
    rows = [
        ["Profile", summary.profile],
        ["Status", summary.status],
        ["Started", summary.started_at],
        ["Commit", f"`{summary.git_sha}`{dirty}"],
        ["Config hash", f"`{summary.config_hash[:16]}`"],
        ["Dataset hash", f"`{summary.dataset_hash[:16]}`"],
        ["Prompt hash", f"`{summary.prompt_hash[:16]}`"],
        ["Judge", summary.judge_model or "none"],
        ["Timeout", f"{summary.timeout_s:g} s"],
    ]
    return lines + _table(["Item", "Value"], rows) + [""]


def _decisions(decisions: Sequence[RoleDecision], roles: Mapping[str, RoleSlo]) -> list[str]:
    lines = ["## Decision", ""]
    for decision in decisions:
        slo = roles.get(decision.role)
        objective = ""
        if slo is not None:
            objective = (
                f" Objective: TTFAT p95 <= {slo.ttfat_p95_ms_max:.0f} ms and "
                f"question accuracy >= {slo.question_accuracy_min:.2f}."
            )
        winner = f"`{decision.winner}`" if decision.winner else "no mode"
        lines.append(f"- **{decision.role}**: {winner}. Reason: {decision.reason}.{objective}")
        for mode_id, problem in sorted(decision.rejected.items()):
            lines.append(f"  - `{mode_id}` is not eligible: {problem}.")
    return lines + [""]


def _modes_table(summary: RunSummary) -> list[str]:
    headers = [
        "Mode",
        "Requests",
        "Errors",
        "TTFT p50 ms",
        "TTFAT p50 ms",
        "TTFAT p95 ms",
        "Quality",
        "Question accuracy",
        "False refusal",
        "False acceptance",
        "Preamble",
        "tok/s p50",
        "Reasoning tokens",
        "USD for one request",
        "Served by",
    ]
    rows = []
    for mode in sorted(summary.modes, key=lambda item: item.ttfat_p95_ms.value):
        served = ", ".join(f"{name} ({count})" for name, count in sorted(mode.served_by.items()))
        rows.append(
            [
                f"`{mode.mode_id}`",
                str(mode.requests),
                f"{mode.errors} ({_pct(mode.error_rate)})",
                _ms(mode.ttft_p50_ms),
                _ms(mode.ttfat_p50_ms),
                _ms(mode.ttfat_p95_ms),
                _unit(mode.quality),
                _unit(mode.question_accuracy),
                _pct(mode.false_refusal_rate),
                _pct(mode.false_acceptance_rate),
                _pct(mode.preamble_rate),
                _num(mode.tok_per_s_p50, 1),
                _num(mode.reasoning_tokens_mean, 0),
                _usd(mode.cost_mean_usd),
                served or "n/a",
            ]
        )
    note = (
        "Latencies are of the single requests, which always have a cold cache. "
        "A failed request counts as the timeout. Intervals are 95% bootstrap intervals."
    )
    return ["## Modes", "", note, ""] + _table(headers, rows) + [""]


def _pareto(summary: RunSummary, images: Mapping[str, str]) -> list[str]:
    points = [
        ParetoPoint(mode.mode_id, mode.quality.value, mode.ttfat_p95_ms.value)
        for mode in summary.modes
        if mode.quality is not None
    ]
    lines = ["## Pareto front: quality against TTFAT p95", ""]
    if "pareto" in images:
        lines += [f"![Pareto front]({images['pareto']})", ""]
    if not points:
        return lines + ["No mode has a quality estimate, so there is no front.", ""]
    rows = [
        [f"`{point.key}`", f"{point.latency_ms:.0f}", f"{point.quality:.3f}"]
        for point in pareto_front(points)
    ]
    lines += _table(["Mode on the front", "TTFAT p95 ms", "Quality"], rows) + [""]
    if "latency" in images:
        lines += [f"![TTFAT of each mode]({images['latency']})", ""]
    return lines


def _overhead(summary: RunSummary, modes: Mapping[str, Mode]) -> list[str]:
    rows = []
    for mode_id, mode in sorted(modes.items()):
        if mode.overhead_of is None:
            continue
        routed = summary.mode(mode_id)
        direct = summary.mode(mode.overhead_of)
        if routed is None or direct is None:
            continue
        p50 = routed.ttfat_p50_ms.value - direct.ttfat_p50_ms.value
        p95 = routed.ttfat_p95_ms.value - direct.ttfat_p95_ms.value
        rows.append([f"`{mode_id}`", f"`{mode.overhead_of}`", f"{p50:+.0f}", f"{p95:+.0f}"])
    if not rows:
        return []
    note = "The difference is the first mode minus the second. A positive value is time that the route adds."
    headers = ["Routed mode", "Direct mode", "TTFAT p50 difference ms", "TTFAT p95 difference ms"]
    return ["## Route overhead", "", note, ""] + _table(headers, rows) + [""]


def _meeting(summary: RunSummary) -> list[str]:
    rows = [
        [
            f"`{mode.mode_id}`",
            _ms(mode.meeting_cold_ttfat_p50_ms),
            _ms(mode.meeting_warm_ttfat_p50_ms),
            _ms(mode.meeting_warm_ttfat_p95_ms),
            _pct(mode.meeting_warm_cached_share),
        ]
        for mode in summary.modes
        if mode.meeting_cold_ttfat_p50_ms is not None or mode.meeting_warm_ttfat_p50_ms is not None
    ]
    if not rows:
        return []
    headers = [
        "Mode",
        "Cold TTFAT p50 ms",
        "Warm TTFAT p50 ms",
        "Warm TTFAT p95 ms",
        "Cached share of warm prompts",
    ]
    note = (
        "A meeting is a sequence of clicks on a transcript that grows. The first click has a "
        "cold cache. Each later click shares the start of its prompt with the click before it."
    )
    return ["## Meeting scenario: cold and warm cache", "", note, ""] + _table(headers, rows) + [""]


def _quality_of(scores: Mapping[int, Mapping[str, ScoreRecord]], request_id: int) -> float | None:
    record = scores.get(request_id, {}).get("derived.quality")
    return None if record is None else record.value


def _pivot(
    requests: Sequence[StoredRequest],
    mode_ids: Sequence[str],
    row_key: dict[int, str],
    cell: dict[int, float],
    median: bool,
) -> list[list[str]]:
    values: dict[str, dict[str, list[float]]] = {}
    for item in requests:
        if item.id not in row_key or item.id not in cell:
            continue
        values.setdefault(row_key[item.id], {}).setdefault(item.record.mode_id, []).append(
            cell[item.id]
        )
    rows = []
    for key in sorted(values, key=_natural):
        row = [key]
        for mode_id in mode_ids:
            data = values[key].get(mode_id, [])
            if not data:
                row.append("n/a")
            elif median:
                row.append(f"{percentile(data, 50):.0f}")
            else:
                row.append(f"{sum(data) / len(data):.2f}")
        rows.append(row)
    return rows


def _natural(text: str) -> tuple[int, str]:
    digits = "".join(char for char in text if char.isdigit())
    return (int(digits) if digits else 0, text)


def _drilldown(
    summary: RunSummary,
    requests: Sequence[StoredRequest],
    scores: Mapping[int, Mapping[str, ScoreRecord]],
) -> list[str]:
    singles = [item for item in requests if item.record.suite == "single"]
    if not singles:
        return []
    mode_ids = [mode.mode_id for mode in summary.modes]
    headers_tail = [f"`{mode_id}`" for mode_id in mode_ids]
    timeout_ms = summary.timeout_s * 1000.0
    quality: dict[int, float] = {}
    for item in singles:
        value = _quality_of(scores, item.id)
        if value is not None:
            quality[item.id] = value
    latency = {
        item.id: effective_latency(item.record, item.record.ttfat_ms, timeout_ms)
        for item in singles
    }
    by_case = {item.id: item.record.case_id for item in singles}
    by_cut = {
        item.id: f"{item.record.truncation_pct}%"
        for item in singles
        if item.record.variant_kind == "truncation"
    }
    by_wer = {
        item.id: f"WER {item.record.asr_wer:g}"
        for item in singles
        if item.record.variant_kind == "asr_noise"
    }
    by_length = {item.id: f"{item.record.duration_min} min" for item in singles}
    lines = ["## Drill-down", ""]
    lines += ["### Mean quality for each case", ""]
    lines += _table(["Case", *headers_tail], _pivot(singles, mode_ids, by_case, quality, False))
    lines += ["", "### Mean quality for each truncation of the question", ""]
    lines += _table(
        ["Words said", *headers_tail], _pivot(singles, mode_ids, by_cut, quality, False)
    )
    wer_rows = _pivot(singles, mode_ids, by_wer, quality, False)
    if wer_rows:
        lines += ["", "### Mean quality for each synthetic word error rate", ""]
        lines += _table(["Noise", *headers_tail], wer_rows)
    lines += ["", "### TTFAT p50 (ms) for each transcript length", ""]
    lines += _table(["Length", *headers_tail], _pivot(singles, mode_ids, by_length, latency, True))
    return lines + [""]


def _failures(requests: Sequence[StoredRequest], summary: RunSummary) -> list[str]:
    counts: dict[tuple[str, str], int] = {}
    for item in requests:
        if not item.record.ok:
            key = (item.record.mode_id, item.record.error_kind or "unknown")
            counts[key] = counts.get(key, 0) + 1
    judge_failures = [(mode.mode_id, mode.judge_failures) for mode in summary.modes]
    lines = ["## Failures", ""]
    if counts:
        rows = [
            [f"`{mode_id}`", kind, str(count)] for (mode_id, kind), count in sorted(counts.items())
        ]
        lines += _table(["Mode", "Kind", "Requests"], rows) + [""]
    else:
        lines += ["No request failed.", ""]
    failed = [(mode_id, count) for mode_id, count in judge_failures if count]
    if failed:
        text = ", ".join(f"`{mode_id}`: {count}" for mode_id, count in failed)
        lines += [f"Answers with no valid verdict from the judge: {text}.", ""]
    return lines


def _baseline(
    summary: RunSummary, baseline: RunSummary | None, comparison: Comparison | None
) -> list[str]:
    if baseline is None or comparison is None:
        return []
    lines = [f"## Difference from the baseline `{baseline.run_id}`", ""]
    rows = []
    for mode in summary.modes:
        before = baseline.mode(mode.mode_id)
        if before is None:
            continue
        change = (mode.ttfat_p95_ms.value / before.ttfat_p95_ms.value - 1.0) * 100.0
        quality_before = before.quality.value if before.quality is not None else None
        quality_now = mode.quality.value if mode.quality is not None else None
        rows.append(
            [
                f"`{mode.mode_id}`",
                f"{before.ttfat_p95_ms.value:.0f}",
                f"{mode.ttfat_p95_ms.value:.0f}",
                f"{change:+.1f}%",
                _num(quality_before, 3),
                _num(quality_now, 3),
            ]
        )
    headers = [
        "Mode",
        "Baseline TTFAT p95 ms",
        "TTFAT p95 ms",
        "Change",
        "Baseline quality",
        "Quality",
    ]
    lines += _table(headers, rows) + [""]
    if comparison.regressions:
        lines.append("**Regressions:**")
        for item in comparison.regressions:
            lines.append(
                f"- `{item.mode_id}`, {item.metric}: {item.baseline:.3f} before, "
                f"{item.current:.3f} now, {item.detail}."
            )
    else:
        lines.append("No regression.")
    for mode_id in comparison.only_in_baseline:
        lines.append(f"- `{mode_id}` is in the baseline only.")
    for mode_id in comparison.only_in_current:
        lines.append(f"- `{mode_id}` is in this run only.")
    return lines + [""]


def render_report(
    summary: RunSummary,
    decisions: Sequence[RoleDecision],
    roles: Mapping[str, RoleSlo],
    modes: Mapping[str, Mode],
    requests: Sequence[StoredRequest],
    scores: Mapping[int, Mapping[str, ScoreRecord]],
    *,
    images: Mapping[str, str] | None = None,
    baseline: RunSummary | None = None,
    comparison: Comparison | None = None,
) -> str:
    """Return the report of a run as Markdown. `images` maps a chart name to its file name."""
    shown = images if images is not None else {}
    lines: list[str] = []
    lines += _header(summary)
    lines += _decisions(decisions, roles)
    lines += _modes_table(summary)
    lines += _pareto(summary, shown)
    lines += _overhead(summary, modes)
    lines += _meeting(summary)
    lines += _drilldown(summary, requests, scores)
    lines += _baseline(summary, baseline, comparison)
    lines += _failures(requests, summary)
    return "\n".join(lines).rstrip() + "\n"


def mode_rows(summaries: Sequence[ModeSummary]) -> list[str]:
    """Return one short line for each mode, for the command line."""
    return [
        f"{mode.mode_id}: TTFAT p95 {mode.ttfat_p95_ms.value:.0f} ms, "
        f"quality {_num(mode.quality.value if mode.quality is not None else None, 3)}, "
        f"errors {_pct(mode.error_rate)}"
        for mode in summaries
    ]
