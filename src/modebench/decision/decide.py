"""Selects the mode of each role.

The rule has three steps, and each step is a fact that the report can show:

1. A mode is eligible for a role if it serves the role and meets the
   objective of the role: the p95 of the time to the first answer token, the
   accuracy of the inferred question and, if it is set, the error rate.
2. The eligible mode with the highest quality wins.
3. A mode whose quality interval overlaps that of the best mode is equal to
   it in quality. Equal modes are ordered by p95 latency. Modes whose p95
   intervals also overlap are ordered by cost. The identifier breaks a tie
   that is left, so that the same data always gives the same decision.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modebench.config import Mode, ModesFile, RoleSlo
from modebench.decision.toml_writer import dumps
from modebench.stats.aggregate import ModeSummary, RunSummary
from modebench.stats.percentiles import Estimate

STATUS_OK = "ok"
STATUS_NONE_WITHIN_SLO = "no_mode_within_slo"
RECOMMENDED_NAME = "recommended_modes.toml"


@dataclass(frozen=True, slots=True)
class RoleDecision:
    """The decision for one role, with the facts that led to it."""

    role: str
    status: str
    winner: str | None
    reason: str
    eligible: list[str] = field(default_factory=list)
    tied_on_quality: list[str] = field(default_factory=list)
    tied_on_latency: list[str] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)


def slo_problem(summary: ModeSummary, slo: RoleSlo) -> str | None:
    """Return why a mode does not meet an objective, or None if it meets it."""
    if summary.quality is None or summary.question_accuracy is None:
        return "no quality data (the run had no judge)"
    p95 = summary.ttfat_p95_ms.value
    if p95 > slo.ttfat_p95_ms_max:
        return f"TTFAT p95 {p95:.0f} ms is above {slo.ttfat_p95_ms_max:.0f} ms"
    accuracy = summary.question_accuracy.value
    if accuracy < slo.question_accuracy_min:
        return f"question accuracy {accuracy:.2f} is below {slo.question_accuracy_min:.2f}"
    if slo.error_rate_max is not None and summary.error_rate > slo.error_rate_max:
        return f"error rate {summary.error_rate:.3f} is above {slo.error_rate_max:.3f}"
    return None


def _cost(summary: ModeSummary) -> float:
    return summary.cost_mean_usd if summary.cost_mean_usd is not None else float("inf")


def decide_role(
    role: str,
    slo: RoleSlo,
    summaries: Sequence[ModeSummary],
    modes: Mapping[str, Mode],
) -> RoleDecision:
    """Return the decision for one role."""
    rejected: dict[str, str] = {}
    eligible: list[ModeSummary] = []
    for summary in summaries:
        mode = modes.get(summary.mode_id)
        if mode is not None and not mode.serves_role(role):
            rejected[summary.mode_id] = f"the mode does not serve the role {role}"
            continue
        problem = slo_problem(summary, slo)
        if problem is not None:
            rejected[summary.mode_id] = problem
            continue
        eligible.append(summary)
    if not eligible:
        return RoleDecision(
            role=role,
            status=STATUS_NONE_WITHIN_SLO,
            winner=None,
            reason="no mode meets the objective of the role",
            rejected=rejected,
        )
    qualities: dict[str, Estimate] = {}
    for candidate in eligible:
        if candidate.quality is not None:
            qualities[candidate.mode_id] = candidate.quality
    best = max(eligible, key=lambda s: (qualities[s.mode_id].value, s.mode_id))
    best_quality = qualities[best.mode_id]
    tied_quality = [s for s in eligible if qualities[s.mode_id].overlaps(best_quality)]
    fastest = min(tied_quality, key=lambda s: (s.ttfat_p95_ms.value, s.mode_id))
    tied_latency = [s for s in tied_quality if s.ttfat_p95_ms.overlaps(fastest.ttfat_p95_ms)]
    winner = min(tied_latency, key=lambda s: (_cost(s), s.ttfat_p95_ms.value, s.mode_id))
    if len(tied_quality) == 1:
        reason = "highest quality within the objective, and no other interval overlaps it"
    elif len(tied_latency) == 1:
        reason = "quality intervals overlap, and this mode has the lowest TTFAT p95"
    else:
        reason = "quality and TTFAT p95 intervals overlap, and this mode has the lowest cost"
    return RoleDecision(
        role=role,
        status=STATUS_OK,
        winner=winner.mode_id,
        reason=reason,
        eligible=[s.mode_id for s in eligible],
        tied_on_quality=[s.mode_id for s in tied_quality],
        tied_on_latency=[s.mode_id for s in tied_latency],
        rejected=rejected,
    )


def decide(
    roles: Mapping[str, RoleSlo], summary: RunSummary, modes: Mapping[str, Mode]
) -> list[RoleDecision]:
    """Return one decision for each role of the bench file."""
    return [decide_role(role, slo, summary.modes, modes) for role, slo in roles.items()]


def modes_of_run(modes_json: str) -> dict[str, Mode]:
    """Return the modes that a run recorded, as they were when it ran."""
    parsed = [Mode.model_validate(item) for item in json.loads(modes_json)]
    return {mode.id: mode for mode in parsed}


def _role_table(
    decision: RoleDecision,
    summary: RunSummary,
    modes: Mapping[str, Mode],
    modes_file: ModesFile | None,
) -> dict[str, Any]:
    table: dict[str, Any] = {"status": decision.status, "reason": decision.reason}
    if decision.winner is None:
        table["rejected"] = dict(decision.rejected)
        return table
    mode = modes[decision.winner]
    stats = summary.mode(decision.winner)
    table.update({"mode": mode.id, "provider": mode.provider, "model": mode.model})
    if modes_file is not None and mode.provider in modes_file.providers:
        provider = modes_file.providers[mode.provider]
        table["base_url"] = provider.base_url
        table["api_key_env"] = provider.api_key_env
    if stats is not None and stats.quality is not None:
        table["quality"] = round(stats.quality.value, 4)
        table["quality_ci"] = [round(stats.quality.ci_low, 4), round(stats.quality.ci_high, 4)]
    if stats is not None:
        if stats.question_accuracy is not None:
            table["question_accuracy"] = round(stats.question_accuracy.value, 4)
        table["ttfat_p95_ms"] = round(stats.ttfat_p95_ms.value, 1)
        table["ttfat_p95_ci_ms"] = [
            round(stats.ttfat_p95_ms.ci_low, 1),
            round(stats.ttfat_p95_ms.ci_high, 1),
        ]
        table["error_rate"] = round(stats.error_rate, 4)
        if stats.cost_mean_usd is not None:
            table["cost_mean_usd"] = round(stats.cost_mean_usd, 6)
    table["tied_on_quality"] = list(decision.tied_on_quality)
    table["tied_on_latency"] = list(decision.tied_on_latency)
    table["params"] = dict(mode.params)
    return table


def recommended_document(
    decisions: Sequence[RoleDecision],
    summary: RunSummary,
    modes: Mapping[str, Mode],
    modes_file: ModesFile | None,
    generated_at: str,
) -> dict[str, Any]:
    """Return the content of the recommendations file."""
    return {
        "generated_at": generated_at,
        "run_id": summary.run_id,
        "profile": summary.profile,
        "git_sha": summary.git_sha,
        "dry_run": summary.dry_run,
        "config_hash": summary.config_hash,
        "dataset_hash": summary.dataset_hash,
        "prompt_hash": summary.prompt_hash,
        "roles": {
            decision.role: _role_table(decision, summary, modes, modes_file)
            for decision in decisions
        },
    }


def write_recommended(document: Mapping[str, Any], path: Path) -> Path:
    """Write the recommendations file."""
    header = "Generated by `modebench decide`. Do not edit: the next decision replaces this file."
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(document, header=header), encoding="utf-8")
    return path
