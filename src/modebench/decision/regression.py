"""Compares a run with a baseline and finds regressions.

Two facts make a regression, and each fact must be outside the noise of the
measurement. The p95 of the time to the first answer token went up by more
than the permitted ratio (20 percent by default), and its interval is fully
above the interval of the baseline. Or the quality interval is fully below the
quality interval of the baseline. Overlapping intervals are a tie, as in the
decision rule. A comparison of two point estimates would call the normal
scatter of two equal runs a regression.

A mode that is only in one of the two summaries is reported, and it is not a
regression. A difference in the profile, the dataset, the prompts, the timeout
or the judge is a warning: the two runs did not measure the same thing.
"""

from dataclasses import dataclass, field

from modebench.stats.aggregate import RunSummary


@dataclass(frozen=True, slots=True)
class Regression:
    """One metric of one mode that got worse."""

    mode_id: str
    metric: str
    baseline: float
    current: float
    detail: str


@dataclass(frozen=True, slots=True)
class Comparison:
    """The result of a comparison."""

    regressions: list[Regression]
    only_in_baseline: list[str]
    only_in_current: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        """Return True if the comparison found a regression."""
        return bool(self.regressions)


def comparability_warnings(current: RunSummary, baseline: RunSummary) -> list[str]:
    """Return one line for each condition that is not the same in the two runs."""
    pairs = (
        ("profile", current.profile, baseline.profile),
        ("dataset hash", current.dataset_hash[:16], baseline.dataset_hash[:16]),
        ("prompt hash", current.prompt_hash[:16], baseline.prompt_hash[:16]),
        ("timeout", f"{current.timeout_s:g} s", f"{baseline.timeout_s:g} s"),
        ("judge", current.judge_model or "none", baseline.judge_model or "none"),
    )
    return [
        f"the {name} is {now} in this run and {before} in the baseline"
        for name, now, before in pairs
        if now != before
    ]


def compare_summaries(
    current: RunSummary, baseline: RunSummary, p95_increase_ratio: float
) -> Comparison:
    """Return the regressions of `current` against `baseline`."""
    regressions: list[Regression] = []
    current_ids = {mode.mode_id for mode in current.modes}
    baseline_ids = {mode.mode_id for mode in baseline.modes}
    for now in current.modes:
        before = baseline.mode(now.mode_id)
        if before is None:
            continue
        limit = before.ttfat_p95_ms.value * (1.0 + p95_increase_ratio)
        above_limit = now.ttfat_p95_ms.value > limit
        outside_noise = now.ttfat_p95_ms.ci_low > before.ttfat_p95_ms.ci_high
        if above_limit and outside_noise:
            regressions.append(
                Regression(
                    mode_id=now.mode_id,
                    metric="ttfat_p95_ms",
                    baseline=before.ttfat_p95_ms.value,
                    current=now.ttfat_p95_ms.value,
                    detail=(
                        f"above the limit of {limit:.0f} ms (+{p95_increase_ratio:.0%}), and the "
                        f"interval starts at {now.ttfat_p95_ms.ci_low:.0f} ms, above the baseline "
                        f"interval, which ends at {before.ttfat_p95_ms.ci_high:.0f} ms"
                    ),
                )
            )
        if before.quality is not None and now.quality is not None:
            if now.quality.ci_high < before.quality.ci_low:
                regressions.append(
                    Regression(
                        mode_id=now.mode_id,
                        metric="quality",
                        baseline=before.quality.value,
                        current=now.quality.value,
                        detail=(
                            f"the interval ends at {now.quality.ci_high:.3f}, below the baseline "
                            f"interval, which starts at {before.quality.ci_low:.3f}"
                        ),
                    )
                )
    return Comparison(
        regressions=regressions,
        only_in_baseline=sorted(baseline_ids - current_ids),
        only_in_current=sorted(current_ids - baseline_ids),
        warnings=comparability_warnings(current, baseline),
    )
