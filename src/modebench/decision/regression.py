"""Compares a run with a baseline and finds regressions.

Two facts make a regression. The p95 of the time to the first answer token
went up by more than the permitted ratio (20 percent by default). Or the
quality is below the confidence interval of the baseline quality. A mode that
is only in one of the two summaries is reported, and it is not a regression.
"""

from dataclasses import dataclass

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

    @property
    def failed(self) -> bool:
        """Return True if the comparison found a regression."""
        return bool(self.regressions)


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
        if now.ttfat_p95_ms.value > limit:
            regressions.append(
                Regression(
                    mode_id=now.mode_id,
                    metric="ttfat_p95_ms",
                    baseline=before.ttfat_p95_ms.value,
                    current=now.ttfat_p95_ms.value,
                    detail=f"above the limit of {limit:.0f} ms (+{p95_increase_ratio:.0%})",
                )
            )
        if before.quality is not None and now.quality is not None:
            if now.quality.value < before.quality.ci_low:
                regressions.append(
                    Regression(
                        mode_id=now.mode_id,
                        metric="quality",
                        baseline=before.quality.value,
                        current=now.quality.value,
                        detail=f"below the baseline interval, which starts at {before.quality.ci_low:.3f}",
                    )
                )
    return Comparison(
        regressions=regressions,
        only_in_baseline=sorted(baseline_ids - current_ids),
        only_in_current=sorted(current_ids - baseline_ids),
    )
