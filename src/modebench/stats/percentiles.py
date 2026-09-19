"""Percentiles and means with bootstrap confidence intervals.

A p95 from a few hundred requests is an estimate, and two modes whose p95
differ by less than the noise are not different. Each estimate has a
percentile bootstrap interval: the data is resampled with replacement, the
statistic is computed on each resample, and the interval is the middle of
those results. The generator has a seed, so a report is reproducible.
"""

from collections.abc import Callable, Sequence
from typing import Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

# The largest resample matrix that is made in one step, in elements.
_BATCH_ELEMENTS = 2_000_000

RowStatistic = Callable[[NDArray[np.float64]], NDArray[np.float64]]


class Estimate(BaseModel):
    """A value, its confidence interval and the number of observations."""

    model_config = ConfigDict(frozen=True)

    value: float
    ci_low: float
    ci_high: float
    n: int

    def overlaps(self, other: Self) -> bool:
        """Return True if the two intervals share a point."""
        return self.ci_low <= other.ci_high and other.ci_low <= self.ci_high


def percentile(values: Sequence[float], q: float) -> float:
    """Return the percentile `q` (0 to 100) with linear interpolation."""
    if len(values) == 0:
        raise ValueError("a percentile needs one value or more")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _row_percentile(q: float) -> RowStatistic:
    def statistic(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(np.percentile(matrix, q, axis=1), dtype=np.float64)

    return statistic


def _row_mean(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(matrix.mean(axis=1), dtype=np.float64)


def _bootstrap(
    values: Sequence[float],
    statistic: RowStatistic,
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> Estimate:
    data = np.asarray(values, dtype=np.float64)
    count = int(data.size)
    if count == 0:
        raise ValueError("an estimate needs one value or more")
    point = float(statistic(data.reshape(1, count))[0])
    if count == 1:
        return Estimate(value=point, ci_low=point, ci_high=point, n=1)
    rng = np.random.default_rng(seed)
    batch = max(1, _BATCH_ELEMENTS // count)
    results: list[NDArray[np.float64]] = []
    remaining = resamples
    while remaining > 0:
        rows = min(batch, remaining)
        indices = rng.integers(0, count, size=(rows, count))
        results.append(statistic(data[indices]))
        remaining -= rows
    stacked = np.concatenate(results)
    alpha = (1.0 - confidence) / 2.0
    bounds = np.quantile(stacked, [alpha, 1.0 - alpha])
    return Estimate(value=point, ci_low=float(bounds[0]), ci_high=float(bounds[1]), n=count)


def bootstrap_percentile(
    values: Sequence[float], q: float, *, resamples: int, confidence: float, seed: int
) -> Estimate:
    """Return the percentile `q` of `values` with its bootstrap interval."""
    return _bootstrap(
        values, _row_percentile(q), resamples=resamples, confidence=confidence, seed=seed
    )


def bootstrap_mean(
    values: Sequence[float], *, resamples: int, confidence: float, seed: int
) -> Estimate:
    """Return the mean of `values` with its bootstrap interval."""
    return _bootstrap(values, _row_mean, resamples=resamples, confidence=confidence, seed=seed)
