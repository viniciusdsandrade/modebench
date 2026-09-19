"""The Pareto front of quality against latency.

A mode is on the front if no other mode is at least as good on the two axes
and better on one. Quality is better when it is higher. Latency is better
when it is lower.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    """One mode on the two axes."""

    key: str
    quality: float
    latency_ms: float


def dominates(first: ParetoPoint, second: ParetoPoint) -> bool:
    """Return True if `first` is at least as good as `second` on each axis and better on one."""
    at_least = first.quality >= second.quality and first.latency_ms <= second.latency_ms
    better = first.quality > second.quality or first.latency_ms < second.latency_ms
    return at_least and better


def pareto_front(points: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """Return the points that no other point dominates, fastest first."""
    front = [
        point
        for point in points
        if not any(dominates(other, point) for other in points if other is not point)
    ]
    return sorted(front, key=lambda point: (point.latency_ms, -point.quality, point.key))
