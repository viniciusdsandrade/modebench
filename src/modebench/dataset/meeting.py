"""The meeting scenario: clicks in sequence on a transcript that grows.

A person clicks Analyze more than one time in a meeting. Each request starts
with the text of the request before it, so a provider with a prompt cache can
answer the later clicks faster. The scenario keeps the first click (cold
cache) apart from the others (warm cache).
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from modebench.dataset.schema import FillerFile
from modebench.dataset.transcript import build_filler
from modebench.dataset.variants import RenderLine, Variant


@dataclass(frozen=True, slots=True)
class MeetingClick:
    """One click: where it is in the meeting and what the transcript held before it."""

    scenario_id: str
    click_index: int
    minute: int
    variant: Variant
    prior: tuple[RenderLine, ...]

    @property
    def cache_state(self) -> str:
        """Return `cold` for the first click of a scenario and `warm` for the others."""
        return "cold" if self.click_index == 0 else "warm"


def scenario_pool(variants: Sequence[Variant]) -> list[Variant]:
    """Return the variants that can be a click: complete, clean, synthetic cases."""
    return [variant for variant in variants if variant.is_baseline and not variant.earlier]


def build_scenarios(
    variants: Sequence[Variant],
    filler: FillerFile,
    click_minutes: Sequence[int],
    scenarios: int,
    words_per_minute: int,
    seed: int,
) -> list[list[MeetingClick]]:
    """Return `scenarios` meetings, each with one click at each of `click_minutes`.

    The cases of a scenario come from the pool in order. After a click, the
    lines of its case are settled and become part of the earlier stretch of
    the next click, as they do in the application.
    """
    pool = scenario_pool(variants)
    if not pool or scenarios <= 0:
        return []
    result: list[list[MeetingClick]] = []
    for scenario_index in range(scenarios):
        scenario_id = f"meeting-{scenario_index + 1}"
        timeline: list[RenderLine] = []
        filler_used = 0
        clicks: list[MeetingClick] = []
        for click_index, minute in enumerate(click_minutes):
            filler_lines = build_filler(filler, minute, words_per_minute, seed)
            timeline.extend(filler_lines[filler_used:])
            filler_used = max(filler_used, len(filler_lines))
            variant = pool[(scenario_index * len(click_minutes) + click_index) % len(pool)]
            clicks.append(
                MeetingClick(
                    scenario_id=scenario_id,
                    click_index=click_index,
                    minute=minute,
                    variant=variant,
                    prior=tuple(timeline),
                )
            )
            timeline.extend(replace(line, partial=False) for line in variant.fresh)
        result.append(clicks)
    return result
