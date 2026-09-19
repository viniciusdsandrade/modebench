"""The plan of a run: which request goes to which mode, and in which order.

The modes take turns on each item (interleaved round-robin), and the mode
that goes first changes from item to item. A slow minute of the network then
falls on each mode equally, and not on the mode that a plain loop would have
measured in that minute.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from modebench.config import MeetingConfig, Profile, TranscriptConfig
from modebench.dataset.meeting import build_scenarios
from modebench.dataset.schema import FillerFile
from modebench.dataset.transcript import Labels, build_filler, nonce_line, render_window
from modebench.dataset.variants import RenderLine, Variant
from modebench.errors import ConfigError
from modebench.hashing import short_token
from modebench.providers.base import ChatRequest


@dataclass(frozen=True, slots=True)
class WorkItem:
    """One variant in one accumulated transcript. `prior` has no nonce line yet."""

    item_id: str
    suite: str
    variant: Variant
    duration_min: int
    prior: tuple[RenderLine, ...]
    scenario_id: str | None = None
    click_index: int | None = None
    cache_state: str = "cold"


@dataclass(frozen=True, slots=True)
class PlannedRequest:
    """One request of the plan."""

    seq: int
    mode_id: str
    item: WorkItem
    repetition: int
    warmup: bool

    @property
    def nonce_key(self) -> str:
        """Return what the nonce of the transcript depends on.

        The clicks of one meeting share a nonce, so their requests share a
        prefix and the cache of the provider can be warm. All other requests
        get a nonce of their own, so their cache is cold.
        """
        scope = self.item.scenario_id if self.item.scenario_id is not None else self.item.item_id
        phase = "warmup" if self.warmup else "measured"
        return f"{scope}|{self.repetition}|{self.mode_id}|{phase}"


def _durations(variant: Variant, profile: Profile) -> list[int]:
    """Return the transcript lengths of a variant.

    In the star design, the complete clean form of a case and the noise cases
    go in each length, and each other variant goes in the baseline length
    only. In the cross design, each variant goes in each length.
    """
    if profile.design == "cross" or variant.is_baseline or variant.kind == "noise":
        return list(profile.durations_min)
    return [profile.baseline_duration_min]


def build_items(
    variants: Sequence[Variant],
    profile: Profile,
    filler: FillerFile,
    transcript: TranscriptConfig,
    meeting: MeetingConfig,
    seed: int,
) -> list[WorkItem]:
    """Return the items of the single suite, then those of the meeting scenario."""
    fillers: dict[int, tuple[RenderLine, ...]] = {}
    items: list[WorkItem] = []
    for variant in variants:
        for duration in _durations(variant, profile):
            if duration not in fillers:
                fillers[duration] = build_filler(
                    filler, duration, transcript.words_per_minute, seed
                )
            items.append(
                WorkItem(
                    item_id=f"{variant.variant_id}|{duration}m",
                    suite="single",
                    variant=variant,
                    duration_min=duration,
                    prior=fillers[duration] + variant.earlier,
                )
            )
    scenarios = build_scenarios(
        variants,
        filler,
        meeting.click_minutes,
        profile.meeting_scenarios,
        transcript.words_per_minute,
        seed,
    )
    for clicks in scenarios:
        for click in clicks:
            items.append(
                WorkItem(
                    item_id=f"{click.scenario_id}|click{click.click_index}",
                    suite="meeting",
                    variant=click.variant,
                    duration_min=click.minute,
                    prior=click.prior,
                    scenario_id=click.scenario_id,
                    click_index=click.click_index,
                    cache_state=click.cache_state,
                )
            )
    return items


def build_plan(
    items: Sequence[WorkItem], mode_ids: Sequence[str], repetitions: int, warmup: int
) -> list[PlannedRequest]:
    """Return the requests in the order in which they go out.

    The warm-up requests come first, one round for each mode. They open the
    connections and they are not part of any statistic.
    """
    if not items:
        raise ConfigError("the profile and the datasets give no item to run")
    if not mode_ids:
        raise ConfigError("no mode is selected")
    modes = list(mode_ids)
    singles = [item for item in items if item.suite == "single"]
    warmup_pool = singles if singles else list(items)
    plan: list[PlannedRequest] = []
    for round_index in range(warmup):
        item = warmup_pool[round_index % len(warmup_pool)]
        for mode_id in modes:
            plan.append(PlannedRequest(len(plan), mode_id, item, repetition=0, warmup=True))
    for repetition in range(1, repetitions + 1):
        for index, item in enumerate(items):
            offset = (index + repetition) % len(modes)
            for mode_id in modes[offset:] + modes[:offset]:
                plan.append(PlannedRequest(len(plan), mode_id, item, repetition, warmup=False))
    return plan


def render_user_message(planned: PlannedRequest, run_id: str, labels: Labels) -> str:
    """Return the user message of a planned request, with its nonce line first."""
    nonce = short_token(run_id, planned.nonce_key)
    prior = (nonce_line(nonce), *planned.item.prior)
    return render_window(prior, planned.item.variant.fresh, labels)


def render_request(
    planned: PlannedRequest, run_id: str, preprompt: str, labels: Labels
) -> ChatRequest:
    """Return the chat request of a planned request.

    The metadata is for the fake provider and for the records. It holds the
    expected answer, and a network provider never sends it.
    """
    variant = planned.item.variant
    metadata = {
        "case_id": variant.case_id,
        "variant_id": variant.variant_id,
        "expect_refusal": "1" if variant.expect_refusal else "0",
        "gold_question": variant.gold_question,
        "key_points": "\n".join(variant.key_points),
        "truncation_pct": str(variant.truncation_pct),
        "cache_state": planned.item.cache_state,
        "repetition": str(planned.repetition),
    }
    user = render_user_message(planned, run_id, labels)
    return ChatRequest(system=preprompt, user=user, metadata=metadata)
