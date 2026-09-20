"""The cost estimate that comes before a run.

The estimate is an upper guide, not an invoice. Input tokens come from the
length of each prompt. Output tokens come from the assumptions of the bench
file: the size of a usual answer and the reasoning that each level uses.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from modebench.config import CostConfig, Mode, Price
from modebench.dataset.transcript import Labels
from modebench.runner.plan import PlannedRequest, render_user_message

JUDGE_ID = "judge"


@dataclass(frozen=True, slots=True)
class ModeEstimate:
    """The estimate of one mode. `cost_usd` is None if the mode has no price."""

    mode_id: str
    requests: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """The estimate of a run."""

    modes: list[ModeEstimate] = field(default_factory=list)

    @property
    def total_usd(self) -> float:
        """Return the sum of the modes that have a price."""
        return sum(item.cost_usd for item in self.modes if item.cost_usd is not None)

    @property
    def unknown_price_modes(self) -> list[str]:
        """Return the modes that have requests and no price."""
        return [item.mode_id for item in self.modes if item.cost_usd is None and item.requests]

    @property
    def requests(self) -> int:
        """Return the number of requests, the judge included."""
        return sum(item.requests for item in self.modes)

    def per_request_usd(self, mode_id: str) -> float:
        """Return the estimate of one request of a mode, or zero if there is no price."""
        for item in self.modes:
            if item.mode_id == mode_id and item.cost_usd is not None and item.requests:
                return item.cost_usd / item.requests
        return 0.0


def _tokens(chars: int, chars_per_token: float) -> int:
    return math.ceil(chars / chars_per_token)


def _cost(input_tokens: int, output_tokens: int, price: Price | None) -> float | None:
    if price is None:
        return None
    total = input_tokens * price.usd_per_mtok_in + output_tokens * price.usd_per_mtok_out
    return total / 1_000_000


def estimate_cost(
    plan: Sequence[PlannedRequest],
    modes: Mapping[str, Mode],
    prices: Mapping[str, Price | None],
    preprompt: str,
    labels: Labels,
    config: CostConfig,
    *,
    judge_prompt_chars: int | None = None,
) -> CostEstimate:
    """Return the estimate of a plan.

    With `judge_prompt_chars`, the estimate includes one judge request for
    each measured request. The price of the judge is `prices["judge"]`.
    """
    prompt_chars: dict[str, int] = {}
    inputs: dict[str, int] = {mode_id: 0 for mode_id in modes}
    outputs: dict[str, int] = {mode_id: 0 for mode_id in modes}
    counts: dict[str, int] = {mode_id: 0 for mode_id in modes}
    judge_input = 0
    judge_count = 0
    answer_chars = int(config.expected_answer_tokens * config.chars_per_token)
    for planned in plan:
        item_id = planned.item.item_id
        if item_id not in prompt_chars:
            user = render_user_message(planned, "estimate", labels)
            prompt_chars[item_id] = len(preprompt) + len(user)
        mode = modes[planned.mode_id]
        reasoning = config.expected_reasoning_tokens.get(
            mode.reasoning_effort(), config.expected_reasoning_tokens.get("default", 0)
        )
        counts[planned.mode_id] += 1
        inputs[planned.mode_id] += _tokens(prompt_chars[item_id], config.chars_per_token)
        outputs[planned.mode_id] += config.expected_answer_tokens + reasoning
        if judge_prompt_chars is not None and not planned.warmup:
            fresh_chars = sum(len(line.text) + 12 for line in planned.item.variant.fresh)
            judge_chars = judge_prompt_chars + fresh_chars + answer_chars
            judge_input += _tokens(judge_chars, config.chars_per_token)
            judge_count += 1
    estimates = [
        ModeEstimate(
            mode_id=mode_id,
            requests=counts[mode_id],
            input_tokens=inputs[mode_id],
            output_tokens=outputs[mode_id],
            cost_usd=_cost(inputs[mode_id], outputs[mode_id], prices.get(mode_id)),
        )
        for mode_id in modes
    ]
    if judge_prompt_chars is not None:
        judge_output = judge_count * config.judge_output_tokens
        estimates.append(
            ModeEstimate(
                mode_id=JUDGE_ID,
                requests=judge_count,
                input_tokens=judge_input,
                output_tokens=judge_output,
                cost_usd=_cost(judge_input, judge_output, prices.get(JUDGE_ID)),
            )
        )
    return CostEstimate(modes=estimates)
