"""A deterministic provider with no network, for dry runs and for tests.

The same mode and the same request always give the same outcome. The times
are computed, not measured, so a dry run of the full profile takes seconds.
The answer is built from the metadata of the request, which a real provider
never sees.
"""

import math
import random
from dataclasses import dataclass, fields
from typing import Any, Self

from modebench.config import Mode
from modebench.hashing import stable_seed
from modebench.providers.base import ChatRequest, StreamEvent, StreamOutcome

REFUSAL_TEXT = "Não há pergunta nova neste trecho da reunião."
WRONG_ANSWER_TEXT = (
    "Pergunta: qual é o assunto da reunião?\nResposta: o assunto ainda não está claro."
)
FABRICATED_TEXT = "Pergunta: qual é o próximo passo?\nResposta: seguir com o plano atual."
PREAMBLE_TEXT = "Claro! Com base na transcrição, "

# Reasoning time and answer quality that a dry run gives to each reasoning level.
_EFFORT_DEFAULTS: dict[str, tuple[float, float]] = {
    "none": (0.0, 0.80),
    "minimal": (250.0, 0.83),
    "low": (600.0, 0.86),
    "medium": (2500.0, 0.90),
    "high": (8000.0, 0.92),
    "default": (1200.0, 0.88),
}


@dataclass(frozen=True, slots=True)
class FakeSpec:
    """How the fake provider behaves for one mode. `params.fake` sets each field."""

    ttft_ms: float = 300.0
    reasoning_ms: float = 0.0
    tok_per_s: float = 90.0
    quality: float = 0.9
    error_rate: float = 0.0
    preamble_rate: float = 0.0
    warm_factor: float = 0.6

    @classmethod
    def for_mode(cls, mode: Mode) -> Self:
        """Return the defaults of the reasoning level with `params.fake` on top."""
        reasoning_ms, quality = _EFFORT_DEFAULTS.get(
            mode.reasoning_effort(), _EFFORT_DEFAULTS["default"]
        )
        values: dict[str, float] = {"reasoning_ms": reasoning_ms, "quality": quality}
        overrides = mode.params.get("fake")
        if isinstance(overrides, dict):
            known = {item.name for item in fields(cls)}
            for key, value in overrides.items():
                if key in known and isinstance(value, int | float) and not isinstance(value, bool):
                    values[key] = float(value)
        return cls(**values)


def _answer(spec: FakeSpec, metadata: dict[str, str], rng: random.Random) -> str:
    expect_refusal = metadata.get("expect_refusal") == "1"
    truncation = float(metadata.get("truncation_pct", "100") or "100")
    chance = spec.quality * (0.6 + 0.4 * truncation / 100.0)
    good = rng.random() < chance
    coin = rng.random()
    if expect_refusal:
        text = REFUSAL_TEXT if good else FABRICATED_TEXT
    elif good:
        question = metadata.get("gold_question") or "qual é o ponto em aberto?"
        points = [point for point in metadata.get("key_points", "").split("\n") if point]
        body = ". ".join(points) if points else "não há dados suficientes no trecho"
        text = f"Pergunta: {question}\nResposta: {body}."
    else:
        text = REFUSAL_TEXT if coin < 0.5 else WRONG_ANSWER_TEXT
    if rng.random() < spec.preamble_rate:
        text = PREAMBLE_TEXT + text
    return text


class FakeProvider:
    """A provider that computes its outcome from a seed."""

    def stream_chat(self, mode: Mode, request: ChatRequest, timeout_s: float) -> StreamOutcome:
        """Return the outcome that this mode always gives for this request."""
        spec = FakeSpec.for_mode(mode)
        metadata = dict(request.metadata)
        rng = random.Random(stable_seed(mode.id, request.user, metadata.get("repetition", "")))
        warm = metadata.get("cache_state") == "warm"
        ttft_ms = spec.ttft_ms * rng.lognormvariate(0.0, 0.18) * (spec.warm_factor if warm else 1.0)
        reasoning_ms = spec.reasoning_ms * rng.lognormvariate(0.0, 0.25)
        ttfat_ms = ttft_ms + reasoning_ms
        failed = rng.random() < spec.error_rate
        answer = _answer(spec, metadata, rng)
        timeout_ms = timeout_s * 1000.0
        if failed:
            return StreamOutcome(
                ok=False,
                total_ms=ttft_ms,
                error_kind="fake_error",
                error_message="the fake provider failed on purpose",
            )
        if ttfat_ms > timeout_ms:
            return StreamOutcome(
                ok=False,
                total_ms=timeout_ms,
                error_kind="timeout",
                error_message=f"no complete answer in {timeout_s:g} s",
            )
        prompt_tokens = math.ceil((len(request.system) + len(request.user)) / 4)
        answer_tokens = max(1, math.ceil(len(answer.split()) * 1.4))
        reasoning_tokens = int(reasoning_ms / 1000.0 * spec.tok_per_s)
        total_ms = ttfat_ms + answer_tokens / spec.tok_per_s * 1000.0
        events = [StreamEvent(offset_ms=ttfat_ms, kind="content", chars=len(answer))]
        if reasoning_ms > 0:
            events.insert(0, StreamEvent(offset_ms=ttft_ms, kind="reasoning", chars=0))
        return StreamOutcome(
            ok=True,
            total_ms=total_ms,
            first_chunk_ms=ttft_ms,
            ttft_ms=ttft_ms,
            first_reasoning_ms=ttft_ms if reasoning_ms > 0 else None,
            ttfat_ms=ttfat_ms,
            answer=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=answer_tokens + reasoning_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_tokens=int(prompt_tokens * 0.8) if warm else 0,
            total_tokens=prompt_tokens + answer_tokens + reasoning_tokens,
            served_by="fake",
            events=events,
        )


def fake_params(**values: Any) -> dict[str, Any]:
    """Return mode parameters that set the behaviour of the fake provider."""
    return {"fake": dict(values)}
