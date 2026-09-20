"""The judge: one fixed model that grades each answer, blind to the mode.

The judge gets the new stretch of the transcript, the expected answer and the
answer to grade. It does not get the mode, the provider or a time, and the
answers come to it in a seeded random order. Its verdict is JSON, and the
JSON is validated. A verdict that is not valid is asked for again, and then
recorded as a judge failure: it is never replaced by a guess.
"""

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from modebench.config import Mode, Price, ProviderConfig
from modebench.hashing import sha256_text
from modebench.providers.base import ChatProvider, ChatRequest
from modebench.providers.metrics import assumed_cost, derive_metrics
from modebench.quality.deterministic import is_refusal
from modebench.storage.records import ScoreRecord

SCORER = "judge"

# The tags that divide the message of the judge into blocks.
_BLOCK_TAG = re.compile(r"<(/?)(new_stretch|expected|answer)\b", re.IGNORECASE)


class JudgeVerdict(BaseModel):
    """The JSON object that the judge must return."""

    model_config = ConfigDict(extra="ignore")

    refused: bool
    inferred_question_correct: bool
    key_points_covered: int = Field(ge=0)
    utility: int = Field(ge=1, le=5)
    hallucination: bool = False
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class JudgeItem:
    """What the judge sees. There is no field for the mode, and that is the point."""

    fresh_text: str
    expect_refusal: bool
    gold_question: str
    key_points: tuple[str, ...]
    reference_answer: str
    answer: str


@dataclass(frozen=True, slots=True)
class JudgeResult:
    """A verdict, or the reason for which there is none.

    `cost_usd` is the measured cost of the attempts. `assumed_cost_usd` is
    what the cost ceiling assumes for the attempts that have no usage block.
    """

    verdict: JudgeVerdict | None
    attempts: int
    error: str | None = None
    cost_usd: float = 0.0
    assumed_cost_usd: float = 0.0


class Judge(Protocol):
    """Grades one answer."""

    def judge(self, item: JudgeItem) -> JudgeResult:
        """Return the verdict for one item."""
        ...


def parse_verdict(text: str, key_points: int) -> JudgeVerdict:
    """Return the verdict in `text`, or raise ValueError.

    The object can stand in a code fence or after a sentence, so the parser
    takes the text from the first brace to the last one.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("the judge returned no JSON object")
    try:
        verdict = JudgeVerdict.model_validate(json.loads(text[start : end + 1]))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"the judge returned an invalid verdict: {exc}") from exc
    if verdict.key_points_covered > key_points:
        verdict = verdict.model_copy(update={"key_points_covered": key_points})
    return verdict


def inert(text: str) -> str:
    """Return `text` with each block tag of the judge message made inert.

    The transcript and the answer are data that the benchmark does not
    control. A text that holds `</answer>` could end its block early and give
    the judge instructions of its own, so the `<` of such a tag becomes `&lt;`.
    """
    return _BLOCK_TAG.sub(lambda match: f"&lt;{match.group(1)}{match.group(2)}", text)


def render_judge_message(item: JudgeItem) -> str:
    """Return the user message of a judge request. Only this function writes block tags."""
    points = "\n".join(f"- {inert(point)}" for point in item.key_points) or "(none)"
    return (
        "<new_stretch>\n"
        f"{inert(item.fresh_text)}\n"
        "</new_stretch>\n\n"
        "<expected>\n"
        f"noise_input: {'true' if item.expect_refusal else 'false'}\n"
        f"gold_question: {inert(item.gold_question) or '(none)'}\n"
        f"key_points:\n{points}\n"
        f"reference_answer: {inert(item.reference_answer) or '(none)'}\n"
        "</expected>\n\n"
        "<answer>\n"
        f"{inert(item.answer)}\n"
        "</answer>"
    )


class LlmJudge:
    """The judge that asks a model. Equal items are judged one time."""

    def __init__(
        self,
        provider: ChatProvider,
        provider_config: ProviderConfig,
        mode: Mode,
        prompt: str,
        *,
        price: Price | None,
        max_attempts: int,
        timeout_s: float,
        assumed_cost_usd: float = 0.0,
        pause_after_error_s: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._provider = provider
        self._provider_config = provider_config
        self._mode = mode
        self._prompt = prompt
        self._price = price
        self._max_attempts = max_attempts
        self._timeout_s = timeout_s
        self._assumed_cost_usd = assumed_cost_usd
        self._pause_after_error_s = pause_after_error_s
        self._sleep = sleep
        self._cache: dict[str, JudgeVerdict] = {}

    def close(self) -> None:
        """Close the HTTP client of the provider, if the provider has one."""
        closer = getattr(self._provider, "close", None)
        if callable(closer):
            closer()

    def judge(self, item: JudgeItem) -> JudgeResult:
        """Ask the model until a verdict is valid or the attempts are used.

        A request that fails is followed by a pause that grows with each
        attempt, so that a rate limit of the endpoint can clear.
        """
        message = render_judge_message(item)
        key = sha256_text(message)
        if key in self._cache:
            return JudgeResult(verdict=self._cache[key], attempts=0)
        cost = 0.0
        assumed = 0.0
        error = "no attempt was made"
        for attempt in range(1, self._max_attempts + 1):
            request = ChatRequest(system=self._prompt, user=message)
            outcome = self._provider.stream_chat(self._mode, request, self._timeout_s)
            metrics = derive_metrics(outcome, self._provider_config, self._price)
            cost += metrics.cost_usd or 0.0
            assumed += assumed_cost(outcome, metrics, self._assumed_cost_usd)
            if not outcome.ok:
                error = f"{outcome.error_kind}: {outcome.error_message}"
                if attempt < self._max_attempts and self._pause_after_error_s > 0:
                    self._sleep(self._pause_after_error_s * attempt)
                continue
            try:
                verdict = parse_verdict(outcome.answer, len(item.key_points))
            except ValueError as exc:
                error = str(exc)
                continue
            self._cache[key] = verdict
            return JudgeResult(
                verdict=verdict, attempts=attempt, cost_usd=cost, assumed_cost_usd=assumed
            )
        return JudgeResult(
            verdict=None,
            attempts=self._max_attempts,
            error=error,
            cost_usd=cost,
            assumed_cost_usd=assumed,
        )


class FakeJudge:
    """A judge with no network, for dry runs and tests. It compares text."""

    def judge(self, item: JudgeItem) -> JudgeResult:
        """Return a verdict from the words that the answer shares with the expected one."""
        answer = item.answer.lower()
        refused = is_refusal(item.answer)
        covered = sum(1 for point in item.key_points if point.lower() in answer)
        correct = bool(item.gold_question) and item.gold_question.lower() in answer
        if item.expect_refusal:
            utility = 5 if refused else 1
        elif refused or not correct:
            utility = 1
        else:
            share = covered / len(item.key_points) if item.key_points else 1.0
            utility = 2 + round(3 * share)
        verdict = JudgeVerdict(
            refused=refused,
            inferred_question_correct=correct and not refused,
            key_points_covered=covered,
            utility=utility,
            rationale="fake judge: text comparison",
        )
        return JudgeResult(verdict=verdict, attempts=1)


def verdict_records(verdict: JudgeVerdict, key_points: int) -> list[ScoreRecord]:
    """Return a verdict as database rows."""
    coverage = verdict.key_points_covered / key_points if key_points else None
    return [
        ScoreRecord(SCORER, "refused", float(verdict.refused)),
        ScoreRecord(SCORER, "inferred_question_correct", float(verdict.inferred_question_correct)),
        ScoreRecord(SCORER, "key_points_covered", float(verdict.key_points_covered)),
        ScoreRecord(SCORER, "key_point_coverage", coverage),
        ScoreRecord(SCORER, "utility", float(verdict.utility), verdict.rationale[:500]),
        ScoreRecord(SCORER, "hallucination", float(verdict.hallucination)),
    ]
