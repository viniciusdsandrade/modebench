"""The judge: one fixed model that grades each answer, blind to the mode.

The judge gets the new stretch of the transcript, the expected answer and the
answer to grade. It does not get the mode, the provider or a time, and the
answers come to it in a seeded random order. Its verdict is JSON, and the
JSON is validated. A verdict that is not valid is asked for again, and then
recorded as a judge failure: it is never replaced by a guess.
"""

import json
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from modebench.config import Mode, Price, ProviderConfig
from modebench.hashing import sha256_text
from modebench.providers.base import ChatProvider, ChatRequest
from modebench.providers.metrics import derive_metrics
from modebench.quality.deterministic import is_refusal
from modebench.storage.records import ScoreRecord

SCORER = "judge"


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
    """A verdict, or the reason for which there is none."""

    verdict: JudgeVerdict | None
    attempts: int
    error: str | None = None
    cost_usd: float = 0.0


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


def render_judge_message(item: JudgeItem) -> str:
    """Return the user message of a judge request."""
    points = "\n".join(f"- {point}" for point in item.key_points) or "(none)"
    return (
        "<new_stretch>\n"
        f"{item.fresh_text}\n"
        "</new_stretch>\n\n"
        "<expected>\n"
        f"noise_input: {'true' if item.expect_refusal else 'false'}\n"
        f"gold_question: {item.gold_question or '(none)'}\n"
        f"key_points:\n{points}\n"
        f"reference_answer: {item.reference_answer or '(none)'}\n"
        "</expected>\n\n"
        "<answer>\n"
        f"{item.answer}\n"
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
    ) -> None:
        self._provider = provider
        self._provider_config = provider_config
        self._mode = mode
        self._prompt = prompt
        self._price = price
        self._max_attempts = max_attempts
        self._timeout_s = timeout_s
        self._cache: dict[str, JudgeVerdict] = {}

    def judge(self, item: JudgeItem) -> JudgeResult:
        """Ask the model until a verdict is valid or the attempts are used."""
        message = render_judge_message(item)
        key = sha256_text(message)
        if key in self._cache:
            return JudgeResult(verdict=self._cache[key], attempts=0)
        cost = 0.0
        error = "no attempt was made"
        for attempt in range(1, self._max_attempts + 1):
            request = ChatRequest(system=self._prompt, user=message)
            outcome = self._provider.stream_chat(self._mode, request, self._timeout_s)
            metrics = derive_metrics(outcome, self._provider_config, self._price)
            cost += metrics.cost_usd or 0.0
            if not outcome.ok:
                error = f"{outcome.error_kind}: {outcome.error_message}"
                continue
            try:
                verdict = parse_verdict(outcome.answer, len(item.key_points))
            except ValueError as exc:
                error = str(exc)
                continue
            self._cache[key] = verdict
            return JudgeResult(verdict=verdict, attempts=attempt, cost_usd=cost)
        return JudgeResult(verdict=None, attempts=self._max_attempts, error=error, cost_usd=cost)


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
