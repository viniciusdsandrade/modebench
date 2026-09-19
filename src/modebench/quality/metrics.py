"""The quality of one answer, as one number from 0 to 1.

Noise has one correct answer, a refusal: 1 for a refusal, 0 for anything
else. Speech must be answered: a refusal or a failed request is 0, and an
answer gets the weighted sum of three parts that the judge grades (the
inferred question, the utility and the key points).
"""

from dataclasses import dataclass

from modebench.config import QualityWeights
from modebench.quality.deterministic import DeterministicScores
from modebench.quality.judge import JudgeVerdict
from modebench.storage.records import ScoreRecord

SCORER = "derived"


@dataclass(frozen=True, slots=True)
class RequestQuality:
    """The final reading of one request.

    `quality` and `question_correct` are None when the answer needed a judge
    and the judge gave no valid verdict.
    """

    refused: bool
    quality: float | None
    question_correct: float | None
    false_refusal: bool
    false_acceptance: bool


def request_quality(
    *,
    ok: bool,
    expect_refusal: bool,
    key_points: int,
    deterministic: DeterministicScores,
    verdict: JudgeVerdict | None,
    weights: QualityWeights,
) -> RequestQuality:
    """Return the quality of one request.

    The judge decides if the answer is a refusal when it gave a verdict. The
    fixed rules decide when it did not.
    """
    if not ok or not deterministic.non_empty:
        return RequestQuality(
            refused=False,
            quality=0.0,
            question_correct=None if expect_refusal else 0.0,
            false_refusal=False,
            false_acceptance=False,
        )
    refused = verdict.refused if verdict is not None else deterministic.is_refusal
    if expect_refusal:
        return RequestQuality(
            refused=refused,
            quality=1.0 if refused else 0.0,
            question_correct=None,
            false_refusal=False,
            false_acceptance=not refused,
        )
    if refused:
        return RequestQuality(
            refused=True,
            quality=0.0,
            question_correct=0.0,
            false_refusal=True,
            false_acceptance=False,
        )
    if verdict is None:
        return RequestQuality(
            refused=False,
            quality=None,
            question_correct=None,
            false_refusal=False,
            false_acceptance=False,
        )
    parts = [
        (weights.question, float(verdict.inferred_question_correct)),
        (weights.utility, (verdict.utility - 1) / 4),
    ]
    if key_points:
        parts.append((weights.key_points, verdict.key_points_covered / key_points))
    weight_sum = sum(weight for weight, _ in parts)
    quality = sum(weight * value for weight, value in parts) / weight_sum if weight_sum else 0.0
    return RequestQuality(
        refused=False,
        quality=quality,
        question_correct=float(verdict.inferred_question_correct),
        false_refusal=False,
        false_acceptance=False,
    )


def quality_records(result: RequestQuality) -> list[ScoreRecord]:
    """Return the final reading as database rows."""
    return [
        ScoreRecord(SCORER, "refused", float(result.refused)),
        ScoreRecord(SCORER, "quality", result.quality),
        ScoreRecord(SCORER, "question_correct", result.question_correct),
        ScoreRecord(SCORER, "false_refusal", float(result.false_refusal)),
        ScoreRecord(SCORER, "false_acceptance", float(result.false_acceptance)),
    ]
