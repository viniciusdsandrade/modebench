"""Word error rate and character error rate, on normalised Portuguese text."""

from dataclasses import dataclass

import jiwer

from modebench.stt.config import SttNormalization
from modebench.stt.normalize import normalize_pt


@dataclass(frozen=True, slots=True)
class ErrorRates:
    """The error rates of one hypothesis, with the counts that make the word rate."""

    wer: float
    cer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    reference_words: int


def error_rates(
    reference: str, hypothesis: str, options: SttNormalization | None = None
) -> ErrorRates | None:
    """Return the error rates, or None if the reference has no word after normalisation.

    An empty hypothesis is a hypothesis in which each word was deleted. That
    case is computed here, because the library does not accept it.
    """
    expected = normalize_pt(reference, options)
    actual = normalize_pt(hypothesis, options)
    words = len(expected.split())
    if words == 0:
        return None
    if not actual:
        return ErrorRates(
            wer=1.0,
            cer=1.0,
            substitutions=0,
            deletions=words,
            insertions=0,
            hits=0,
            reference_words=words,
        )
    measured = jiwer.process_words(expected, actual)
    return ErrorRates(
        wer=float(measured.wer),
        cer=float(jiwer.cer(expected, actual)),
        substitutions=int(measured.substitutions),
        deletions=int(measured.deletions),
        insertions=int(measured.insertions),
        hits=int(measured.hits),
        reference_words=words,
    )
