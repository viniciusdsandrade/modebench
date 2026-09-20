"""Speaker attribution: does the provider give each utterance to the correct speaker?

A provider names its speakers as it likes ("A", "speaker_0"), so its labels
are first matched to the speakers of the reference. The match that makes the
most words correct is used, and the score is the share of words that are then
with the correct speaker. A provider that gives no labels has no score.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import permutations

from modebench.stt.config import ReferenceUtterance

MAX_EXACT_LABELS = 6


@dataclass(frozen=True, slots=True)
class FinalUtterance:
    """One settled utterance of a hypothesis."""

    index: int
    text: str
    speaker: str | None
    received_ms: float
    audio_start_ms: float | None
    audio_end_ms: float | None
    latency_ms: float | None


def _overlap(final: FinalUtterance, reference: ReferenceUtterance) -> float:
    if final.audio_end_ms is None:
        return 0.0
    start = final.audio_start_ms if final.audio_start_ms is not None else final.audio_end_ms
    return max(0.0, min(final.audio_end_ms, reference.end_ms) - max(start, reference.start_ms))


def match_reference(
    final: FinalUtterance, position: int, references: Sequence[ReferenceUtterance]
) -> ReferenceUtterance:
    """Return the reference utterance of a final: most overlap in time, or the same position."""
    best = max(references, key=lambda reference: _overlap(final, reference))
    if _overlap(final, best) > 0:
        return best
    return references[min(position, len(references) - 1)]


def _best_mapping_score(table: dict[str, dict[str, int]], speakers: list[str]) -> int:
    labels = list(table)
    if len(labels) <= MAX_EXACT_LABELS and len(speakers) <= MAX_EXACT_LABELS:
        slots: list[str | None] = [*speakers, *([None] * max(0, len(labels) - len(speakers)))]
        best = 0
        for assignment in permutations(slots, len(labels)):
            score = 0
            for label, speaker in zip(labels, assignment, strict=True):
                if speaker is not None:
                    score += table[label].get(speaker, 0)
            best = max(best, score)
        return best
    taken: set[str] = set()
    total = 0
    cells = sorted(
        ((count, label, speaker) for label, row in table.items() for speaker, count in row.items()),
        reverse=True,
    )
    used_labels: set[str] = set()
    for count, label, speaker in cells:
        if label in used_labels or speaker in taken:
            continue
        used_labels.add(label)
        taken.add(speaker)
        total += count
    return total


def speaker_accuracy(
    finals: Sequence[FinalUtterance], references: Sequence[ReferenceUtterance]
) -> float | None:
    """Return the share of words with the correct speaker, or None if it cannot be scored."""
    speakers = sorted({reference.speaker for reference in references if reference.speaker})
    labelled = [(position, final) for position, final in enumerate(finals) if final.speaker]
    if not speakers or not labelled:
        return None
    table: dict[str, dict[str, int]] = {}
    total = 0
    for position, final in labelled:
        reference = match_reference(final, position, references)
        weight = max(1, len(final.text.split()))
        total += weight
        if reference.speaker and final.speaker is not None:
            row = table.setdefault(final.speaker, {})
            row[reference.speaker] = row.get(reference.speaker, 0) + weight
    if total == 0 or not table:
        return None
    return _best_mapping_score(table, speakers) / total
