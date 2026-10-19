"""The measures of this package: WDER, the name accuracy and the echo rate.

- WDER (Shafey et al., Interspeech 2019, https://doi.org/10.21437/Interspeech.2019-1943)
  is (S_IS + C_IS) / (S + C): of the reference words that the hypothesis has,
  as a correct word or as a substitution, the share that is with the wrong
  speaker. Insertions and deletions are not in the formula, so the WER of the
  same alignment is given beside it. The words are compared after the
  Portuguese normalisation of this repository. An echo line is not read: the
  same words are on the line of the call.
- The name accuracy uses the match of speakers by shared time, and only the
  names that the application found itself, by its rules or its model: a name
  from the configuration or typed by a person measures nothing. Recall is the
  share of the named reference speakers whose match has a correct found name.
  Precision is the share of the found names that are correct, of those whose
  reference speaker has a name in the gold set. Names are compared without
  accents and letter case.
- The operator confusion is the share of the microphone speech that went to
  the wrong side of the conversation: the operator under a person of the call,
  or a person of the call (an echo) under the operator. The side of a
  reference speaker is the start of its label, `mic-` or `sys-`. Speech of
  two sides at the same time counts against it.
- The echo rate is the share of the microphone words that a line of the call
  has at the same time. The marked rate is the share of the microphone words
  in the lines that the application marked as an echo.
"""

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jiwer

from modebench.speakers.gold import Turn, WordSegment
from modebench.speakers.hypothesis import ECHO, Hypothesis, HypothesisLine
from modebench.speakers.mapping import best_mapping, mapped_total
from modebench.stt.normalize import normalize_pt

ECHO_TOLERANCE_S = 1.0
MIC_PREFIX = "mic-"
SYSTEM_PREFIX = "sys-"


@dataclass(frozen=True, slots=True)
class WordDiarization:
    """The word diarization error rate, with the counts of its alignment."""

    wder: float | None
    wer: float
    correct: int
    substitutions: int
    deletions: int
    insertions: int
    wrong_speaker: int
    reference_words: int
    mapping: dict[str, str]


@dataclass(frozen=True, slots=True)
class NameAccuracy:
    """How many found names are correct, in the two directions."""

    recall: float | None
    precision: float | None
    named_reference: int
    named_hypothesis: int


@dataclass(frozen=True, slots=True)
class NameCheck:
    """One name of the application, against the gold set."""

    speaker: str
    name: str
    origin: str | None
    confidence_pct: int | None
    reference: str | None
    correct: bool | None


@dataclass(frozen=True, slots=True)
class OperatorConfusion:
    """How much microphone speech went to the wrong side of the conversation."""

    operator_as_remote_s: float
    remote_as_operator_s: float
    microphone_s: float
    rate: float | None


@dataclass(frozen=True, slots=True)
class EchoRate:
    """How much of the microphone is the loudspeaker, and how much of it was marked."""

    rate: float | None
    marked_rate: float | None
    microphone_words: int


def _tokens(text: str) -> list[str]:
    return normalize_pt(text).split()


def fold_name(name: str) -> str:
    """Return a name without accents, letter case and extra spaces."""
    decomposed = unicodedata.normalize("NFD", name)
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(plain.casefold().split())


def _int(value: Any) -> int:
    return int(value)


def word_diarization(
    reference: Sequence[WordSegment], hypothesis: Sequence[HypothesisLine]
) -> WordDiarization | None:
    """Return the WDER and the WER of the hypothesis words, or None without reference words."""
    reference_words: list[str] = []
    reference_speakers: list[str] = []
    for segment in sorted(reference, key=lambda item: (item.start, item.end)):
        tokens = _tokens(segment.words)
        reference_words.extend(tokens)
        reference_speakers.extend([segment.speaker] * len(tokens))
    if not reference_words:
        return None

    hypothesis_words: list[str] = []
    hypothesis_speakers: list[str] = []
    for line in hypothesis:
        if line.origin == ECHO:
            continue
        tokens = _tokens(line.text)
        hypothesis_words.extend(tokens)
        hypothesis_speakers.extend([line.speaker] * len(tokens))

    if not hypothesis_words:
        return WordDiarization(
            wder=None,
            wer=1.0,
            correct=0,
            substitutions=0,
            deletions=len(reference_words),
            insertions=0,
            wrong_speaker=0,
            reference_words=len(reference_words),
            mapping={},
        )

    measured = jiwer.process_words(" ".join(reference_words), " ".join(hypothesis_words))
    pairs: list[tuple[int, int]] = []
    correct = substitutions = deletions = insertions = 0
    for chunk in measured.alignments[0]:
        kind = str(chunk.type)
        ref_start, ref_end = _int(chunk.ref_start_idx), _int(chunk.ref_end_idx)
        hyp_start, hyp_end = _int(chunk.hyp_start_idx), _int(chunk.hyp_end_idx)
        if kind == "delete":
            deletions += ref_end - ref_start
            continue
        if kind == "insert":
            insertions += hyp_end - hyp_start
            continue
        paired = min(ref_end - ref_start, hyp_end - hyp_start)
        pairs.extend((ref_start + offset, hyp_start + offset) for offset in range(paired))
        if kind == "equal":
            correct += paired
        else:
            substitutions += paired
        deletions += ref_end - ref_start - paired
        insertions += hyp_end - hyp_start - paired

    table: dict[str, dict[str, float]] = {}
    for ref_index, hyp_index in pairs:
        row = table.setdefault(hypothesis_speakers[hyp_index], {})
        speaker = reference_speakers[ref_index]
        row[speaker] = row.get(speaker, 0.0) + 1.0
    mapping = best_mapping(table)
    kept = int(round(mapped_total(table, mapping)))
    aligned = len(pairs)
    wrong = aligned - kept
    return WordDiarization(
        wder=wrong / aligned if aligned else None,
        wer=(substitutions + deletions + insertions) / len(reference_words),
        correct=correct,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        wrong_speaker=wrong,
        reference_words=len(reference_words),
        mapping=mapping,
    )


def overlap_table(
    reference: Sequence[Turn], hypothesis: Sequence[HypothesisLine]
) -> dict[str, dict[str, float]]:
    """Return the seconds that each hypothesis speaker shares with each reference speaker."""
    table: dict[str, dict[str, float]] = {}
    for line in hypothesis:
        for turn in reference:
            shared = min(line.end, turn.end) - max(line.start, turn.start)
            if shared > 0:
                row = table.setdefault(line.speaker, {})
                row[turn.speaker] = row.get(turn.speaker, 0.0) + shared
    return table


def time_mapping(reference: Sequence[Turn], hypothesis: Sequence[HypothesisLine]) -> dict[str, str]:
    """Return the match of hypothesis speakers to reference speakers by shared time."""
    return best_mapping(overlap_table(reference, hypothesis))


def reference_side(label: str) -> str | None:
    """Return the channel a reference label tells, `mic` or `system`, or None."""
    if label.startswith(MIC_PREFIX):
        return "mic"
    if label.startswith(SYSTEM_PREFIX):
        return "system"
    return None


def name_checks(
    gold_names: Mapping[str, Sequence[str]],
    hypothesis: Hypothesis,
    mapping: Mapping[str, str],
) -> list[NameCheck]:
    """Return each name of the application with its reference speaker and its verdict.

    The verdict is None when the matched reference speaker has no name in the
    gold set, or when the speaker has no match: nothing says the name is wrong.
    """
    correct_forms = {
        speaker: {fold_name(form) for form in forms} for speaker, forms in gold_names.items()
    }
    checks: list[NameCheck] = []
    for speaker, given in sorted(hypothesis.names.items()):
        reference = mapping.get(speaker)
        forms = correct_forms.get(reference) if reference is not None else None
        checks.append(
            NameCheck(
                speaker=speaker,
                name=given.name,
                origin=given.origin,
                confidence_pct=given.confidence_pct,
                reference=reference,
                correct=None if forms is None else fold_name(given.name) in forms,
            )
        )
    return checks


def name_accuracy(
    gold_names: Mapping[str, Sequence[str]],
    hypothesis: Hypothesis,
    mapping: Mapping[str, str],
) -> NameAccuracy:
    """Return the share of correct found names, from the reference side and the hypothesis side.

    A reference speaker whose match carries a name from the configuration or
    from a person is left out of the recall: the application did not have to
    find that name.
    """
    correct_forms = {
        speaker: {fold_name(form) for form in forms} for speaker, forms in gold_names.items()
    }
    found_names = {
        speaker: fold_name(given.name)
        for speaker, given in hypothesis.names.items()
        if given.found()
    }
    given_elsewhere = {speaker for speaker, given in hypothesis.names.items() if not given.found()}
    match_of = {reference: speaker for speaker, reference in mapping.items()}

    to_find = [
        reference for reference in correct_forms if match_of.get(reference) not in given_elsewhere
    ]
    found = sum(
        1
        for reference in to_find
        if found_names.get(match_of.get(reference, ""), None) in correct_forms[reference]
    )

    judged = [
        (speaker, name)
        for speaker, name in found_names.items()
        if mapping.get(speaker) in correct_forms
    ]
    right = sum(1 for speaker, name in judged if name in correct_forms[mapping[speaker]])

    return NameAccuracy(
        recall=found / len(to_find) if to_find else None,
        precision=right / len(judged) if judged else None,
        named_reference=len(to_find),
        named_hypothesis=len(judged),
    )


def operator_confusion(reference: Sequence[Turn], hypothesis: Hypothesis) -> OperatorConfusion:
    """Return the share of microphone speech under a speaker of the wrong side."""
    operator_as_remote = remote_as_operator = total = 0.0
    for line in hypothesis.lines:
        if line.channel != "mic":
            continue
        by_operator = line.speaker in hypothesis.local
        for turn in reference:
            side = reference_side(turn.speaker)
            shared = min(line.end, turn.end) - max(line.start, turn.start)
            if side is None or shared <= 0:
                continue
            total += shared
            if side == "mic" and not by_operator:
                operator_as_remote += shared
            elif side == "system" and by_operator:
                remote_as_operator += shared
    return OperatorConfusion(
        operator_as_remote_s=operator_as_remote,
        remote_as_operator_s=remote_as_operator,
        microphone_s=total,
        rate=(operator_as_remote + remote_as_operator) / total if total > 0 else None,
    )


def echo_rate(
    hypothesis: Sequence[HypothesisLine], tolerance_s: float = ECHO_TOLERANCE_S
) -> EchoRate:
    """Return the share of microphone words that the call has at the same time."""
    system = [(line, set(_tokens(line.text))) for line in hypothesis if line.channel == "system"]
    total = echoed = marked = 0
    for line in hypothesis:
        if line.channel != "mic":
            continue
        tokens = _tokens(line.text)
        heard: set[str] = set()
        for other, words in system:
            if other.start - tolerance_s < line.end and line.start < other.end + tolerance_s:
                heard |= words
        total += len(tokens)
        echoed += sum(1 for token in tokens if token in heard)
        if line.origin == ECHO:
            marked += len(tokens)
    return EchoRate(
        rate=echoed / total if total else None,
        marked_rate=marked / total if total else None,
        microphone_words=total,
    )
