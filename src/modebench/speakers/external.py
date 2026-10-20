"""DER and JER through pyannote.metrics, cpWER and tcpWER through MeetEval.

The two libraries are not dependencies of modebench. They are large, MeetEval
0.4.3 is published as a source package only (its install compiles C++ code),
and only this measure uses them. Add them for one command:

    uv run --with pyannote.metrics==4.2.0 --with meeteval==0.4.3 modebench speakers score ...

Without them, these four numbers are not measured, and the score says so.

The settings are those of ADR 0001 of stt-bridge, section 6.1: no collar and
the overlapped speech scored for DER and JER, an explicit UEM, and a collar of
5 seconds for tcpWER. The words go to MeetEval already normalised by the
Portuguese normaliser of this repository, and MeetEval gets no normaliser of
its own, because its normalisers remove accented letters.

Each annotation is reduced to its support before it is scored: an echo under
the speaker of the call line it repeats is that speaker speaking once, and
two tracks of one label at the same time would count as a false alarm.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from modebench.speakers.gold import Turn, WordSegment
from modebench.speakers.hypothesis import ECHO, HypothesisLine
from modebench.stt.normalize import normalize_pt

PYANNOTE = "pyannote.metrics==4.2.0"
MEETEVAL = "meeteval==0.4.3"
TCPWER_COLLAR_S = 5.0


@dataclass(frozen=True, slots=True)
class DiarizationRates:
    """DER and JER of one meeting."""

    der: float
    jer: float


@dataclass(frozen=True, slots=True)
class PermutationRates:
    """cpWER and tcpWER of one meeting. None when one side has no word."""

    cpwer: float | None
    tcpwer: float | None


def diarization_rates(
    reference: Sequence[Turn],
    hypothesis: Sequence[HypothesisLine],
    scored: Sequence[tuple[float, float]],
    uri: str,
) -> DiarizationRates | None:
    """Return DER and JER, or None if pyannote.metrics is not installed.

    Give only the loopback turns and lines to score the loopback alone.
    """
    try:
        from pyannote.core import Annotation, Segment, Timeline
        from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
    except ImportError:
        return None

    truth = Annotation(uri=uri)
    for index, turn in enumerate(reference):
        if turn.end > turn.start:
            truth[Segment(turn.start, turn.end), f"r{index}"] = turn.speaker
    found = Annotation(uri=uri)
    for index, line in enumerate(hypothesis):
        if line.end > line.start:
            found[Segment(line.start, line.end), f"h{index}"] = line.speaker
    uem = Timeline([Segment(start, end) for start, end in scored], uri=uri)
    truth = truth.support()
    found = found.support()

    der = DiarizationErrorRate(collar=0.0, skip_overlap=False)(truth, found, uem=uem)
    jer = JaccardErrorRate(collar=0.0, skip_overlap=False)(truth, found, uem=uem)
    return DiarizationRates(der=float(der), jer=float(jer))


def _rate(results: Any, session: str) -> float | None:
    value = results[session].error_rate
    return None if value is None else float(value)


def permutation_rates(
    reference: Sequence[WordSegment],
    hypothesis: Sequence[HypothesisLine],
    session: str,
    collar_s: float = TCPWER_COLLAR_S,
) -> PermutationRates | None:
    """Return cpWER and tcpWER, or None if MeetEval is not installed."""
    try:
        from meeteval.wer import cpwer, tcpwer
    except ImportError:
        return None

    truth = [
        {
            "session_id": session,
            "speaker": segment.speaker,
            "start_time": segment.start,
            "end_time": segment.end,
            "words": words,
        }
        for segment in reference
        if (words := normalize_pt(segment.words))
    ]
    found = [
        {
            "session_id": session,
            "speaker": line.speaker,
            "start_time": line.start,
            "end_time": line.end,
            "words": words,
        }
        for line in hypothesis
        if line.origin != ECHO and (words := normalize_pt(line.text))
    ]
    if not truth or not found:
        return PermutationRates(cpwer=None, tcpwer=None)

    return PermutationRates(
        cpwer=_rate(cpwer(truth, found, normalizer=None), session),
        tcpwer=_rate(tcpwer(truth, found, collar=collar_s, normalizer=None), session),
    )
