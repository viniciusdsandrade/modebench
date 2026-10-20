"""The score of one meeting: every measure of ADR 0001 of stt-bridge.

The latency comes from the probe lines the recorder wrote to its error stream
(`modebench.speakers.latency`), and the cost from `stt-bridge billing
--meeting`, given in dollars; the cost per hour divides it by the length of
the meeting.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from modebench.speakers.external import (
    MEETEVAL,
    PYANNOTE,
    diarization_rates,
    permutation_rates,
)
from modebench.speakers.gold import GoldMeeting
from modebench.speakers.hypothesis import Hypothesis
from modebench.speakers.latency import SpeakerLatency
from modebench.speakers.metrics import (
    NameCheck,
    echo_rate,
    name_accuracy,
    name_checks,
    operator_confusion,
    reference_side,
    time_mapping,
    word_diarization,
)

SCORE_SUFFIX = ".score.json"
# The threshold at which stt-bridge gives a name (`NAME_THRESHOLD` in its
# `crates/core/src/naming.rs`), written down with the confidences it judges.
APP_NAME_THRESHOLD_PCT = 75


@dataclass(frozen=True, slots=True)
class SpeakerScore:
    """The measures of one meeting. None is a measure that the inputs cannot give."""

    meeting: str
    mode: str
    der: float | None
    jer: float | None
    loopback_der: float | None
    operator_confusion: float | None
    wder: float | None
    wer: float | None
    cpwer: float | None
    tcpwer: float | None
    name_recall: float | None
    name_precision: float | None
    echo_rate: float | None
    echo_marked_rate: float | None
    reference_speakers: int
    hypothesis_speakers: int
    mapping: dict[str, str]
    names: list[NameCheck] = field(default_factory=list)
    name_threshold_pct: int = APP_NAME_THRESHOLD_PCT
    latency: SpeakerLatency | None = None
    duration_s: float | None = None
    cost_usd: float | None = None
    cost_per_hour_usd: float | None = None
    not_measured: list[str] = field(default_factory=list)


def score_meeting(
    gold: GoldMeeting,
    hypothesis: Hypothesis,
    *,
    mode: str,
    external: bool = True,
    latency: SpeakerLatency | None = None,
    cost_usd: float | None = None,
) -> SpeakerScore:
    """Return the score of one meeting.

    With `external` false, the measures of pyannote.metrics and MeetEval are
    not attempted, which is what the tests of this package use.
    """
    missing: list[str] = []
    lines = hypothesis.lines
    loopback_turns = [turn for turn in gold.turns if reference_side(turn.speaker) == "system"]
    loopback_lines = [line for line in lines if line.channel == "system"]

    rates = loopback = None
    if not external:
        missing.append("DER, JER and loopback DER: not attempted")
    else:
        rates = diarization_rates(gold.turns, lines, gold.scored, gold.id)
        if rates is None:
            missing.append(f"DER, JER and loopback DER: add {PYANNOTE}")
        elif loopback_turns:
            loopback = diarization_rates(loopback_turns, loopback_lines, gold.scored, gold.id)
        else:
            missing.append("loopback DER: no reference label starts with sys-")

    permuted = None
    if not gold.words:
        missing.append("WDER, cpWER and tcpWER: the gold set has no words for this meeting")
    elif not external:
        missing.append("cpWER and tcpWER: not attempted")
    else:
        permuted = permutation_rates(gold.words, lines, gold.id)
        if permuted is None:
            missing.append(f"cpWER and tcpWER: add {MEETEVAL}")
        elif permuted.cpwer is None:
            missing.append("cpWER and tcpWER: the hypothesis has no words")

    words = word_diarization(gold.words, lines) if gold.words else None
    mapping = time_mapping(gold.turns, lines)
    names = name_accuracy(gold.names, hypothesis, mapping)
    if not gold.names:
        missing.append("name accuracy: the gold set names nobody in this meeting")
    confusion = operator_confusion(gold.turns, hypothesis)
    if confusion.rate is None:
        missing.append("operator confusion: no microphone line meets a mic- or sys- turn")
    echo = echo_rate(lines)
    if latency is None:
        missing.append("latency: no probe log was given")
    duration = hypothesis.duration_s
    cost_per_hour = None
    if cost_usd is None:
        missing.append("cost per hour: no cost was given")
    elif duration is None:
        missing.append("cost per hour: the meeting has no end")
    else:
        cost_per_hour = cost_usd * 3600.0 / duration

    return SpeakerScore(
        meeting=gold.id,
        mode=mode,
        der=rates.der if rates else None,
        jer=rates.jer if rates else None,
        loopback_der=loopback.der if loopback else None,
        operator_confusion=confusion.rate,
        wder=words.wder if words else None,
        wer=words.wer if words else None,
        cpwer=permuted.cpwer if permuted else None,
        tcpwer=permuted.tcpwer if permuted else None,
        name_recall=names.recall,
        name_precision=names.precision,
        echo_rate=echo.rate,
        echo_marked_rate=echo.marked_rate,
        reference_speakers=len({turn.speaker for turn in gold.turns}),
        hypothesis_speakers=len(hypothesis.speakers()),
        mapping=mapping,
        names=name_checks(gold.names, hypothesis, mapping),
        latency=latency,
        duration_s=duration,
        cost_usd=cost_usd,
        cost_per_hour_usd=cost_per_hour,
        not_measured=missing,
    )


def write_score(score: SpeakerScore, path: Path) -> None:
    """Write the score as JSON, below a directory that is made if it is not there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(score), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _percent(value: float | None) -> str:
    return "not measured" if value is None else f"{value * 100:.2f} %"


def _ms(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.0f} ms"


def render_score(score: SpeakerScore) -> list[str]:
    """Return the score as lines for the terminal."""
    lines = [
        f"meeting {score.meeting}, mode {score.mode}: "
        f"{score.reference_speakers} reference speakers, "
        f"{score.hypothesis_speakers} hypothesis speakers",
        f"  DER {_percent(score.der)}, JER {_percent(score.jer)}, "
        f"loopback DER {_percent(score.loopback_der)}",
        f"  operator confusion {_percent(score.operator_confusion)}",
        f"  WDER {_percent(score.wder)}, WER {_percent(score.wer)}",
        f"  cpWER {_percent(score.cpwer)}, tcpWER {_percent(score.tcpwer)}",
        f"  names: recall {_percent(score.name_recall)}, "
        f"precision {_percent(score.name_precision)}",
        f"  echo: {_percent(score.echo_rate)} of the microphone words, "
        f"{_percent(score.echo_marked_rate)} marked",
    ]
    if score.latency is not None:
        lines.append(
            f"  attribution adds {_ms(score.latency.attribution_p50_ms)} p50, "
            f"{_ms(score.latency.attribution_p95_ms)} p95 over {score.latency.lines} lines; "
            f"a name reaches an earlier line after {_ms(score.latency.name_delay_p50_ms)} p50, "
            f"{_ms(score.latency.name_delay_p95_ms)} p95 over {score.latency.named_lines} lines"
        )
    if score.cost_per_hour_usd is not None:
        lines.append(
            f"  cost: {score.cost_usd:.4f} USD, {score.cost_per_hour_usd:.4f} USD per hour"
        )
    for check in score.names:
        verdict = {True: "right", False: "wrong", None: "not judged"}[check.correct]
        confidence = "" if check.confidence_pct is None else f", {check.confidence_pct} %"
        lines.append(
            f"  name {check.speaker}: {check.name} ({check.origin or 'unknown'}{confidence}, "
            f"threshold {score.name_threshold_pct} %): {verdict}"
        )
    lines.extend(f"  not measured: {reason}" for reason in score.not_measured)
    return lines
