"""Turns the trace of a replay into the numbers of the report.

- First partial: the time from the first chunk of audio to the first text.
- Final latency: for each utterance, the time from the end of its audio to
  the first message that settled it. The end of the audio comes from the word
  times of the provider, or from the reference when the provider gives none.
- WER and CER: all the settled text against the reference, after normalisation.
- Speaker accuracy and cost: see `speakers` and the billing basis.
"""

from dataclasses import dataclass

from modebench.stats.percentiles import percentile
from modebench.stt.base import FINAL, PARTIAL
from modebench.stt.config import AudioItem, SttBilling, SttNormalization
from modebench.stt.replay import SessionTrace
from modebench.stt.speakers import FinalUtterance, speaker_accuracy
from modebench.stt.wer import ErrorRates, error_rates


@dataclass(frozen=True, slots=True)
class SessionMetrics:
    """The numbers of one replay."""

    first_partial_ms: float | None
    first_partial_after_speech_ms: float | None
    finals: list[FinalUtterance]
    final_latency_p50_ms: float | None
    final_latency_p95_ms: float | None
    rates: ErrorRates | None
    speaker_accuracy: float | None
    cost_usd: float
    hypothesis: str


def session_cost(billing: SttBilling, audio_ms: float, session_ms: float) -> float:
    """Return the cost of one session by the billing basis of the provider.

    `audio_ms` is the audio that the provider received, silence included. A
    session that failed to connect sent no audio, and then it costs nothing.
    """
    billed_ms = session_ms if billing.basis == "session_seconds" else audio_ms
    return billed_ms / 3_600_000.0 * billing.total_usd_per_hour


def collect_finals(trace: SessionTrace, audio: AudioItem) -> list[FinalUtterance]:
    """Return one settled utterance for each utterance number of the trace.

    The latency uses the first message that settled the utterance, because
    that is when the person sees settled text. The text is that of the last
    message, which is the best version. Word times include the silence that
    was sent before the audio, so that silence is taken out.
    """
    order: list[int] = []
    first_at: dict[int, float] = {}
    text: dict[int, str] = {}
    speaker: dict[int, str] = {}
    start: dict[int, float] = {}
    end: dict[int, float] = {}
    counter = 0
    for item in trace.events:
        event = item.event
        if event.kind != FINAL:
            continue
        if event.utterance is None:
            key = -1 - counter
            counter += 1
        else:
            key = event.utterance
        if key not in first_at:
            order.append(key)
            first_at[key] = item.at_ms
        if event.text or key not in text:
            text[key] = event.text
        if event.speaker:
            speaker[key] = event.speaker
        if event.audio_start_ms is not None:
            start[key] = event.audio_start_ms - trace.preroll_ms
        if event.audio_end_ms is not None:
            end[key] = event.audio_end_ms - trace.preroll_ms
    references = sorted(audio.utterances, key=lambda item: item.start_ms)
    finals: list[FinalUtterance] = []
    for position, key in enumerate(order):
        if not text[key].strip():
            continue
        audio_end = end.get(key)
        if audio_end is None:
            audio_end = float(references[min(position, len(references) - 1)].end_ms)
        finals.append(
            FinalUtterance(
                index=len(finals),
                text=text[key],
                speaker=speaker.get(key),
                received_ms=first_at[key],
                audio_start_ms=start.get(key),
                audio_end_ms=audio_end,
                latency_ms=first_at[key] - audio_end,
            )
        )
    return finals


def _first_text_ms(trace: SessionTrace) -> float | None:
    for item in trace.events:
        if item.event.kind in (PARTIAL, FINAL) and item.event.text.strip():
            return item.at_ms
    return None


def session_metrics(
    trace: SessionTrace,
    audio: AudioItem,
    billing: SttBilling,
    normalization: SttNormalization,
) -> SessionMetrics:
    """Return the numbers of one replay."""
    finals = collect_finals(trace, audio)
    latencies: list[float] = []
    for final in finals:
        if final.latency_ms is not None:
            latencies.append(final.latency_ms)
    first_ms = _first_text_ms(trace)
    speech_start = min(item.start_ms for item in audio.utterances)
    hypothesis = " ".join(final.text for final in finals)
    return SessionMetrics(
        first_partial_ms=first_ms,
        first_partial_after_speech_ms=None if first_ms is None else first_ms - speech_start,
        finals=finals,
        final_latency_p50_ms=percentile(latencies, 50) if latencies else None,
        final_latency_p95_ms=percentile(latencies, 95) if latencies else None,
        rates=error_rates(audio.reference_text, hypothesis, normalization),
        speaker_accuracy=speaker_accuracy(finals, audio.utterances),
        cost_usd=session_cost(billing, trace.sent_ms, trace.session_ms),
        hypothesis=hypothesis,
    )
