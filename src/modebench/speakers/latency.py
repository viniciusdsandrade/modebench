"""What speaker attribution adds to the life of a line, read from the probes of stt-bridge.

With `STT_BRIDGE_LATENCY_PROBE` in its environment, the recorder of stt-bridge
writes three kinds of line to its error stream:

- `latency-probe speaker-arrived source=<channel> utterance=<id> wall_unix_ms=<t>`
  when a settled line reaches the recorder;
- `latency-probe speaker-attributed source=<channel> utterance=<id> segment=<row>
  speaker=<label> wall_unix_ms=<t>` when its speaker is recorded;
- `latency-probe speaker-named speaker=<label> wall_unix_ms=<t>` when a
  speaker gets a name.

The added latency of attribution is the time from the first to the second, for
each line. The name delay of a line is the time from its attribution to the
first name of its speaker, for each line that was said before its speaker had
a name: those are the earlier lines a name reaches. A probe carries labels,
never names.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from modebench.errors import DatasetError
from modebench.stats.percentiles import percentile

PROBE = "latency-probe"
ARRIVED = "speaker-arrived"
ATTRIBUTED = "speaker-attributed"
NAMED = "speaker-named"


@dataclass(frozen=True, slots=True)
class SpeakerLatency:
    """The added latency of attribution and the delay of names, in milliseconds."""

    lines: int
    attribution_p50_ms: float | None
    attribution_p95_ms: float | None
    named_lines: int
    name_delay_p50_ms: float | None
    name_delay_p95_ms: float | None


def _fields(line: str) -> tuple[str, dict[str, str]] | None:
    words = line.split()
    if len(words) < 2 or words[0] != PROBE:
        return None
    fields = dict(word.split("=", 1) for word in words[2:] if "=" in word)
    return words[1], fields


def speaker_latency(lines: Iterable[str]) -> SpeakerLatency:
    """Return the latency figures of one recording, from the lines of its error stream.

    Lines that are not speaker probes are skipped, so the whole error stream
    can be given.
    """
    arrived: dict[tuple[str, str], int] = {}
    attributed: list[tuple[str, int, int]] = []
    named: dict[str, int] = {}
    for line in lines:
        parsed = _fields(line)
        if parsed is None:
            continue
        stage, fields = parsed
        try:
            at = int(fields["wall_unix_ms"])
            if stage == ARRIVED:
                arrived[(fields["source"], fields["utterance"])] = at
            elif stage == ATTRIBUTED:
                started = arrived.pop((fields["source"], fields["utterance"]), at)
                attributed.append((fields["speaker"], started, at))
            elif stage == NAMED:
                named.setdefault(fields["speaker"], at)
        except (KeyError, ValueError):
            continue

    added = [float(done - started) for _, started, done in attributed]
    delays = [
        float(named[speaker] - done)
        for speaker, _, done in attributed
        if speaker in named and named[speaker] >= done
    ]
    return SpeakerLatency(
        lines=len(added),
        attribution_p50_ms=percentile(added, 50) if added else None,
        attribution_p95_ms=percentile(added, 95) if added else None,
        named_lines=len(delays),
        name_delay_p50_ms=percentile(delays, 50) if delays else None,
        name_delay_p95_ms=percentile(delays, 95) if delays else None,
    )


def load_speaker_latency(path: Path) -> SpeakerLatency:
    """Read the error stream of one recording from `path`."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise DatasetError(f"probe log not found: {path}") from exc
    except OSError as exc:
        raise DatasetError(f"{path} cannot be read: {exc}") from exc
    return speaker_latency(text.splitlines())
