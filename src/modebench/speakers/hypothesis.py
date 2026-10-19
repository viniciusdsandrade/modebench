"""What the stt-bridge application wrote: the lines of one meeting, with their speakers.

The input is the detail of one meeting, as `GET /api/meetings/{id}/detail`
returns it, or the `.json` export of the page, which holds the same detail
under `meeting`.

With attribution on (`native` or `fusion`), the speaker of a line is its
meeting label (`speaker_label`). One label is one person for the whole meeting
on the two channels: an echo on the microphone has the label of the person on
the call. With attribution off, a label of the service is correct only on its
channel, so the speaker is the channel and the label (`mic:A`, `system:A`).
A line without a label has the channel and `?`.

The names come with where they came from (`config`, `manual`, `rule` or
`model`) and how sure the application was, so the score can tell a name the
application found from a name it was given.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modebench.errors import DatasetError

MODE_OFF = "off"
MODES = (MODE_OFF, "native", "fusion")
ECHO = "echo"
UNKNOWN = "?"
CHANNELS = ("mic", "system")
LOCAL = "local"
FOUND_ORIGINS = ("rule", "model")


@dataclass(frozen=True, slots=True)
class HypothesisLine:
    """One settled line of the application, in seconds from the start of the gold audio."""

    segment_id: int
    channel: str
    start: float
    end: float
    text: str
    speaker: str
    origin: str | None


@dataclass(frozen=True, slots=True)
class HypothesisName:
    """The name of one speaker of the application, with where it came from."""

    name: str
    origin: str | None = None
    confidence_pct: int | None = None

    def found(self) -> bool:
        """Return True if the application found the name itself, by its rules or its model."""
        return self.origin in FOUND_ORIGINS


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """The lines of one meeting, the names of its speakers and who the operator is."""

    meeting_id: int
    lines: list[HypothesisLine]
    names: dict[str, HypothesisName]
    local: frozenset[str] = frozenset()

    def speakers(self) -> list[str]:
        """Return the speakers of the lines, in the order of their first line."""
        return list(dict.fromkeys(line.speaker for line in self.lines))


def _detail(document: object) -> dict[str, Any]:
    if isinstance(document, dict) and isinstance(document.get("meeting"), dict):
        document = document["meeting"]
    if not isinstance(document, dict) or not isinstance(document.get("segments"), list):
        raise DatasetError("the hypothesis is not the detail of a meeting: it has no segments")
    return document


def _key(segment: dict[str, Any], channel: str, attributed: bool) -> str:
    label = segment.get("speaker_label")
    if attributed and isinstance(label, str) and label:
        return label
    speaker = segment.get("speaker")
    if not attributed and isinstance(speaker, str) and speaker:
        return f"{channel}:{speaker}"
    return f"{channel}:{UNKNOWN}"


def read_hypothesis(document: object, *, mode: str, offset_s: float = 0.0) -> Hypothesis:
    """Return the hypothesis of one meeting from the decoded JSON of the application."""
    if mode not in MODES:
        raise DatasetError(f"unknown mode {mode}. Use {', '.join(MODES)}")
    detail = _detail(document)
    attributed = mode != MODE_OFF

    lines: list[HypothesisLine] = []
    labelled = False
    for index, segment in enumerate(detail["segments"]):
        if not isinstance(segment, dict):
            raise DatasetError(f"segment {index} of the hypothesis is not an object")
        channel = segment.get("source")
        if channel not in CHANNELS:
            raise DatasetError(f"segment {index} has the unknown channel {channel!r}")
        try:
            segment_id = int(segment["id"])
            start = float(segment["ts_start_ms"]) / 1000.0 + offset_s
            end = float(segment["ts_end_ms"]) / 1000.0 + offset_s
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetError(f"segment {index} has no id or no times") from exc
        text = segment.get("text")
        origin = segment.get("speaker_origin")
        labelled = labelled or isinstance(segment.get("speaker_label"), str)
        lines.append(
            HypothesisLine(
                segment_id=segment_id,
                channel=str(channel),
                start=start,
                end=max(start, end),
                text=text if isinstance(text, str) else "",
                speaker=_key(segment, str(channel), attributed),
                origin=origin if isinstance(origin, str) else None,
            )
        )

    if attributed and lines and not labelled:
        raise DatasetError(
            "the hypothesis has no meeting label. It was recorded with attribution off: "
            "use --mode off"
        )
    if not attributed and labelled:
        raise DatasetError(
            "the hypothesis has meeting labels. It was recorded with attribution on: "
            "use --mode native or --mode fusion"
        )

    names: dict[str, HypothesisName] = {}
    local: set[str] = set()
    if attributed:
        for speaker in detail.get("speakers") or []:
            if not isinstance(speaker, dict):
                continue
            label = speaker.get("label")
            if not isinstance(label, str):
                continue
            if speaker.get("kind") == LOCAL:
                local.add(label)
            name = speaker.get("name")
            if isinstance(name, str) and name.strip():
                origin = speaker.get("name_origin")
                confidence = speaker.get("name_confidence_pct")
                names[label] = HypothesisName(
                    name=name.strip(),
                    origin=origin if isinstance(origin, str) else None,
                    confidence_pct=confidence if isinstance(confidence, int) else None,
                )
    else:
        # Without attribution the microphone is the operator, line by line.
        local = {line.speaker for line in lines if line.channel == "mic"}

    lines.sort(key=lambda line: (line.start, line.segment_id))
    meeting_id = detail.get("id")
    return Hypothesis(
        meeting_id=meeting_id if isinstance(meeting_id, int) else -1,
        lines=lines,
        names=names,
        local=frozenset(local),
    )


def load_hypothesis(path: Path, *, mode: str, offset_s: float = 0.0) -> Hypothesis:
    """Read the hypothesis of one meeting from a JSON file of the application."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetError(f"hypothesis file not found: {path}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError(f"{path} cannot be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path} is not valid JSON: {exc}") from exc
    return read_hypothesis(document, mode=mode, offset_s=offset_s)
