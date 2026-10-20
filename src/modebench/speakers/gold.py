"""The gold set: who spoke when, which words they said, and their real names.

The index is a YAML file. Each meeting names its files, relative to the index:

    meetings:
      - id: meeting-12
        rttm: meeting-12.rttm
        uem: meeting-12.uem
        words: meeting-12.seglst.json
        names:
          mic-op: Marina
          sys-ana: [Ana Souza, Ana]
        offset_s: 0.0

- RTTM: one `SPEAKER` line for each turn. A label is unique in its file and
  says the channel (`mic-op`, `sys-ana`), because the channel field is not
  read. The other line types are ignored.
- UEM: the parts of the meeting that are scored, one `<file> <channel>
  <start> <end>` line for each part. Without a UEM file, the scored part is
  from the start of the first turn to the end of the last turn.
- SegLST: the words, in the JSON form of MeetEval: a list of segments with
  `speaker`, `start_time`, `end_time` and `words`.
- `names`: the real name of each reference speaker that has one. A list gives
  each form that is correct, for example the full name and the first name.
- `offset_s`: the seconds to add to the times of the application, when the
  audio of the gold set does not start at the start of the meeting.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import yaml
from pydantic import Field, ValidationError, model_validator

from modebench.config import StrictModel
from modebench.errors import DatasetError

SPEAKER_LINE = "SPEAKER"
RTTM_FIELDS = 8
UEM_FIELDS = 4


@dataclass(frozen=True, slots=True)
class Turn:
    """One stretch of speech of one reference speaker, in seconds."""

    speaker: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class WordSegment:
    """One segment of words of one reference speaker, in seconds."""

    speaker: str
    start: float
    end: float
    words: str


@dataclass(frozen=True, slots=True)
class GoldMeeting:
    """The gold set of one meeting."""

    id: str
    turns: list[Turn]
    scored: list[tuple[float, float]]
    words: list[WordSegment] = field(default_factory=list)
    names: dict[str, tuple[str, ...]] = field(default_factory=dict)
    offset_s: float = 0.0


class GoldEntry(StrictModel):
    """One meeting of the index."""

    id: str = Field(min_length=1)
    rttm: Path
    uem: Path | None = None
    words: Path | None = None
    names: dict[str, str | list[str]] = Field(default_factory=dict)
    offset_s: float = 0.0


class GoldIndex(StrictModel):
    """The index of the gold set."""

    meetings: list[GoldEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        seen: set[str] = set()
        for entry in self.meetings:
            if entry.id in seen:
                raise ValueError(f"meeting {entry.id} is in the index two times")
            seen.add(entry.id)
        return self


def _number(text: str, source: str, line: int) -> float:
    try:
        return float(text)
    except ValueError as exc:
        raise DatasetError(f"{source}, line {line}: {text!r} is not a number") from exc


def parse_rttm(text: str, source: str) -> list[Turn]:
    """Return the turns of an RTTM file that holds one meeting."""
    turns: list[Turn] = []
    files: set[str] = set()
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] != SPEAKER_LINE:
            continue
        if len(fields) < RTTM_FIELDS:
            raise DatasetError(f"{source}, line {number}: a SPEAKER line has 8 fields or more")
        onset = _number(fields[3], source, number)
        duration = _number(fields[4], source, number)
        if onset < 0 or duration < 0:
            raise DatasetError(f"{source}, line {number}: a time is negative")
        files.add(fields[1])
        turns.append(Turn(speaker=fields[7], start=onset, end=onset + duration))
    if len(files) > 1:
        raise DatasetError(f"{source} holds more than one meeting: {', '.join(sorted(files))}")
    if not turns:
        raise DatasetError(f"{source} has no SPEAKER line")
    return turns


def parse_uem(text: str, source: str) -> list[tuple[float, float]]:
    """Return the scored parts of a UEM file, in seconds."""
    parts: list[tuple[float, float]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < UEM_FIELDS:
            raise DatasetError(f"{source}, line {number}: a UEM line has 4 fields")
        start = _number(fields[2], source, number)
        end = _number(fields[3], source, number)
        if start < 0 or end <= start:
            raise DatasetError(f"{source}, line {number}: the end is not after the start")
        parts.append((start, end))
    if not parts:
        raise DatasetError(f"{source} has no scored part")
    return parts


def parse_seglst(text: str, source: str) -> list[WordSegment]:
    """Return the word segments of a SegLST file that holds one meeting."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{source} is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise DatasetError(f"{source} is not a list of segments")
    segments: list[WordSegment] = []
    sessions: set[str] = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise DatasetError(f"{source}, segment {index}: a segment is an object")
        try:
            speaker = item["speaker"]
            start = float(item["start_time"])
            end = float(item["end_time"])
            words = item["words"]
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetError(
                f"{source}, segment {index}: speaker, start_time, end_time and words are needed"
            ) from exc
        if not isinstance(speaker, str) or not isinstance(words, str):
            raise DatasetError(f"{source}, segment {index}: speaker and words are text")
        session = item.get("session_id")
        if session is not None:
            sessions.add(str(session))
        segments.append(WordSegment(speaker=speaker, start=start, end=end, words=words))
    if len(sessions) > 1:
        raise DatasetError(f"{source} holds more than one meeting: {', '.join(sorted(sessions))}")
    return segments


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DatasetError(f"gold file not found: {path}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError(f"{path} cannot be read: {exc}") from exc


def load_index(path: Path) -> GoldIndex:
    """Read and check the index of the gold set."""
    text = _read(path)
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise DatasetError(f"{path} is not valid YAML: {exc}") from exc
    try:
        return GoldIndex.model_validate(data)
    except ValidationError as exc:
        raise DatasetError(f"{path} is not a valid gold index:\n{exc}") from exc


def load_gold(index_path: Path, meeting_id: str) -> GoldMeeting:
    """Read the gold set of one meeting of the index."""
    index = load_index(index_path)
    entry = next((item for item in index.meetings if item.id == meeting_id), None)
    if entry is None:
        known = ", ".join(item.id for item in index.meetings)
        raise DatasetError(f"meeting {meeting_id} is not in {index_path}. The index has: {known}")

    base = index_path.parent
    rttm_path = base / entry.rttm
    turns = parse_rttm(_read(rttm_path), str(rttm_path))
    if entry.uem is not None:
        uem_path = base / entry.uem
        scored = parse_uem(_read(uem_path), str(uem_path))
    else:
        scored = [(min(turn.start for turn in turns), max(turn.end for turn in turns))]
    speakers = {turn.speaker for turn in turns}
    words: list[WordSegment] = []
    if entry.words is not None:
        words_path = base / entry.words
        words = parse_seglst(_read(words_path), str(words_path))
        strangers = sorted({segment.speaker for segment in words} - speakers)
        if strangers:
            raise DatasetError(
                f"{words_path.name} has speakers that {rttm_path.name} does not have: "
                f"{', '.join(strangers)}"
            )

    names: dict[str, tuple[str, ...]] = {}
    for speaker, given in entry.names.items():
        if speaker not in speakers:
            raise DatasetError(
                f"{index_path}: meeting {meeting_id} names {speaker}, "
                f"which is not a speaker of {rttm_path.name}"
            )
        forms = tuple(form.strip() for form in ([given] if isinstance(given, str) else given))
        if not forms or not all(forms):
            raise DatasetError(f"{index_path}: the name of {speaker} is empty")
        names[speaker] = forms

    return GoldMeeting(
        id=meeting_id,
        turns=turns,
        scored=scored,
        words=words,
        names=names,
        offset_s=entry.offset_s,
    )


def _relabelled_rttm(text: str, meeting_id: str, prefix: str) -> str:
    lines = []
    for raw in text.splitlines():
        fields = raw.split()
        if len(fields) >= RTTM_FIELDS and fields[0] == SPEAKER_LINE:
            fields[1] = meeting_id
            if not fields[7].startswith(prefix):
                fields[7] = prefix + fields[7]
            lines.append(" ".join(fields))
    return "\n".join(lines) + "\n"


def _renamed_uem(text: str, meeting_id: str) -> str:
    lines = []
    for raw in text.splitlines():
        fields = raw.split()
        if len(fields) >= UEM_FIELDS:
            fields[0] = meeting_id
            lines.append(" ".join(fields))
    return "\n".join(lines) + "\n"


def import_rttm(
    rttm_path: Path,
    uem_path: Path | None,
    meeting_id: str,
    prefix: str,
    out_dir: Path,
) -> Path:
    """Add one annotated meeting of a public set to the gold set below `out_dir`.

    Every label gets `prefix` (`sys-` for a meeting played as the computer
    audio, `mic-` for one played as the microphone), because the side of a
    reference speaker is read from its label. The files are copied under the
    meeting id, and the meeting is added to `gold.yaml`, replacing an entry
    of the same id. The answer is the path of the index.
    """
    rttm_text = _read(rttm_path)
    parse_rttm(rttm_text, str(rttm_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{meeting_id}.rttm").write_text(
        _relabelled_rttm(rttm_text, meeting_id, prefix), encoding="utf-8"
    )
    entry: dict[str, object] = {"id": meeting_id, "rttm": f"{meeting_id}.rttm"}
    if uem_path is not None:
        uem_text = _read(uem_path)
        parse_uem(uem_text, str(uem_path))
        (out_dir / f"{meeting_id}.uem").write_text(
            _renamed_uem(uem_text, meeting_id), encoding="utf-8"
        )
        entry["uem"] = f"{meeting_id}.uem"

    index_path = out_dir / "gold.yaml"
    meetings: list[dict[str, object]] = []
    if index_path.exists():
        meetings = [
            item.model_dump(mode="json", exclude_defaults=True)
            for item in load_index(index_path).meetings
            if item.id != meeting_id
        ]
    meetings.append(entry)
    index_path.write_text(
        yaml.safe_dump({"meetings": meetings}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return index_path
