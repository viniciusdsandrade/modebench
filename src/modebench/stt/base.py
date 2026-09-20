"""What the replay needs from a speech to text provider.

An adapter is pure: it builds the connect URL and the headers, it wraps a
chunk of audio, and its parser turns a server message into events. It opens
no socket, so each adapter is tested against a fake connection.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

OPEN = "open"
PARTIAL = "partial"
FINAL = "final"
OTHER = "other"
CLOSED = "closed"
ERROR = "error"


@dataclass(frozen=True, slots=True)
class SttEvent:
    """One normalised event. Times are positions in the audio, in milliseconds."""

    kind: str
    text: str = ""
    utterance: int | None = None
    audio_start_ms: float | None = None
    audio_end_ms: float | None = None
    speaker: str | None = None
    message: str = ""


class SttParser(Protocol):
    """Reads the messages of one session. A parser can keep state between messages."""

    def parse(self, raw: str | bytes) -> list[SttEvent]:
        """Return the events of one server message."""
        ...


class SttAdapter(Protocol):
    """The dialect of one provider."""

    name: str

    def connect_url(self, sample_rate: int) -> str:
        """Return the URL of a session for audio of this sample rate."""
        ...

    def headers(self, api_key: str) -> dict[str, str]:
        """Return the headers that carry the credential."""
        ...

    def audio_message(self, pcm: bytes, sample_rate: int) -> str | bytes:
        """Return the message that carries one chunk of audio."""
        ...

    def closing_messages(self) -> list[str]:
        """Return the messages that end a session. The list can be empty."""
        ...

    def new_parser(self) -> SttParser:
        """Return a parser for a new session."""
        ...


def decode_object(raw: str | bytes) -> dict[str, Any] | None:
    """Return the JSON object of a server message, or None if it is not one."""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def scalar_text(value: object) -> str:
    """Return a query value as text. A boolean is `true` or `false`, as the APIs expect."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def join_url(base: str, pairs: list[tuple[str, str]]) -> str:
    """Return `base` with the pairs as a percent-encoded query."""
    query = "&".join(f"{quote(key, safe='')}={quote(value, safe='')}" for key, value in pairs)
    return f"{base}?{query}" if query else base


def number(mapping: Mapping[str, Any], key: str) -> float | None:
    """Return a numeric field, or None if it is absent or not a number."""
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
