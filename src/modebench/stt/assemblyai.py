"""The AssemblyAI streaming dialect (Universal Streaming, API version 3).

Reference: https://www.assemblyai.com/docs/api-reference/streaming-api/streaming-api

- The session is configured on the connect URL. A list value is a JSON array.
- The key goes in the `Authorization` header, with no `Bearer` prefix.
- Audio is a binary message of raw PCM, from 50 ms to 1000 ms long.
- A `Turn` message carries the transcript. `end_of_turn` makes it final. Word
  times are in milliseconds. `turn_order` identifies the utterance.
- `{"type": "Terminate"}` ends the session. The session is billed for the time
  that the connection is open, so the replay always ends it.
"""

import json
from collections.abc import Mapping
from typing import Any

from modebench.stt.base import (
    CLOSED,
    ERROR,
    FINAL,
    OPEN,
    OTHER,
    PARTIAL,
    SttEvent,
    SttParser,
    decode_object,
    join_url,
    number,
    scalar_text,
)

DEFAULT_URL = "wss://streaming.assemblyai.com/v3/ws"
TERMINATE = '{"type":"Terminate"}'


def _words(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    words = message.get("words")
    if not isinstance(words, list):
        return []
    return [word for word in words if isinstance(word, dict)]


def _turn(message: Mapping[str, Any]) -> SttEvent:
    words = _words(message)
    transcript = message.get("transcript")
    text = transcript if isinstance(transcript, str) else ""
    if not text:
        text = " ".join(str(word.get("text", "")) for word in words).strip()
    order = message.get("turn_order")
    speaker = message.get("speaker_label")
    return SttEvent(
        kind=FINAL if message.get("end_of_turn") is True else PARTIAL,
        text=text,
        utterance=order if isinstance(order, int) and not isinstance(order, bool) else None,
        audio_start_ms=number(words[0], "start") if words else None,
        audio_end_ms=number(words[-1], "end") if words else None,
        speaker=speaker if isinstance(speaker, str) and speaker else None,
    )


class AssemblyAiParser:
    """Reads AssemblyAI messages. The protocol numbers its turns, so no state is kept."""

    def parse(self, raw: str | bytes) -> list[SttEvent]:
        """Return the events of one server message."""
        message = decode_object(raw)
        if message is None:
            return [SttEvent(kind=OTHER, message="not a JSON object")]
        kind = message.get("type")
        if kind == "Begin":
            return [SttEvent(kind=OPEN)]
        if kind == "Turn":
            return [_turn(message)]
        if kind == "Termination":
            return [SttEvent(kind=CLOSED)]
        if kind == "Error":
            detail = f"{message.get('error_code', '')}: {message.get('error', '')}".strip(": ")
            return [SttEvent(kind=ERROR, message=detail)]
        return [SttEvent(kind=OTHER, message=str(kind))]


class AssemblyAiAdapter:
    """Builds the requests of the AssemblyAI dialect."""

    name = "assemblyai"

    def __init__(self, query: Mapping[str, Any], url: str = "") -> None:
        self._query = dict(query)
        self._url = url or DEFAULT_URL

    def connect_url(self, sample_rate: int) -> str:
        """Return the connect URL. The raw query of the config comes after the audio format."""
        pairs = [("encoding", "pcm_s16le"), ("sample_rate", str(sample_rate))]
        for key, value in self._query.items():
            if isinstance(value, list | dict):
                pairs.append((key, json.dumps(value, ensure_ascii=False, separators=(",", ":"))))
            else:
                pairs.append((key, scalar_text(value)))
        return join_url(self._url, pairs)

    def headers(self, api_key: str) -> dict[str, str]:
        """Return the credential header. This API takes the key with no prefix."""
        return {"Authorization": api_key}

    def audio_message(self, pcm: bytes, sample_rate: int) -> str | bytes:
        """Return the chunk as it is: this API takes raw binary audio."""
        return pcm

    def closing_messages(self) -> list[str]:
        """Return the message that ends the session."""
        return [TERMINATE]

    def new_parser(self) -> SttParser:
        """Return a parser for a new session."""
        return AssemblyAiParser()
