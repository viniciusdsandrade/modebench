"""The ElevenLabs realtime dialect (Scribe realtime).

Reference: https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime

- The session is configured on the connect URL. A list value is a repeated
  parameter. `audio_format` names the encoding and the sample rate together.
- The key goes in the `xi-api-key` header.
- Audio is a JSON text message with the chunk in base64.
- `partial_transcript` is interim text. A commit settles a segment. One
  segment can be settled four times: a final transcript, its twin with
  timestamps, the commit, and the twin of the commit. Only the twins carry
  word times, and those times are in seconds.
- The protocol has no utterance number, so the parser counts the commits.
- No message ends a session. The session is billed for the audio that it got.
"""

import base64
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

from modebench.stt.base import (
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

DEFAULT_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

ERROR_TYPES = frozenset(
    {
        "error",
        "auth_error",
        "quota_exceeded",
        "commit_throttled",
        "unaccepted_terms",
        "rate_limited",
        "queue_overflow",
        "resource_exhausted",
        "session_time_limit_exceeded",
        "input_error",
        "invalid_request",
        "chunk_size_exceeded",
        "insufficient_audio_activity",
        "transcriber_error",
    }
)
_FINAL_TYPES = ("final_transcript", "final_transcript_with_timestamps")
_COMMIT_TYPES = ("committed_transcript", "committed_transcript_with_timestamps")


def _span(message: Mapping[str, Any]) -> tuple[float | None, float | None, str | None]:
    """Return the start, the end and the main speaker of the words of a message."""
    words = message.get("words")
    if not isinstance(words, list):
        return None, None, None
    start: float | None = None
    end: float | None = None
    speakers: Counter[str] = Counter()
    for word in words:
        if not isinstance(word, dict):
            continue
        word_start = number(word, "start")
        word_end = number(word, "end")
        if start is None and word_start is not None:
            start = word_start * 1000.0
        if word_end is not None:
            end = word_end * 1000.0
        speaker = word.get("speaker_id")
        if isinstance(speaker, str) and speaker:
            speakers[speaker] += 1
    main = speakers.most_common(1)[0][0] if speakers else None
    return start, end, main


class ElevenLabsParser:
    """Reads ElevenLabs messages and numbers the utterances.

    A commit closes a segment, so the first transcript after a commit opens
    the next utterance. A plain commit that comes when the segment is already
    committed belongs to a new segment. A twin with timestamps belongs to a
    new segment only if a twin was already seen for the open one.
    """

    def __init__(self) -> None:
        self._utterance = 0
        self._committed = False
        self._committed_timed = False

    def _open_utterance(self) -> int:
        if self._committed:
            self._committed = False
            self._committed_timed = False
            self._utterance += 1
        return self._utterance

    def parse(self, raw: str | bytes) -> list[SttEvent]:
        """Return the events of one server message."""
        message = decode_object(raw)
        if message is None:
            return [SttEvent(kind=OTHER, message="not a JSON object")]
        kind = message.get("message_type")
        text = message.get("text")
        if isinstance(text, str) and text:
            words = text
        else:
            word_list = message.get("words")
            if isinstance(word_list, list):
                words = " ".join(
                    str(w.get("text", "")) for w in word_list if isinstance(w, dict)
                ).strip()
            else:
                words = ""
        if kind == "session_started":
            return [SttEvent(kind=OPEN)]
        if kind == "partial_transcript":
            return [SttEvent(kind=PARTIAL, text=words, utterance=self._open_utterance())]
        if kind in _FINAL_TYPES:
            start, end, speaker = _span(message)
            return [SttEvent(FINAL, words, self._open_utterance(), start, end, speaker)]
        if kind in _COMMIT_TYPES:
            timed = kind == "committed_transcript_with_timestamps"
            closes_a_further_segment = self._committed_timed if timed else self._committed
            if closes_a_further_segment:
                self._open_utterance()
            self._committed = True
            self._committed_timed = self._committed_timed or timed
            start, end, speaker = _span(message)
            return [SttEvent(FINAL, words, self._utterance, start, end, speaker)]
        if isinstance(kind, str) and kind in ERROR_TYPES:
            return [SttEvent(kind=ERROR, message=f"{kind}: {message.get('error', '')}")]
        return [SttEvent(kind=OTHER, message=str(kind))]


class ElevenLabsAdapter:
    """Builds the requests of the ElevenLabs dialect."""

    name = "elevenlabs"

    def __init__(self, query: Mapping[str, Any], url: str = "") -> None:
        self._query = dict(query)
        self._url = url or DEFAULT_URL

    def connect_url(self, sample_rate: int) -> str:
        """Return the connect URL. A list of the config becomes a repeated parameter."""
        pairs = [("audio_format", f"pcm_{sample_rate}")]
        for key, value in self._query.items():
            if isinstance(value, list):
                pairs.extend((key, scalar_text(item)) for item in value)
            else:
                pairs.append((key, scalar_text(value)))
        return join_url(self._url, pairs)

    def headers(self, api_key: str) -> dict[str, str]:
        """Return the credential header."""
        return {"xi-api-key": api_key}

    def audio_message(self, pcm: bytes, sample_rate: int) -> str | bytes:
        """Return the chunk in the JSON envelope of the API. The provider decides the commits."""
        return json.dumps(
            {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(pcm).decode("ascii"),
                "commit": False,
                "sample_rate": sample_rate,
            },
            separators=(",", ":"),
        )

    def closing_messages(self) -> list[str]:
        """Return nothing: this API has no message that ends a session."""
        return []

    def new_parser(self) -> SttParser:
        """Return a parser for a new session."""
        return ElevenLabsParser()
