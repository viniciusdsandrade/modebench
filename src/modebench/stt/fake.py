"""A fake speech to text provider with no network, for dry runs and for tests.

The fake server follows the position of the audio that it receives. It sends
a partial a fixed time after an utterance starts and a final a fixed time
after it ends, with the words of the reference. With the virtual clock of the
tests, the latencies that the replay measures are then known in advance.

A real provider counts its word times from the first byte of the stream, so
the silence that goes out before the audio moves each time. The fake server
does the same when it knows the length of that silence.
"""

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from modebench.stt.audio import BYTES_PER_SAMPLE
from modebench.stt.base import (
    CLOSED,
    FINAL,
    OPEN,
    OTHER,
    PARTIAL,
    SttEvent,
    SttParser,
    decode_object,
    number,
)
from modebench.stt.config import AudioItem, ReferenceUtterance
from modebench.stt.replay import ConnectionEnded, Connector, WsConnection

END_MESSAGE = '{"type":"end"}'
_KINDS = {"open": OPEN, "partial": PARTIAL, "final": FINAL, "closed": CLOSED}


class FakeSttParser:
    """Reads the messages of the fake server."""

    def parse(self, raw: str | bytes) -> list[SttEvent]:
        """Return the event of one message."""
        message = decode_object(raw)
        if message is None:
            return [SttEvent(kind=OTHER, message="not a JSON object")]
        kind = _KINDS.get(str(message.get("type")), OTHER)
        utterance = message.get("utterance")
        speaker = message.get("speaker")
        return [
            SttEvent(
                kind=kind,
                text=str(message.get("text", "")),
                utterance=utterance if isinstance(utterance, int) else None,
                audio_start_ms=number(message, "start_ms"),
                audio_end_ms=number(message, "end_ms"),
                speaker=speaker if isinstance(speaker, str) and speaker else None,
            )
        ]


class FakeSttAdapter:
    """The dialect of the fake server. It keeps the name of the provider that it stands for."""

    def __init__(self, name: str = "fake") -> None:
        self.name = name

    def connect_url(self, sample_rate: int) -> str:
        """Return a URL that no network can open."""
        return f"fake://stt?sample_rate={sample_rate}"

    def headers(self, api_key: str) -> dict[str, str]:
        """Return no header: the fake server has no credential."""
        return {}

    def audio_message(self, pcm: bytes, sample_rate: int) -> str | bytes:
        """Return the chunk as it is."""
        return pcm

    def closing_messages(self) -> list[str]:
        """Return the message that ends the fake session."""
        return [END_MESSAGE]

    def new_parser(self) -> SttParser:
        """Return a parser for a new session."""
        return FakeSttParser()


def _corrupt(text: str, every: int) -> str:
    if every <= 0:
        return text
    words = text.split()
    return " ".join("xx" if (index + 1) % every == 0 else word for index, word in enumerate(words))


class FakeSttServer:
    """A connection that answers as a provider does, from the reference of the audio."""

    def __init__(
        self,
        audio: AudioItem,
        sample_rate: int,
        *,
        partial_delay_ms: float = 400.0,
        final_delay_ms: float = 700.0,
        word_error_every: int = 0,
        preroll_ms: float = 0.0,
    ) -> None:
        self._utterances = sorted(audio.utterances, key=lambda item: item.start_ms)
        self._rate = sample_rate
        self._preroll_ms = preroll_ms
        self._partial_delay_ms = partial_delay_ms
        self._final_delay_ms = final_delay_ms
        self._word_error_every = word_error_every
        self._position_ms = 0.0
        self._partials: set[int] = set()
        self._finals: set[int] = set()
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._queue.put_nowait(json.dumps({"type": "open"}))

    def _message(self, kind: str, index: int, utterance: ReferenceUtterance) -> str:
        text = _corrupt(utterance.text, self._word_error_every)
        if kind == "partial":
            text = " ".join(text.split()[:2])
        payload: dict[str, Any] = {
            "type": kind,
            "utterance": index,
            "text": text,
            "start_ms": utterance.start_ms + self._preroll_ms,
            "end_ms": utterance.end_ms + self._preroll_ms,
            "speaker": utterance.speaker,
        }
        return json.dumps(payload, ensure_ascii=False)

    def _emit_due(self, flush: bool = False) -> None:
        for index, utterance in enumerate(self._utterances):
            partial_at = self._preroll_ms + utterance.start_ms + self._partial_delay_ms
            final_at = self._preroll_ms + utterance.end_ms + self._final_delay_ms
            if index not in self._partials and (flush or self._position_ms >= partial_at):
                self._partials.add(index)
                self._queue.put_nowait(self._message("partial", index, utterance))
            if index not in self._finals and (flush or self._position_ms >= final_at):
                self._finals.add(index)
                self._queue.put_nowait(self._message("final", index, utterance))

    async def send(self, message: str | bytes) -> None:
        """Take one chunk of audio, or the message that ends the session."""
        if isinstance(message, bytes):
            self._emit_due()
            self._position_ms += len(message) / BYTES_PER_SAMPLE / self._rate * 1000.0
        else:
            self._emit_due(flush=True)
            self._queue.put_nowait(json.dumps({"type": "closed"}))
        await asyncio.sleep(0)

    async def recv(self) -> str | bytes:
        """Return the next message of the server."""
        item = await self._queue.get()
        if item is None:
            raise ConnectionEnded("the fake server closed the connection")
        return item

    async def close(self) -> None:
        """End the connection: the next `recv` raises ConnectionEnded."""
        self._queue.put_nowait(None)


def fake_connector(
    audio: AudioItem, sample_rate: int, settings: Mapping[str, Any], *, preroll_ms: float = 0.0
) -> Connector:
    """Return a connector that opens a fake server for one audio item."""

    def _value(key: str, default: float) -> float:
        value = settings.get(key, default)
        return float(value) if isinstance(value, int | float) else default

    async def connect(url: str, headers: Mapping[str, str], timeout_s: float) -> WsConnection:
        return FakeSttServer(
            audio,
            sample_rate,
            partial_delay_ms=_value("partial_delay_ms", 400.0),
            final_delay_ms=_value("final_delay_ms", 700.0),
            word_error_every=int(_value("word_error_every", 0.0)),
            preroll_ms=preroll_ms,
        )

    return connect
