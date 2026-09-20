"""Replays audio to a provider at the speed of real time and records what comes back.

The sender puts one chunk of 50 ms on the socket each 50 ms, by an absolute
schedule, so that a slow step does not move the steps after it. The receiver
is a second task that reads the clock when a message arrives. The zero of each
time is the moment at which the first chunk went out, so a time in the trace
can be compared with a position in the audio.

The clock and the sleep are parameters. The tests give a virtual clock, and a
replay of one minute of audio then takes a few milliseconds.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from modebench.redact import redact
from modebench.stt.audio import BYTES_PER_SAMPLE, PcmAudio, chunk_bytes, silence
from modebench.stt.base import CLOSED, ERROR, SttAdapter, SttEvent
from modebench.stt.config import SttReplayConfig

MAX_BURST_MS = 1000
EARLY_END = "the provider ended the session before the audio ended"


class ConnectionEnded(Exception):
    """The other side closed the connection."""


class WsConnection(Protocol):
    """The part of a websocket connection that the replay uses."""

    async def send(self, message: str | bytes) -> None:
        """Send one message."""
        ...

    async def recv(self) -> str | bytes:
        """Return the next message, or raise ConnectionEnded."""
        ...

    async def close(self) -> None:
        """Close the connection."""
        ...


Connector = Callable[[str, Mapping[str, str], float], Awaitable[WsConnection]]
Clock = Callable[[], int]
Sleep = Callable[[float], Awaitable[None]]


class VirtualTime:
    """A clock that moves only when the replay sleeps.

    With a fake server, a replay on this clock takes no real time, and each
    time that it measures is exact. A sleep first gives the receiver its turn
    and then moves the clock, so a message that the server sent at a moment is
    received at that same moment.
    """

    def __init__(self) -> None:
        self._now_ns = 0

    def clock(self) -> int:
        """Return the virtual time in nanoseconds."""
        return self._now_ns

    async def sleep(self, seconds: float) -> None:
        """Let the other tasks run, and then move the clock forward."""
        await asyncio.sleep(0)
        self._now_ns += round(seconds * 1e9)


@dataclass(frozen=True, slots=True)
class ReceivedEvent:
    """An event and the time at which it arrived, from the first chunk of audio."""

    at_ms: float
    event: SttEvent


@dataclass(slots=True)
class SessionTrace:
    """What one replay recorded.

    `audio_ms` is the length of the audio file. `sent_ms` is the audio that
    went out, with the silence before and after it, and it is shorter than the
    plan when the session ended early.
    """

    provider: str
    audio_ms: float
    preroll_ms: float = 0.0
    connect_ms: float | None = None
    session_ms: float = 0.0
    sent_ms: float = 0.0
    max_send_lag_ms: float = 0.0
    events: list[ReceivedEvent] = field(default_factory=list)
    error: str | None = None


@dataclass(slots=True)
class _Reception:
    """What the receiver task shares with the sender."""

    received: list[tuple[int, SttEvent]] = field(default_factory=list)
    ended: asyncio.Event = field(default_factory=asyncio.Event)
    close_reason: str | None = None


def _duration_ms(pcm: bytes, sample_rate: int) -> float:
    return len(pcm) / BYTES_PER_SAMPLE / sample_rate * 1000.0


async def _receive(
    connection: WsConnection, adapter: SttAdapter, clock: Clock, reception: _Reception
) -> None:
    parser = adapter.new_parser()
    try:
        while True:
            raw = await connection.recv()
            now_ns = clock()
            try:
                events = parser.parse(raw)
            except Exception as exc:
                # A defect of one dialect is the error of one session, not the end of the suite.
                detail = f"the parser failed: {type(exc).__name__}: {exc}"
                events = [SttEvent(kind=ERROR, message=detail)]
            for event in events:
                reception.received.append((now_ns, event))
                if event.kind in (CLOSED, ERROR):
                    reception.ended.set()
    except ConnectionEnded as exc:
        reception.close_reason = str(exc) or None
    except Exception as exc:
        detail = f"the receiver failed: {type(exc).__name__}: {exc}"
        reception.received.append((clock(), SttEvent(kind=ERROR, message=detail)))
    finally:
        reception.ended.set()


async def replay_session(
    adapter: SttAdapter,
    audio: PcmAudio,
    connector: Connector,
    api_key: str,
    options: SttReplayConfig,
    *,
    preroll_ms: int = 0,
    clock: Clock = time.perf_counter_ns,
    sleep: Sleep = asyncio.sleep,
) -> SessionTrace:
    """Send `audio` at the speed of real time and return the trace of the session.

    A session that ends before each chunk went out is a failure, also when
    the receiver sees the end first. Without that rule, a provider that drops
    the connection in the middle of the audio gives a short transcript that
    looks like a bad recognition and not like an error.
    """
    trace = SessionTrace(
        provider=adapter.name, audio_ms=audio.duration_ms, preroll_ms=float(preroll_ms)
    )
    rate = audio.sample_rate
    connect_start_ns = clock()
    try:
        connection = await connector(
            adapter.connect_url(rate), adapter.headers(api_key), options.connect_timeout_s
        )
    except (OSError, TimeoutError, ConnectionEnded) as exc:
        trace.error = redact(f"connect failed: {type(exc).__name__}: {exc}", [api_key])
        return trace
    open_ns = clock()
    trace.connect_ms = (open_ns - connect_start_ns) / 1e6
    reception = _Reception()
    receiver = asyncio.create_task(_receive(connection, adapter, clock, reception))
    zero_ns = open_ns
    audio_complete = False
    send_failure: str | None = None
    try:
        if preroll_ms > 0:
            for burst in chunk_bytes(silence(rate, preroll_ms), rate, MAX_BURST_MS):
                await connection.send(adapter.audio_message(burst, rate))
                trace.sent_ms += _duration_ms(burst, rate)
        chunks = chunk_bytes(audio.samples, rate, options.chunk_ms)
        chunks += chunk_bytes(silence(rate, options.tail_silence_ms), rate, options.chunk_ms)
        step_ns = options.chunk_ms * 1_000_000
        zero_ns = clock()
        for index, chunk in enumerate(chunks):
            if reception.ended.is_set():
                break
            target_ns = zero_ns + index * step_ns
            wait_ns = target_ns - clock()
            if wait_ns > 0:
                await sleep(wait_ns / 1e9)
            lag_ms = (clock() - target_ns) / 1e6
            trace.max_send_lag_ms = max(trace.max_send_lag_ms, lag_ms)
            await connection.send(adapter.audio_message(chunk, rate))
            trace.sent_ms += _duration_ms(chunk, rate)
        else:
            audio_complete = True
        if audio_complete:
            closing = adapter.closing_messages()
            for message in closing:
                await connection.send(message)
            if closing:
                try:
                    await asyncio.wait_for(reception.ended.wait(), options.drain_timeout_s)
                except TimeoutError:
                    trace.error = "the provider did not confirm the end of the session"
            elif options.close_grace_s > 0:
                await sleep(options.close_grace_s)
    except ConnectionEnded as exc:
        send_failure = str(exc) or None
        if audio_complete and not reception.ended.is_set():
            trace.error = "the provider closed the connection before the session ended"
    finally:
        await connection.close()
        try:
            await asyncio.wait_for(receiver, options.drain_timeout_s)
        except TimeoutError:
            receiver.cancel()
    trace.session_ms = (clock() - open_ns) / 1e6
    trace.events = [
        ReceivedEvent(at_ms=(at_ns - zero_ns) / 1e6, event=event)
        for at_ns, event in reception.received
    ]
    errors = [item.event.message for item in trace.events if item.event.kind == ERROR]
    if errors and trace.error is None:
        trace.error = redact(errors[0], [api_key])
    if not audio_complete and trace.error is None:
        reason = reception.close_reason or send_failure
        detail = f"{EARLY_END}: {reason}" if reason else EARLY_END
        trace.error = redact(detail, [api_key])
    return trace
