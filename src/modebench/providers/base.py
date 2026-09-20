"""What every chat provider receives and what it gives back."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from modebench.config import Mode


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """One Analyze request.

    `metadata` is for the fake provider and for the records of the run. A
    provider that talks to a network must not send it: it holds the expected
    answer, and the mode under test must not see that.
    """

    system: str
    user: str
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One step of the stream: when it came, what it carried and how much."""

    offset_ms: float
    kind: str
    chars: int


@dataclass(slots=True)
class StreamOutcome:
    """The measured result of one request. A failure is an outcome, not an exception.

    `new_connection` is True if the request had to open a connection, and
    `connect_ms` is then the time that the TCP and TLS handshakes took. That
    time is a part of each latency of the request. `finish_reason` is what the
    API gave as the reason for the end of the answer: `length` is an answer
    that the token limit cut.
    """

    ok: bool
    total_ms: float
    error_kind: str | None = None
    error_message: str | None = None
    http_status: int | None = None
    first_chunk_ms: float | None = None
    ttft_ms: float | None = None
    first_reasoning_ms: float | None = None
    ttfat_ms: float | None = None
    answer: str = ""
    reasoning_chars: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None
    reported_cost_usd: float | None = None
    served_by: str | None = None
    finish_reason: str | None = None
    new_connection: bool = False
    connect_ms: float | None = None
    malformed_chunks: int = 0
    events: list[StreamEvent] = field(default_factory=list)


class ChatProvider(Protocol):
    """A provider streams one chat completion and measures it."""

    def stream_chat(self, mode: Mode, request: ChatRequest, timeout_s: float) -> StreamOutcome:
        """Send the request with the raw parameters of `mode` and return the outcome."""
        ...
