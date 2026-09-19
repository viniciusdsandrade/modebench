"""The adapter for each API that speaks the OpenAI chat completions protocol.

One adapter serves OpenRouter and the Google Gemini API. The differences
between them are data: the base URL, the key, and the raw parameters of the
mode. The adapter adds nothing to those parameters and removes nothing.

The clock is `perf_counter_ns`. It is read one time when the request starts
and one time for each block of bytes that the network delivers, so an event
has the time at which its bytes were in the process.
"""

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from modebench.config import Mode, ProviderConfig
from modebench.providers.base import ChatRequest, StreamEvent, StreamOutcome
from modebench.providers.sse import DONE, SseEvent, SseParser
from modebench.redact import redact

Clock = Callable[[], int]

_ERROR_BODY_LIMIT = 500
_CONNECT_TIMEOUT_S = 10.0


def build_body(provider: ProviderConfig, mode: Mode, request: ChatRequest) -> dict[str, Any]:
    """Return the JSON body: the provider defaults, then the raw mode parameters.

    The three keys that the adapter owns go in last, so a parameter of the
    mode cannot switch the stream off or change the conversation.
    """
    messages: list[dict[str, str]] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.user})
    body: dict[str, Any] = {}
    body.update(provider.extra_body)
    body.update(mode.params)
    body["model"] = mode.model
    body["messages"] = messages
    body["stream"] = True
    return body


@dataclass(slots=True)
class _StreamState:
    start_ns: int
    first_chunk_ns: int | None = None
    first_token_ns: int | None = None
    first_reasoning_ns: int | None = None
    first_content_ns: int | None = None
    answer_parts: list[str] = field(default_factory=list)
    reasoning_chars: int = 0
    usage: dict[str, Any] | None = None
    served_by: str | None = None
    malformed: int = 0
    stream_error: str | None = None
    done: bool = False
    events: list[StreamEvent] = field(default_factory=list)

    def offset_ms(self, at_ns: int | None) -> float | None:
        if at_ns is None:
            return None
        return (at_ns - self.start_ns) / 1e6


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _nested_int(usage: dict[str, Any], outer: str, inner: str) -> int | None:
    details = usage.get(outer)
    if isinstance(details, dict):
        return _int_or_none(details.get(inner))
    return None


def _error_text(error: object) -> str:
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        if code is not None:
            return f"{code}: {message}"
        return str(message)
    return str(error)


def _reasoning_delta(delta: dict[str, Any]) -> tuple[str, bool]:
    """Return the reasoning text of a delta, and whether the delta reasons at all.

    Encrypted reasoning has no text, but it shows that the model is at work,
    so it counts for the time to the first token.
    """
    for key in ("reasoning", "reasoning_content"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            return value, True
    details = delta.get("reasoning_details")
    if not isinstance(details, list):
        return "", False
    parts: list[str] = []
    active = False
    for item in details:
        if not isinstance(item, dict):
            continue
        for key in ("text", "summary"):
            value = item.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
                active = True
        if isinstance(item.get("data"), str) and item["data"]:
            active = True
    return "".join(parts), active


def _content_delta(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts = [
            item["text"]
            for item in value
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "".join(texts)
    return ""


def _consume(event: SseEvent, state: _StreamState) -> None:
    """Apply one event of the stream to the state."""
    if event.data.strip() == DONE:
        state.done = True
        return
    try:
        chunk = json.loads(event.data)
    except json.JSONDecodeError:
        state.malformed += 1
        return
    if not isinstance(chunk, dict):
        state.malformed += 1
        return
    if state.first_chunk_ns is None:
        state.first_chunk_ns = event.at_ns
    if chunk.get("error"):
        state.stream_error = _error_text(chunk["error"])
        return
    provider = chunk.get("provider")
    if state.served_by is None and isinstance(provider, str) and provider:
        state.served_by = provider
    usage = chunk.get("usage")
    if isinstance(usage, dict):
        state.usage = usage
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        delta = choice.get("delta") if isinstance(choice, dict) else None
        if not isinstance(delta, dict):
            continue
        _consume_delta(delta, event.at_ns, state)


def _consume_delta(delta: dict[str, Any], at_ns: int, state: _StreamState) -> None:
    offset = (at_ns - state.start_ns) / 1e6
    reasoning, reasons = _reasoning_delta(delta)
    if reasons:
        if state.first_token_ns is None:
            state.first_token_ns = at_ns
        if state.first_reasoning_ns is None:
            state.first_reasoning_ns = at_ns
        state.reasoning_chars += len(reasoning)
        state.events.append(StreamEvent(offset_ms=offset, kind="reasoning", chars=len(reasoning)))
    content = _content_delta(delta.get("content"))
    if content:
        state.answer_parts.append(content)
        if state.first_token_ns is None:
            state.first_token_ns = at_ns
        if state.first_content_ns is None and content.strip():
            state.first_content_ns = at_ns
        state.events.append(StreamEvent(offset_ms=offset, kind="content", chars=len(content)))


def _split_lines(buffer: bytearray) -> list[str]:
    """Remove each complete line from `buffer` and return the lines as text."""
    lines: list[str] = []
    while True:
        index = buffer.find(b"\n")
        if index < 0:
            return lines
        raw = bytes(buffer[:index])
        del buffer[: index + 1]
        lines.append(raw.rstrip(b"\r").decode("utf-8", errors="replace"))


def _timed_lines(blocks: Iterator[bytes], clock: Clock) -> Iterator[tuple[str, int]]:
    """Yield each line with the clock reading of the block of bytes that completed it."""
    buffer = bytearray()
    for block in blocks:
        now_ns = clock()
        buffer.extend(block)
        for line in _split_lines(buffer):
            yield line, now_ns
    if buffer:
        yield bytes(buffer).rstrip(b"\r").decode("utf-8", errors="replace"), clock()


class OpenAICompatProvider:
    """Streams a chat completion from one endpoint and measures it."""

    def __init__(
        self,
        config: ProviderConfig,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        clock: Clock = time.perf_counter_ns,
    ) -> None:
        self._config = config
        self._api_key = api_key
        self._client = client if client is not None else httpx.Client()
        self._owns_client = client is None
        self._clock = clock

    def close(self) -> None:
        """Close the HTTP client if this object made it."""
        if self._owns_client:
            self._client.close()

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "text/event-stream", **self._config.headers}
        headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def stream_chat(self, mode: Mode, request: ChatRequest, timeout_s: float) -> StreamOutcome:
        """Send one request and return what the stream did, with its times."""
        url = self._config.base_url.rstrip("/") + "/chat/completions"
        body = build_body(self._config, mode, request)
        timeout = httpx.Timeout(timeout_s, connect=min(_CONNECT_TIMEOUT_S, timeout_s))
        deadline_ns = int(timeout_s * 1e9)
        parser = SseParser()
        state = _StreamState(start_ns=self._clock())
        error_kind: str | None = None
        error_message: str | None = None
        http_status: int | None = None
        try:
            with self._client.stream(
                "POST", url, json=body, headers=self._headers(), timeout=timeout
            ) as response:
                http_status = response.status_code
                if http_status != 200:
                    text = response.read().decode("utf-8", errors="replace")
                    error_kind = "http_error"
                    error_message = text[:_ERROR_BODY_LIMIT]
                else:
                    for line, now_ns in _timed_lines(response.iter_bytes(), self._clock):
                        if now_ns - state.start_ns > deadline_ns:
                            error_kind = "timeout"
                            error_message = f"no complete answer in {timeout_s:g} s"
                            break
                        event = parser.feed(line, now_ns)
                        if event is not None:
                            _consume(event, state)
                        if state.done or state.stream_error is not None:
                            break
                    else:
                        last = parser.flush()
                        if last is not None:
                            _consume(last, state)
        except httpx.TimeoutException as exc:
            error_kind = "timeout"
            error_message = f"{type(exc).__name__} after {timeout_s:g} s"
        except httpx.HTTPError as exc:
            error_kind = "transport_error"
            error_message = f"{type(exc).__name__}: {exc}"
        end_ns = self._clock()
        if error_kind is None and state.stream_error is not None:
            error_kind = "stream_error"
            error_message = state.stream_error
        answer = "".join(state.answer_parts)
        if error_kind is None and not answer.strip():
            error_kind = "empty_output"
            error_message = "the stream ended with no visible content"
        return self._outcome(state, end_ns, error_kind, error_message, http_status, answer, parser)

    def _outcome(
        self,
        state: _StreamState,
        end_ns: int,
        error_kind: str | None,
        error_message: str | None,
        http_status: int | None,
        answer: str,
        parser: SseParser,
    ) -> StreamOutcome:
        usage = state.usage or {}
        if error_message is not None:
            error_message = redact(error_message, [self._api_key])
        if parser.comments:
            state.events.append(StreamEvent(offset_ms=0.0, kind="comments", chars=parser.comments))
        return StreamOutcome(
            ok=error_kind is None,
            total_ms=(end_ns - state.start_ns) / 1e6,
            error_kind=error_kind,
            error_message=error_message,
            http_status=http_status,
            first_chunk_ms=state.offset_ms(state.first_chunk_ns),
            ttft_ms=state.offset_ms(state.first_token_ns),
            first_reasoning_ms=state.offset_ms(state.first_reasoning_ns),
            ttfat_ms=state.offset_ms(state.first_content_ns),
            answer=answer,
            reasoning_chars=state.reasoning_chars,
            prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
            completion_tokens=_int_or_none(usage.get("completion_tokens")),
            reasoning_tokens=_nested_int(usage, "completion_tokens_details", "reasoning_tokens"),
            cached_tokens=_nested_int(usage, "prompt_tokens_details", "cached_tokens"),
            total_tokens=_int_or_none(usage.get("total_tokens")),
            reported_cost_usd=_float_or_none(usage.get("cost")),
            served_by=state.served_by,
            malformed_chunks=state.malformed,
            events=state.events,
        )
