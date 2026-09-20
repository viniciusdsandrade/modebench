"""The OpenAI-compatible adapter: timing, usage, failures and the request body.

The streaming tests use the mock transport of httpx, which delivers the byte
blocks one at a time. The clock moves one millisecond each time it is read:
one time at the start, one time for each block, and one time at the end.
"""

import json
from collections.abc import Callable, Iterator

import httpx
import respx

from helpers import TickClock
from modebench.config import Mode, ProviderConfig
from modebench.providers.base import ChatRequest
from modebench.providers.openai_compat import OpenAICompatProvider, build_body

BASE_URL = "https://openrouter.test/api/v1"
URL = f"{BASE_URL}/chat/completions"
API_KEY = "sk-test-secret-123456"

PROVIDER = ProviderConfig(
    kind="openai_compat",
    base_url=BASE_URL,
    api_key_env="OPENROUTER_API_KEY",
    privacy="openrouter",
    cost_source="usage",
    headers={"X-Title": "modebench"},
    extra_body={"stream_options": {"include_usage": True}},
)
MODE = Mode(
    id="gemini-low",
    provider="openrouter",
    model="google/gemini-test",
    params={
        "reasoning": {"effort": "low"},
        "provider": {"order": ["google-ai-studio"], "data_collection": "deny"},
    },
)
REQUEST = ChatRequest(
    system="Infer the question.",
    user="=== NEW ===\nCHAMADA: qual é o prazo",
    metadata={"gold_question": "SECRET-GOLD-QUESTION"},
)


def event(payload: dict[str, object]) -> bytes:
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n"


def delta(**fields: object) -> bytes:
    return event({"choices": [{"delta": fields}]})


def provider_for(
    blocks: list[bytes], clock: Callable[[], int] | None = None
) -> tuple[OpenAICompatProvider, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def stream() -> Iterator[bytes]:
        yield from blocks

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=stream())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatProvider(PROVIDER, API_KEY, client=client, clock=clock or TickClock())
    return provider, seen


def test_reasoning_before_content_gives_separate_ttft_and_ttfat() -> None:
    usage = {
        "prompt_tokens": 1200,
        "completion_tokens": 310,
        "total_tokens": 1510,
        "cost": 0.00123,
        "completion_tokens_details": {"reasoning_tokens": 250},
        "prompt_tokens_details": {"cached_tokens": 1000},
    }
    blocks = [
        b": OPENROUTER PROCESSING\n\n",
        event(
            {
                "id": "gen-1",
                "provider": "Google AI Studio",
                "choices": [{"delta": {"role": "assistant", "content": ""}}],
            }
        ),
        delta(
            reasoning="Pensando", reasoning_details=[{"type": "reasoning.text", "text": "Pensando"}]
        ),
        delta(reasoning=" mais"),
        delta(content="\n"),
        delta(content="Pergunta: qual é o prazo?"),
        delta(content="\nResposta: seis semanas."),
        event({"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}], "usage": usage}),
        b"data: [DONE]\n\n",
    ]
    provider, _ = provider_for(blocks)
    outcome = provider.stream_chat(MODE, REQUEST, 60.0)

    assert outcome.ok
    assert outcome.first_chunk_ms == 2.0
    assert outcome.ttft_ms == 3.0
    assert outcome.first_reasoning_ms == 3.0
    assert outcome.ttfat_ms == 6.0
    assert outcome.total_ms == 10.0
    assert outcome.answer == "\nPergunta: qual é o prazo?\nResposta: seis semanas."
    assert outcome.reasoning_chars == len("Pensando") + len(" mais")
    assert outcome.prompt_tokens == 1200
    assert outcome.completion_tokens == 310
    assert outcome.reasoning_tokens == 250
    assert outcome.cached_tokens == 1000
    assert outcome.total_tokens == 1510
    assert outcome.reported_cost_usd == 0.00123
    assert outcome.served_by == "Google AI Studio"
    kinds = [item.kind for item in outcome.events]
    assert kinds == ["reasoning", "reasoning", "content", "content", "content", "comments"]


def test_content_with_no_reasoning_has_equal_ttft_and_ttfat() -> None:
    provider, _ = provider_for([delta(content="Olá"), b"data: [DONE]\n\n"])
    outcome = provider.stream_chat(MODE, REQUEST, 60.0)
    assert outcome.ok
    assert outcome.ttft_ms == outcome.ttfat_ms == 1.0
    assert outcome.first_reasoning_ms is None
    assert outcome.reasoning_tokens is None


def test_reasoning_content_field_and_encrypted_details_count_as_reasoning() -> None:
    blocks = [
        delta(reasoning_details=[{"type": "reasoning.encrypted", "data": "QUJD"}]),
        delta(reasoning_content="raciocínio"),
        delta(content=[{"type": "text", "text": "Resposta"}]),
    ]
    provider, _ = provider_for(blocks)
    outcome = provider.stream_chat(MODE, REQUEST, 60.0)
    assert outcome.ok
    assert outcome.ttft_ms == 1.0
    assert outcome.ttfat_ms == 3.0
    assert outcome.reasoning_chars == len("raciocínio")
    assert outcome.answer == "Resposta"


def test_an_event_cut_in_two_blocks_gets_the_time_of_the_block_that_completes_it() -> None:
    whole = delta(content="Oi")
    blocks = [whole[:20], whole[20:] + b"data: [DONE]\n\n"]
    provider, _ = provider_for(blocks)
    outcome = provider.stream_chat(MODE, REQUEST, 60.0)
    assert outcome.ok
    assert outcome.answer == "Oi"
    assert outcome.ttfat_ms == 2.0


def test_a_stream_that_ends_with_no_blank_line_is_still_read() -> None:
    line = b'data: {"choices":[{"delta":{"content":"Fim"}}]}'
    provider, _ = provider_for([line])
    outcome = provider.stream_chat(MODE, REQUEST, 60.0)
    assert outcome.ok
    assert outcome.answer == "Fim"


def test_malformed_chunks_are_counted_and_do_not_stop_the_stream() -> None:
    blocks = [
        b"data: {not json}\n\n",
        b"data: [1, 2]\n\n",
        delta(content="ok"),
        b"data: [DONE]\n\n",
    ]
    provider, _ = provider_for(blocks)
    outcome = provider.stream_chat(MODE, REQUEST, 60.0)
    assert outcome.ok
    assert outcome.malformed_chunks == 2


def test_an_error_in_the_stream_is_a_failure_with_its_message() -> None:
    blocks = [
        delta(content="Perg"),
        event({"error": {"code": 502, "message": "Provider disconnected"}, "choices": []}),
        delta(content="never read"),
    ]
    provider, _ = provider_for(blocks)
    outcome = provider.stream_chat(MODE, REQUEST, 60.0)
    assert not outcome.ok
    assert outcome.error_kind == "stream_error"
    assert outcome.error_message == "502: Provider disconnected"
    assert outcome.answer == "Perg"


def test_a_stream_with_no_visible_content_is_a_failure() -> None:
    blocks = [delta(role="assistant", content=""), delta(content="  \n"), b"data: [DONE]\n\n"]
    provider, _ = provider_for(blocks)
    outcome = provider.stream_chat(MODE, REQUEST, 60.0)
    assert not outcome.ok
    assert outcome.error_kind == "empty_output"
    assert outcome.ttfat_ms is None


def test_the_deadline_stops_a_stream_that_is_too_slow() -> None:
    blocks = [delta(reasoning="a"), delta(reasoning="b"), delta(content="late")]
    provider, _ = provider_for(blocks, clock=TickClock(step_ms=40_000))
    outcome = provider.stream_chat(MODE, REQUEST, 60.0)
    assert not outcome.ok
    assert outcome.error_kind == "timeout"
    assert outcome.answer == ""


def test_the_body_has_the_raw_parameters_and_never_the_metadata() -> None:
    provider, seen = provider_for([delta(content="ok")])
    provider.stream_chat(MODE, REQUEST, 60.0)
    request = seen[0]
    body = json.loads(request.content)
    assert body["model"] == "google/gemini-test"
    assert body["stream"] is True
    assert body["reasoning"] == {"effort": "low"}
    assert body["provider"] == {"order": ["google-ai-studio"], "data_collection": "deny"}
    assert body["stream_options"] == {"include_usage": True}
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert "SECRET-GOLD-QUESTION" not in request.content.decode("utf-8")
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert request.headers["X-Title"] == "modebench"
    assert str(request.url) == URL


def test_mode_parameters_cannot_change_the_keys_that_the_adapter_owns() -> None:
    mode = Mode(
        id="m", provider="openrouter", model="real/model", params={"stream": False, "model": "x"}
    )
    body = build_body(PROVIDER, mode, ChatRequest(system="", user="hello"))
    assert body["stream"] is True
    assert body["model"] == "real/model"
    assert body["messages"] == [{"role": "user", "content": "hello"}]


@respx.mock
def test_an_http_error_is_a_failure_and_the_key_is_redacted() -> None:
    respx.post(URL).mock(
        return_value=httpx.Response(
            429, json={"error": {"code": 429, "message": f"rate limited for {API_KEY}"}}
        )
    )
    provider = OpenAICompatProvider(PROVIDER, API_KEY, clock=TickClock())
    outcome = provider.stream_chat(MODE, REQUEST, 60.0)
    provider.close()
    assert not outcome.ok
    assert outcome.error_kind == "http_error"
    assert outcome.http_status == 429
    assert outcome.error_message is not None
    assert "rate limited" in outcome.error_message
    assert API_KEY not in outcome.error_message


@respx.mock
def test_a_read_timeout_is_a_timeout_failure() -> None:
    respx.post(URL).mock(side_effect=httpx.ReadTimeout("no bytes"))
    provider = OpenAICompatProvider(PROVIDER, API_KEY, clock=TickClock())
    outcome = provider.stream_chat(MODE, REQUEST, 60.0)
    provider.close()
    assert not outcome.ok
    assert outcome.error_kind == "timeout"


@respx.mock
def test_a_connection_error_is_a_transport_failure() -> None:
    respx.post(URL).mock(side_effect=httpx.ConnectError("refused"))
    provider = OpenAICompatProvider(PROVIDER, API_KEY, clock=TickClock())
    outcome = provider.stream_chat(MODE, REQUEST, 60.0)
    provider.close()
    assert not outcome.ok
    assert outcome.error_kind == "transport_error"
    assert outcome.error_message is not None
    assert "ConnectError" in outcome.error_message
