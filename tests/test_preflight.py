"""The preflight: the model list of an endpoint, its pages, its prices and its errors."""

from collections.abc import Callable

import httpx
import pytest

from helpers import fake_mode, local_modes_file, openrouter_mode
from modebench.config import Mode, ProviderConfig
from modebench.errors import PreflightError
from modebench.runner.preflight import (
    catalogue_price,
    fetch_catalogue,
    normalize_model_id,
    run_preflight,
)

API_KEY = "sk-or-secret-123456"
ENV = {"OPENROUTER_API_KEY": API_KEY, "GEMINI_API_KEY": "gm-secret-123456"}
PROVIDER = ProviderConfig(
    kind="openai_compat",
    base_url="https://openrouter.test/api/v1/",
    api_key_env="OPENROUTER_API_KEY",
    headers={"X-Title": "modebench", "Authorization": "Bearer from-the-config"},
)


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_the_catalogue_follows_the_pages_and_keeps_the_credential() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.params.get("page") == "2":
            return httpx.Response(200, json={"data": [{"id": "models/gemini-test"}, "junk"]})
        first = {"data": [{"id": "openai/gpt-test"}], "links": {"next": "/api/v1/models?page=2"}}
        return httpx.Response(200, json=first)

    with client_for(handler) as client:
        catalogue = fetch_catalogue(client, PROVIDER, API_KEY)
    assert set(catalogue) == {"openai/gpt-test", "gemini-test"}
    assert [str(request.url) for request in seen] == [
        "https://openrouter.test/api/v1/models",
        "https://openrouter.test/api/v1/models?page=2",
    ]
    # A header of the config cannot replace the credential.
    assert all(request.headers["Authorization"] == f"Bearer {API_KEY}" for request in seen)
    assert seen[0].headers["X-Title"] == "modebench"


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(401, text=f"bad key {API_KEY}"), "answered HTTP 401"),
        (httpx.Response(200, text="<html>not json</html>"), "is not JSON"),
        (httpx.Response(200, json={"models": []}), "has no data array"),
        (httpx.Response(200, json=["a list"]), "has no data array"),
    ],
)
def test_a_model_list_that_cannot_be_read_is_a_preflight_error(
    response: httpx.Response, message: str
) -> None:
    with client_for(lambda request: response) as client:
        with pytest.raises(PreflightError, match=message) as raised:
            fetch_catalogue(client, PROVIDER, API_KEY)
    assert API_KEY not in str(raised.value)


def test_a_network_failure_is_a_preflight_error_with_no_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused for {API_KEY}")

    with client_for(handler) as client:
        with pytest.raises(PreflightError, match="cannot read the model list") as raised:
            fetch_catalogue(client, PROVIDER, API_KEY)
    assert API_KEY not in str(raised.value)


def test_catalogue_prices_are_for_one_token() -> None:
    price = catalogue_price(
        {"pricing": {"prompt": "0.000001", "completion": "0.000004", "input_cache_read": "1e-7"}}
    )
    assert price is not None
    assert price.usd_per_mtok_in == pytest.approx(1.0)
    assert price.usd_per_mtok_out == pytest.approx(4.0)
    assert price.usd_per_mtok_cached_in == pytest.approx(0.1)
    assert catalogue_price({"pricing": {"prompt": "0.000001", "completion": "free"}}) is None
    assert catalogue_price({"pricing": {"prompt": "-1", "completion": "0"}}) is None
    assert catalogue_price({"id": "no pricing"}) is None
    assert normalize_model_id("models/gemini-test") == "gemini-test"


def test_the_preflight_lists_each_problem_in_one_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        entry = {"id": "google/gemini-test", "pricing": {"prompt": "0.000001", "completion": "0"}}
        return httpx.Response(200, json={"data": [entry]})

    known = openrouter_mode("known")
    routed = openrouter_mode("routed", model="google/gemini-test:nitro")
    modes_file = local_modes_file(known, routed)
    with client_for(handler) as client:
        report = run_preflight(modes_file, [known, routed, fake_mode("local-mode")], ENV, client)
    assert report.matched == {"known": "google/gemini-test", "routed": "google/gemini-test"}
    assert report.prices["known"].usd_per_mtok_in == pytest.approx(1.0)

    wrong = openrouter_mode("wrong", model="google/absent")
    no_key = Mode(id="no-key", provider="google", model="gemini-test")
    stranger = Mode(id="stranger", provider="elsewhere", model="x")
    with client_for(handler) as client:
        with pytest.raises(PreflightError) as raised:
            run_preflight(
                modes_file, [wrong, no_key, stranger], {"OPENROUTER_API_KEY": API_KEY}, client
            )
    text = str(raised.value)
    assert "wrong: the model google/absent is not in openrouter" in text
    assert "no-key: the key GEMINI_API_KEY is not set" in text
    assert "stranger: the provider elsewhere is not in the modes file" in text
