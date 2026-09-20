"""The preflight: each model identifier must be in the catalogue of its provider.

A wrong identifier costs a full run of errors. The preflight reads the model
list of each endpoint one time (`GET {base_url}/models`, which is
`/api/v1/models` on OpenRouter) and stops the run if a mode is not there. The
catalogue also gives the token prices of the modes that have no price table.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from modebench.config import Mode, ModesFile, Price, ProviderConfig
from modebench.errors import PreflightError
from modebench.redact import redact

MAX_PAGES = 20
# Routing shortcuts of OpenRouter. They are not catalogue entries of their own.
ROUTING_SUFFIXES = (":nitro", ":floor", ":online", ":exacto")


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """What the preflight found: the catalogue entry and the price of each mode."""

    matched: dict[str, str] = field(default_factory=dict)
    prices: dict[str, Price] = field(default_factory=dict)


def normalize_model_id(model_id: str) -> str:
    """Return the identifier without the `models/` prefix of the Google list."""
    return model_id.removeprefix("models/")


def _catalogue_key(model: str, catalogue: Mapping[str, dict[str, Any]]) -> str | None:
    wanted = normalize_model_id(model)
    if wanted in catalogue:
        return wanted
    for suffix in ROUTING_SUFFIXES:
        if wanted.endswith(suffix) and wanted.removesuffix(suffix) in catalogue:
            return wanted.removesuffix(suffix)
    return None


def _per_mtok(value: object) -> float | None:
    try:
        number = float(str(value))
    except ValueError:
        return None
    return number * 1_000_000 if number >= 0 else None


def catalogue_price(entry: Mapping[str, Any]) -> Price | None:
    """Return the price of a catalogue entry. OpenRouter gives dollars for one token."""
    pricing = entry.get("pricing")
    if not isinstance(pricing, dict):
        return None
    prompt = _per_mtok(pricing.get("prompt"))
    completion = _per_mtok(pricing.get("completion"))
    if prompt is None or completion is None:
        return None
    cached = _per_mtok(pricing.get("input_cache_read")) if "input_cache_read" in pricing else None
    return Price(usd_per_mtok_in=prompt, usd_per_mtok_out=completion, usd_per_mtok_cached_in=cached)


def fetch_catalogue(
    client: httpx.Client, provider: ProviderConfig, api_key: str
) -> dict[str, dict[str, Any]]:
    """Return the model list of one endpoint: identifier, then entry."""
    url: str | None = provider.base_url.rstrip("/") + provider.models_path
    headers = {"Authorization": f"Bearer {api_key}", **provider.headers}
    catalogue: dict[str, dict[str, Any]] = {}
    pages = 0
    while url is not None and pages < MAX_PAGES:
        pages += 1
        try:
            response = client.get(url, headers=headers, timeout=30.0)
        except httpx.HTTPError as exc:
            message = redact(f"{type(exc).__name__}: {exc}", [api_key])
            raise PreflightError(f"cannot read the model list at {url}: {message}") from exc
        if response.status_code != 200:
            body = redact(response.text[:300], [api_key])
            raise PreflightError(
                f"the model list at {url} answered HTTP {response.status_code}: {body}"
            )
        payload = response.json()
        entries = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise PreflightError(f"the model list at {url} has no data array")
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                catalogue[normalize_model_id(entry["id"])] = entry
        links = payload.get("links") if isinstance(payload, dict) else None
        following = links.get("next") if isinstance(links, dict) else None
        url = str(httpx.URL(url).join(following)) if isinstance(following, str) else None
    return catalogue


def run_preflight(
    modes_file: ModesFile,
    modes: Iterable[Mode],
    env: Mapping[str, str],
    client: httpx.Client,
) -> PreflightReport:
    """Check each mode against the catalogue of its provider.

    Each problem is collected, and one error lists them all, so that one
    preflight is sufficient to correct a modes file.
    """
    report = PreflightReport()
    problems: list[str] = []
    catalogues: dict[str, dict[str, dict[str, Any]]] = {}
    for mode in modes:
        provider = modes_file.providers[mode.provider]
        if provider.kind != "openai_compat":
            continue
        if mode.provider not in catalogues:
            api_key = env.get(provider.api_key_env, "")
            if not api_key:
                problems.append(f"{mode.id}: the key {provider.api_key_env} is not set")
                continue
            catalogues[mode.provider] = fetch_catalogue(client, provider, api_key)
        catalogue = catalogues[mode.provider]
        key = _catalogue_key(mode.model, catalogue)
        if key is None:
            problems.append(f"{mode.id}: the model {mode.model} is not in {mode.provider}")
            continue
        report.matched[mode.id] = key
        price = catalogue_price(catalogue[key])
        if price is not None:
            report.prices[mode.id] = price
    if problems:
        details = "\n".join(f"- {problem}" for problem in problems)
        raise PreflightError(f"The preflight failed:\n{details}")
    return report
