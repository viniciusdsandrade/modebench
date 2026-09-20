"""Builds the provider object of each mode."""

from collections.abc import Iterable, Mapping

import httpx

from modebench.config import Mode, ModesFile, ProviderConfig
from modebench.errors import PreflightError
from modebench.providers.base import ChatProvider
from modebench.providers.fake import FakeProvider
from modebench.providers.openai_compat import OpenAICompatProvider

FAKE_PROVIDER_CONFIG = ProviderConfig(kind="fake", privacy="local")


def effective_provider_config(
    modes_file: ModesFile, mode: Mode, *, dry_run: bool
) -> ProviderConfig:
    """Return the provider of a mode. In a dry run, each mode gets the fake provider."""
    if dry_run:
        return FAKE_PROVIDER_CONFIG
    return modes_file.providers[mode.provider]


def build_providers(
    modes_file: ModesFile,
    modes: Iterable[Mode],
    env: Mapping[str, str],
    *,
    dry_run: bool,
    client: httpx.Client | None = None,
) -> dict[str, ChatProvider]:
    """Return one provider object for each mode identifier.

    Modes that share an endpoint share one object, so they share one HTTP
    connection pool, as the requests of one application do.
    """
    by_name: dict[str, ChatProvider] = {}
    result: dict[str, ChatProvider] = {}
    missing: list[str] = []
    for mode in modes:
        config = effective_provider_config(modes_file, mode, dry_run=dry_run)
        name = "fake" if config.kind == "fake" else mode.provider
        if name not in by_name:
            if config.kind == "fake":
                by_name[name] = FakeProvider()
            else:
                api_key = env.get(config.api_key_env, "")
                if not api_key:
                    missing.append(config.api_key_env)
                    continue
                by_name[name] = OpenAICompatProvider(config, api_key, client=client)
        result[mode.id] = by_name[name]
    if missing:
        names = ", ".join(sorted(set(missing)))
        raise PreflightError(f"missing keys in the environment or in .env: {names}")
    return result


def close_providers(providers: Iterable[ChatProvider]) -> None:
    """Close the HTTP client of each provider object, one time for each object."""
    seen: set[int] = set()
    for provider in providers:
        if id(provider) in seen:
            continue
        seen.add(id(provider))
        if isinstance(provider, OpenAICompatProvider):
            provider.close()
