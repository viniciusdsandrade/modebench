"""Builders that the tests share. No helper opens a network connection."""

from pathlib import Path

from modebench.config import BenchConfig, Mode, ModesFile, Price, ProviderConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DASH = chr(0x2014)


class TickClock:
    """A clock that moves one step forward each time it is read."""

    def __init__(self, step_ms: float = 1.0) -> None:
        self.calls = 0
        self._step_ns = int(step_ms * 1_000_000)

    def __call__(self) -> int:
        value = self.calls * self._step_ns
        self.calls += 1
        return value


def bench_for(bench: BenchConfig, runs_dir: Path, **path_updates: object) -> BenchConfig:
    """Return `bench` with its results below `runs_dir` and with a small bootstrap."""
    paths = bench.paths.model_copy(update={"runs_dir": runs_dir, **path_updates})
    stats = bench.stats.model_copy(update={"bootstrap_resamples": 200})
    return bench.model_copy(update={"paths": paths, "stats": stats})


def local_modes_file(*modes: Mode) -> ModesFile:
    """Return a modes file with a fake provider and two network providers that no test calls."""
    providers = {
        "local": ProviderConfig(kind="fake", privacy="local"),
        "openrouter": ProviderConfig(
            kind="openai_compat",
            base_url="https://openrouter.test/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            privacy="openrouter",
            cost_source="usage",
        ),
        "google": ProviderConfig(
            kind="openai_compat",
            base_url="https://google.test/v1beta/openai",
            api_key_env="GEMINI_API_KEY",
            privacy="attested",
        ),
    }
    return ModesFile(providers=providers, modes=list(modes))


def fake_mode(mode_id: str, **fake: float) -> Mode:
    """Return a mode of the fake provider with the given behaviour."""
    return Mode(
        id=mode_id,
        provider="local",
        model=f"fake/{mode_id}",
        family="fake",
        params={"fake": dict(fake)},
        price=Price(usd_per_mtok_in=1.0, usd_per_mtok_out=2.0),
    )


def openrouter_mode(mode_id: str, model: str = "google/gemini-test", **params: object) -> Mode:
    """Return a mode of the OpenRouter test provider."""
    return Mode(id=mode_id, provider="openrouter", model=model, params=dict(params))
