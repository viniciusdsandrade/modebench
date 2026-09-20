"""Configuration models and their loaders.

Two files configure a run. The modes file holds the providers and the modes.
The bench file holds the paths, the profiles, the judge and the objectives of
each role. Both are TOML, and both are validated before a request leaves.
"""

import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from modebench.errors import ConfigError

NoiseKind = Literal["empty", "two_words", "hesitation"]

# Model name prefixes of the direct APIs, which have no author in the identifier.
_FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("gemini", "google"),
    ("gemma", "google"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("claude", "anthropic"),
    ("glm", "z-ai"),
    ("llama", "meta-llama"),
    ("mistral", "mistralai"),
    ("qwen", "qwen"),
    ("deepseek", "deepseek"),
    ("grok", "x-ai"),
)


class StrictModel(BaseModel):
    """Base of every configuration model: unknown keys are errors."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Price(StrictModel):
    """Token prices of one mode, in United States dollars for one million tokens."""

    usd_per_mtok_in: float = Field(ge=0)
    usd_per_mtok_out: float = Field(ge=0)
    usd_per_mtok_cached_in: float | None = Field(default=None, ge=0)


class ProviderConfig(StrictModel):
    """One API endpoint that a mode can use.

    `privacy` tells the guard how to read the data policy of the route:
    `openrouter` needs `provider.data_collection = "deny"` in the mode,
    `attested` needs `private_data_ok = true` in the mode, and `local` never
    sends data to a network.
    """

    kind: Literal["openai_compat", "fake"]
    base_url: str = ""
    api_key_env: str = ""
    privacy: Literal["openrouter", "attested", "local"] = "attested"
    cost_source: Literal["usage", "price_table"] = "price_table"
    headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    models_path: str = "/models"

    @model_validator(mode="after")
    def _check_endpoint(self) -> Self:
        if self.kind == "openai_compat" and (not self.base_url or not self.api_key_env):
            raise ValueError("an openai_compat provider needs base_url and api_key_env")
        return self


class Mode(StrictModel):
    """A provider, a model and the raw parameters that go to the API unchanged."""

    # The identifier goes into Markdown tables and code spans as it is.
    id: str = Field(min_length=1, pattern=r"^[^\s|`]+$")
    provider: str
    model: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    roles: list[str] = Field(default_factory=list)
    family: str = ""
    price: Price | None = None
    private_data_ok: bool = False
    overhead_of: str | None = None
    enabled: bool = True

    def resolved_family(self) -> str:
        """Return the model family, which the judge must not share."""
        if self.family:
            return self.family.lower()
        model = self.model.lower()
        if "/" in model and not model.startswith("models/"):
            return model.split("/", 1)[0]
        name = model.removeprefix("models/")
        for prefix, family in _FAMILY_PREFIXES:
            if name.startswith(prefix):
                return family
        return self.provider.lower()

    def reasoning_effort(self) -> str:
        """Return the reasoning level that the raw parameters ask for."""
        nested = self.params.get("reasoning")
        if isinstance(nested, dict):
            effort = nested.get("effort")
            if isinstance(effort, str):
                return effort
            if nested.get("enabled") is False:
                return "none"
        flat = self.params.get("reasoning_effort")
        if isinstance(flat, str):
            return flat
        return "default"

    def serves_role(self, role: str) -> bool:
        """Return True if the mode is a candidate for `role`. No list means all roles."""
        return not self.roles or role in self.roles


class ModesFile(StrictModel):
    """The providers and the modes of a benchmark."""

    providers: dict[str, ProviderConfig]
    modes: list[Mode]

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        ids = [mode.id for mode in self.modes]
        duplicates = sorted({mode_id for mode_id in ids if ids.count(mode_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate mode ids: {', '.join(duplicates)}")
        for mode in self.modes:
            if mode.provider not in self.providers:
                raise ValueError(f"mode {mode.id} names the unknown provider {mode.provider}")
            if mode.overhead_of is not None and mode.overhead_of not in ids:
                raise ValueError(f"mode {mode.id} names the unknown mode {mode.overhead_of}")
        return self

    def mode(self, mode_id: str) -> Mode:
        """Return the mode with this identifier."""
        for mode in self.modes:
            if mode.id == mode_id:
                return mode
        raise ConfigError(f"unknown mode: {mode_id}")

    def select(self, ids: Sequence[str] | None = None) -> list[Mode]:
        """Return the enabled modes, or the named modes in the order given."""
        if ids is None:
            return [mode for mode in self.modes if mode.enabled]
        return [self.mode(mode_id) for mode_id in ids]


class PathsConfig(StrictModel):
    """Where the inputs and the outputs are. A relative path starts at the root."""

    modes_file: Path = Path("configs/modes.toml")
    datasets: list[Path] = Field(default_factory=lambda: [Path("data/public/analyze/cases.yaml")])
    filler: Path = Path("data/public/analyze/filler.yaml")
    preprompt_paths: list[Path] = Field(default_factory=lambda: [Path("prompts/analyze.md")])
    judge_prompt_path: Path = Path("prompts/judge.md")
    runs_dir: Path = Path("runs")


class TranscriptConfig(StrictModel):
    """How a transcript line looks. The defaults are those of the application."""

    mic_label: str = "VOCE"
    system_label: str = "CHAMADA"
    partial_marker: str = "[partial]"
    words_per_minute: int = Field(default=150, gt=0)


class ExecutionConfig(StrictModel):
    """How the requests go out. Concurrency is one, and only one."""

    timeout_s: float = Field(default=60.0, gt=0)
    concurrency: Literal[1] = 1
    pause_between_requests_s: float = Field(default=0.0, ge=0)
    pause_after_error_s: float = Field(default=2.0, ge=0)
    seed: int = 20260919


class Profile(StrictModel):
    """The size of a run: which variants, how many repetitions, which ceiling."""

    cases: int | None = Field(default=None, gt=0)
    truncations: list[int] = Field(min_length=1)
    durations_min: list[int] = Field(min_length=1)
    baseline_duration_min: int
    asr_wers: list[float] = Field(default_factory=list)
    noise_kinds: list[NoiseKind] = Field(default_factory=list)
    noise_instances: int = Field(default=1, ge=0)
    repetitions: int = Field(default=3, gt=0)
    warmup: int = Field(default=1, ge=0)
    design: Literal["star", "cross"] = "star"
    meeting_scenarios: int = Field(default=0, ge=0)
    max_cost_usd: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_ranges(self) -> Self:
        if any(pct < 1 or pct > 100 for pct in self.truncations):
            raise ValueError("a truncation is a percentage from 1 to 100")
        if any(minutes <= 0 for minutes in self.durations_min):
            raise ValueError("a transcript length is a number of minutes above 0")
        if any(wer <= 0 or wer > 0.5 for wer in self.asr_wers):
            raise ValueError("a synthetic WER is above 0 and at most 0.5")
        if self.baseline_duration_min not in self.durations_min:
            raise ValueError("baseline_duration_min must be one of durations_min")
        return self


class CostConfig(StrictModel):
    """The assumptions of the cost estimate that comes before a run."""

    chars_per_token: float = Field(default=3.5, gt=0)
    expected_answer_tokens: int = Field(default=220, ge=0)
    expected_reasoning_tokens: dict[str, int] = Field(
        default_factory=lambda: {
            "none": 0,
            "minimal": 64,
            "low": 256,
            "medium": 1024,
            "high": 4096,
            "default": 512,
        }
    )
    judge_output_tokens: int = Field(default=160, ge=0)


class JudgeConfig(StrictModel):
    """The fixed judge. Its family must be different from that of each candidate."""

    enabled: bool = True
    mode: Mode
    max_attempts: int = Field(default=3, gt=0)


class QualityWeights(StrictModel):
    """The weights of the quality score of one answer. They add up to one."""

    question: float = Field(default=0.5, ge=0)
    utility: float = Field(default=0.3, ge=0)
    key_points: float = Field(default=0.2, ge=0)

    @model_validator(mode="after")
    def _check_sum(self) -> Self:
        total = self.question + self.utility + self.key_points
        if abs(total - 1.0) > 1e-9:
            raise ValueError("the quality weights must add up to 1")
        return self


class StatsConfig(StrictModel):
    """The bootstrap that gives each estimate its confidence interval."""

    bootstrap_resamples: int = Field(default=2000, gt=0)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    seed: int = 7


class RoleSlo(StrictModel):
    """The objective of one role."""

    ttfat_p95_ms_max: float = Field(gt=0)
    question_accuracy_min: float = Field(default=0.8, ge=0, le=1)
    error_rate_max: float | None = Field(default=None, ge=0, le=1)


class RegressionConfig(StrictModel):
    """When a comparison with the baseline fails."""

    p95_increase_ratio: float = Field(default=0.20, gt=0)


class MeetingConfig(StrictModel):
    """The minutes of a meeting at which the operator clicks Analyze."""

    click_minutes: list[int] = Field(default_factory=lambda: [5, 15, 30, 60], min_length=1)

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if sorted(set(self.click_minutes)) != self.click_minutes:
            raise ValueError("click_minutes must increase")
        return self


class BenchConfig(StrictModel):
    """The whole bench file."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    transcript: TranscriptConfig = Field(default_factory=TranscriptConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    profiles: dict[str, Profile]
    cost: CostConfig = Field(default_factory=CostConfig)
    judge: JudgeConfig
    quality_weights: QualityWeights = Field(default_factory=QualityWeights)
    stats: StatsConfig = Field(default_factory=StatsConfig)
    roles: dict[str, RoleSlo]
    regression: RegressionConfig = Field(default_factory=RegressionConfig)
    meeting: MeetingConfig = Field(default_factory=MeetingConfig)

    def profile(self, name: str) -> Profile:
        """Return the profile with this name."""
        if name not in self.profiles:
            known = ", ".join(sorted(self.profiles))
            raise ConfigError(f"unknown profile {name}. Known profiles: {known}")
        return self.profiles[name]


def load_toml(path: Path) -> dict[str, Any]:
    """Read one TOML file."""
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc


def load_bench_config(path: Path) -> BenchConfig:
    """Read and validate the bench file."""
    try:
        return BenchConfig.model_validate(load_toml(path))
    except ValidationError as exc:
        raise ConfigError(f"{path} is not a valid bench file:\n{exc}") from exc


def load_modes_file(path: Path) -> ModesFile:
    """Read and validate the modes file."""
    try:
        return ModesFile.model_validate(load_toml(path))
    except ValidationError as exc:
        raise ConfigError(f"{path} is not a valid modes file:\n{exc}") from exc


def resolve_path(root: Path, path: Path) -> Path:
    """Return `path` as it stands if absolute, else below `root`."""
    return path if path.is_absolute() else root / path
