"""The configuration of the speech to text suite, and the manifest of its audio."""

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import Field, ValidationError, model_validator

from modebench.config import StrictModel, load_toml
from modebench.errors import ConfigError, DatasetError


class SttBilling(StrictModel):
    """How a provider charges.

    `session_seconds` is the time for which the connection is open, idle time
    included. `audio_seconds` is the length of the audio that was sent.
    """

    basis: Literal["session_seconds", "audio_seconds"]
    usd_per_hour: float = Field(ge=0)
    addons_usd_per_hour: dict[str, float] = Field(default_factory=dict)

    @property
    def total_usd_per_hour(self) -> float:
        """Return the base price and the add-ons together."""
        return self.usd_per_hour + sum(self.addons_usd_per_hour.values())


class SttProviderConfig(StrictModel):
    """One speech to text provider. `query` goes to the connect URL unchanged."""

    kind: Literal["assemblyai", "elevenlabs", "fake"]
    url: str = ""
    api_key_env: str = ""
    query: dict[str, Any] = Field(default_factory=dict)
    billing: SttBilling
    preroll_silence_ms: int = Field(default=0, ge=0)
    private_data_ok: bool = False
    enabled: bool = True


class SttReplayConfig(StrictModel):
    """How the audio goes out: the chunk, the silence after the audio, and the waits."""

    chunk_ms: int = Field(default=50, ge=10, le=1000)
    tail_silence_ms: int = Field(default=3000, ge=0)
    close_grace_s: float = Field(default=1.0, ge=0)
    drain_timeout_s: float = Field(default=8.0, gt=0)
    connect_timeout_s: float = Field(default=15.0, gt=0)


class SttNormalization(StrictModel):
    """What the text normalisation does before the error rates are computed."""

    expand_numbers: bool = True
    expand_contractions: bool = True
    drop_hesitations: bool = True


class SttProfile(StrictModel):
    """The size of a run of the speech suite."""

    audios: int | None = Field(default=None, gt=0)
    repetitions: int = Field(default=3, gt=0)
    max_cost_usd: float = Field(gt=0)


class SttConfig(StrictModel):
    """The whole speech to text config file."""

    manifest: Path = Path("data/private/stt/manifest.yaml")
    runs_dir: Path = Path("runs")
    replay: SttReplayConfig = Field(default_factory=SttReplayConfig)
    normalization: SttNormalization = Field(default_factory=SttNormalization)
    providers: dict[str, SttProviderConfig]
    profiles: dict[str, SttProfile]

    def profile(self, name: str) -> SttProfile:
        """Return the profile with this name."""
        if name not in self.profiles:
            known = ", ".join(sorted(self.profiles))
            raise ConfigError(f"unknown profile {name}. Known profiles: {known}")
        return self.profiles[name]


class ReferenceUtterance(StrictModel):
    """One utterance of the reference: when it was said, who said it, and the words."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    speaker: str = ""
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        if self.end_ms <= self.start_ms:
            raise ValueError("an utterance must end after it starts")
        return self


class AudioItem(StrictModel):
    """One WAV file and its reference transcript. The path starts at the manifest."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    wav: Path
    language: str = "pt"
    utterances: list[ReferenceUtterance] = Field(min_length=1)

    @property
    def reference_text(self) -> str:
        """Return the words of the reference, in the order in which they were said."""
        ordered = sorted(self.utterances, key=lambda item: item.start_ms)
        return " ".join(item.text for item in ordered)


class SttManifest(StrictModel):
    """The audio files of the suite. A manifest is private unless it says that it is public."""

    version: Literal[1] = 1
    visibility: Literal["public", "private"] = "private"
    audios: list[AudioItem] = Field(min_length=1)


def load_stt_config(path: Path) -> SttConfig:
    """Read and validate the speech to text config file."""
    try:
        return SttConfig.model_validate(load_toml(path))
    except ValidationError as exc:
        raise ConfigError(f"{path} is not a valid speech to text config:\n{exc}") from exc


def load_manifest(path: Path) -> SttManifest:
    """Read and validate a manifest."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetError(f"manifest not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise DatasetError(f"{path} is not valid YAML: {exc}") from exc
    try:
        return SttManifest.model_validate(data)
    except ValidationError as exc:
        raise DatasetError(f"{path} is not a valid manifest:\n{exc}") from exc
