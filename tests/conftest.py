"""Shared fixtures. No test opens a network connection."""

from pathlib import Path

import pytest

from helpers import REPO_ROOT
from modebench.config import BenchConfig, ModesFile, load_bench_config, load_modes_file


@pytest.fixture
def repo_root() -> Path:
    """Return the root of the repository, where the seed configs and the public data are."""
    return REPO_ROOT


@pytest.fixture
def seed_bench() -> BenchConfig:
    """Return the bench file of the repository."""
    return load_bench_config(REPO_ROOT / "configs" / "bench.toml")


@pytest.fixture
def seed_modes() -> ModesFile:
    """Return the modes file of the repository."""
    return load_modes_file(REPO_ROOT / "configs" / "modes.toml")
