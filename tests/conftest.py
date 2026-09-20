"""Shared fixtures. No test opens a network connection, and no test writes in the repository."""

import shutil
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


@pytest.fixture
def isolated_root(tmp_path: Path) -> Path:
    """Return a copy of the inputs of the repository, so that a run writes below `tmp_path`.

    A command line test that used the repository as its root would put fake
    runs in the true `runs/` directory, and `--run latest` would then find them.
    """
    root = tmp_path / "root"
    shutil.copytree(REPO_ROOT / "configs", root / "configs")
    shutil.copytree(REPO_ROOT / "prompts", root / "prompts")
    shutil.copytree(REPO_ROOT / "data" / "public", root / "data" / "public")
    return root
