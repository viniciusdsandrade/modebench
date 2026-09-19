"""Reads dataset files and decides which of them are private."""

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from modebench.dataset.schema import DatasetFile, FillerFile
from modebench.errors import DatasetError

PRIVATE_PARTS = ("data", "private")


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """A dataset with its path and its privacy."""

    path: Path
    private: bool
    data: DatasetFile


def is_private_path(root: Path, path: Path) -> bool:
    """Return True if `path` is below `root/data/private`."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return relative.parts[: len(PRIVATE_PARTS)] == PRIVATE_PARTS


def _read_yaml(path: Path) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetError(f"dataset file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise DatasetError(f"{path} is not valid YAML: {exc}") from exc


def load_dataset(root: Path, path: Path) -> LoadedDataset:
    """Read one dataset file.

    A dataset is private if its file says so or if it is in the private
    directory. One of the two is sufficient, so a private file with a wrong
    header stays private.
    """
    try:
        data = DatasetFile.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise DatasetError(f"{path} is not a valid dataset:\n{exc}") from exc
    private = data.visibility == "private" or is_private_path(root, path)
    return LoadedDataset(path=path, private=private, data=data)


def load_filler(path: Path) -> FillerFile:
    """Read the filler file."""
    try:
        return FillerFile.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise DatasetError(f"{path} is not a valid filler file:\n{exc}") from exc
