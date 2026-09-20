"""Reads dataset files and decides which of them are private."""

import os
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


def _below_private(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    parts = tuple(part.casefold() for part in relative.parts[: len(PRIVATE_PARTS)])
    return parts == PRIVATE_PARTS


def is_private_path(root: Path, path: Path) -> bool:
    """Return True if `path` is below `root/data/private`.

    The path is read two times: as it is written, and with its symbolic links
    resolved. One match is sufficient. A private directory that is a link to
    another disk is thus private, and so is a link in a public directory that
    points into the private one. Letter case is ignored, because the usual
    file systems of macOS and Windows ignore it.
    """
    written = _below_private(Path(os.path.abspath(root)), Path(os.path.abspath(path)))
    return written or _below_private(root.resolve(), path.resolve())


def _read_yaml(path: Path) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetError(f"dataset file not found: {path}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError(f"{path} cannot be read: {exc}") from exc
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
