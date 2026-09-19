"""Keys come from the process environment or from the .env file, and from nowhere else."""

import os
from collections.abc import Mapping
from pathlib import Path


def parse_env_text(text: str) -> dict[str, str]:
    """Parse the text of a .env file.

    A line is `KEY=value`. An `export ` prefix, blank lines and `#` comments
    are ignored. One pair of quotes around the value is removed.
    """
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_env(root: Path, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the variables of `root/.env` with the process environment on top.

    An empty value counts as absent, so an unfilled line of the example file
    does not hide a key that the shell exports.
    """
    merged: dict[str, str] = {}
    env_file = root / ".env"
    if env_file.is_file():
        merged.update(parse_env_text(env_file.read_text(encoding="utf-8")))
    source = os.environ if environ is None else environ
    merged.update({key: value for key, value in source.items() if value})
    return {key: value for key, value in merged.items() if value}
