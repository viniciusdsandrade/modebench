"""A small TOML writer for the recommendations file.

The standard library reads TOML and does not write it. The file that this
module writes is small and has a fixed shape, so a writer of forty lines is
better than one more dependency. The tests read its output with `tomllib`.
"""

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _key(name: str) -> str:
    return name if _BARE_KEY.match(name) else _string(name)


def _string(text: str) -> str:
    parts: list[str] = []
    for char in text:
        if char in _ESCAPES:
            parts.append(_ESCAPES[char])
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            parts.append(f"\\u{ord(char):04x}")
        else:
            parts.append(char)
    return '"' + "".join(parts) + '"'


def _float(number: float) -> str:
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    text = repr(number)
    return text if any(mark in text for mark in ".en") else text + ".0"


def format_value(value: Any) -> str:
    """Return a value as it stands on the right of `=`. A mapping becomes an inline table."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _float(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, Mapping):
        items = [f"{_key(str(k))} = {format_value(v)}" for k, v in value.items() if v is not None]
        return "{ " + ", ".join(items) + " }" if items else "{}"
    if isinstance(value, Sequence):
        return "[" + ", ".join(format_value(item) for item in value if item is not None) + "]"
    raise TypeError(f"TOML cannot hold a value of type {type(value).__name__}")


def _emit(path: list[str], table: Mapping[str, Any], lines: list[str]) -> None:
    scalars = {k: v for k, v in table.items() if v is not None and not isinstance(v, Mapping)}
    tables = {k: v for k, v in table.items() if isinstance(v, Mapping)}
    if path:
        if lines:
            lines.append("")
        lines.append("[" + ".".join(_key(part) for part in path) + "]")
    for name, value in scalars.items():
        lines.append(f"{_key(name)} = {format_value(value)}")
    for name, value in tables.items():
        _emit([*path, name], value, lines)


def dumps(document: Mapping[str, Any], header: str = "") -> str:
    """Return a TOML document. `header` becomes comment lines at the top.

    A key whose value is None is left out, because TOML has no null.
    """
    lines: list[str] = [f"# {line}".rstrip() for line in header.splitlines()]
    _emit([], document, lines)
    return "\n".join(lines) + "\n"
