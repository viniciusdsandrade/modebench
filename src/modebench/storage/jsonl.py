"""The raw log of a run: one JSON object for each line, written as the run goes."""

import json
from pathlib import Path
from types import TracebackType
from typing import Any, Self

RAW_NAME = "raw.jsonl"


class RawLog:
    """Appends records to a JSONL file and flushes after each one."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        """Append one record."""
        self._handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        self._handle.write("\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the file."""
        self._handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Return each record of a JSONL file."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


# Convenience alias
JsonlWriter = RawLog
