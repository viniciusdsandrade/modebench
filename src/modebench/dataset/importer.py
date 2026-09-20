"""Imports sessions of the stt-bridge application into a private dataset.

The application keeps each meeting in SQLite: the settled lines in
`segments`, and each answer of the Analyze button in `interpretations`. One
completed answer becomes one case. The lines that the answer covered become
the context of the case, the lines before them become its earlier stretch,
and the answer that production gave becomes the reference answer.

The database is opened read-only. The output is always marked private, only
its owner can read it, and it cannot go to a directory of the repository that
git follows.
"""

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from modebench.dataset.loader import is_private_path
from modebench.dataset.schema import Case, DatasetFile, Line
from modebench.errors import DatasetError

ANALYZE_TASK = "recent_question"


@dataclass(frozen=True, slots=True)
class ImportOptions:
    """Which answers to import."""

    task: str | None = ANALYZE_TASK
    windowed_only: bool = True
    meeting_id: int | None = None
    limit: int | None = None


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.is_file():
        raise DatasetError(f"database not found: {db_path}")
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row["name"]) for row in rows}


def _interpretations(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    columns = _columns(connection, "interpretations")
    if not columns:
        raise DatasetError("the database has no interpretations table")
    windowed = "windowed" if "windowed" in columns else "0"
    upper_id = "up_to_segment_id" if "up_to_segment_id" in columns else "0"
    query = (
        f"SELECT id, meeting_id, task, up_to_ts_ms, model, text, "
        f"{windowed} AS windowed, {upper_id} AS upper_id "
        f"FROM interpretations WHERE status = 'completed' AND text != '' "
        f"ORDER BY meeting_id, id"
    )
    return connection.execute(query).fetchall()


def _segments(connection: sqlite3.Connection, meeting_id: int) -> list[sqlite3.Row]:
    query = (
        "SELECT id, source, ts_start_ms, text FROM segments "
        "WHERE meeting_id = ? ORDER BY ts_start_ms, id"
    )
    return connection.execute(query, (meeting_id,)).fetchall()


def _lines(rows: list[sqlite3.Row]) -> list[Line]:
    lines: list[Line] = []
    for row in rows:
        text = " ".join(str(row["text"]).split())
        if text and row["source"] in ("mic", "system"):
            lines.append(Line(speaker=row["source"], text=text))
    return lines


def _in_window(row: sqlite3.Row, floor: tuple[int, int], upper: tuple[int, int]) -> int:
    """Return -1 for a line before the window, 0 for a line in it and 1 for a line after it.

    A bound is a segment identifier and a time. The identifier decides when
    the answer recorded one. Rows of old databases have no identifier, and
    then the time decides.
    """
    upper_id, upper_ts = upper
    floor_id, floor_ts = floor
    if upper_id > 0:
        if int(row["id"]) > upper_id:
            return 1
        return -1 if int(row["id"]) <= floor_id else 0
    if int(row["ts_start_ms"]) > upper_ts:
        return 1
    return -1 if int(row["ts_start_ms"]) <= floor_ts and floor_ts > 0 else 0


def import_cases(db_path: Path, options: ImportOptions | None = None) -> list[Case]:
    """Return one case for each completed answer that the options select."""
    chosen = options if options is not None else ImportOptions()
    connection = _connect(db_path)
    try:
        return _import(connection, chosen)
    except sqlite3.Error as exc:
        raise DatasetError(f"{db_path} is not a database of the application: {exc}") from exc
    finally:
        connection.close()


def _import(connection: sqlite3.Connection, chosen: ImportOptions) -> list[Case]:
    cases: list[Case] = []
    floors: dict[int, tuple[int, int]] = {}
    segments: dict[int, list[sqlite3.Row]] = {}
    for row in _interpretations(connection):
        meeting_id = int(row["meeting_id"])
        windowed = bool(row["windowed"])
        upper = (int(row["upper_id"]), int(row["up_to_ts_ms"]))
        floor = floors.get(meeting_id, (0, 0)) if windowed else (0, 0)
        if windowed:
            floors[meeting_id] = upper
        if chosen.task is not None and row["task"] != chosen.task:
            continue
        if chosen.windowed_only and not windowed:
            continue
        if chosen.meeting_id is not None and meeting_id != chosen.meeting_id:
            continue
        if meeting_id not in segments:
            segments[meeting_id] = _segments(connection, meeting_id)
        rows = segments[meeting_id]
        earlier = _lines([item for item in rows if _in_window(item, floor, upper) < 0])
        fresh = _lines([item for item in rows if _in_window(item, floor, upper) == 0])
        cases.append(
            Case(
                id=f"m{meeting_id}-i{int(row['id'])}",
                context=fresh,
                earlier=earlier,
                expect_refusal=not fresh,
                reference_answer=str(row["text"]),
                needs_annotation=True,
                tags=[str(row["task"]), str(row["model"])],
            )
        )
        if chosen.limit is not None and len(cases) >= chosen.limit:
            break
    return cases


def dataset_payload(cases: list[Case]) -> dict[str, Any]:
    """Return the YAML document of a private dataset."""
    return {
        "version": 1,
        "language": "pt-BR",
        "visibility": "private",
        "cases": [case.model_dump(mode="json", exclude_defaults=True) for case in cases],
    }


def _inside(root: Path, path: Path) -> bool:
    try:
        Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root)))
    except ValueError:
        return False
    return True


def write_private_dataset(
    cases: list[Case], out_path: Path, *, force: bool = False, root: Path | None = None
) -> Path:
    """Write the cases to `out_path`. An existing file stays unless `force` is set.

    With `root`, a path in the repository must be below `data/private`, which
    git ignores. Meeting text in a directory that git follows is one `git add`
    from a public commit. A path out of the repository is permitted.
    """
    if not cases:
        raise DatasetError("the database has no completed answer that the options select")
    if root is not None and _inside(root, out_path) and not is_private_path(root, out_path):
        raise DatasetError(
            f"{out_path} is in the repository and not below data/private. "
            "Write the private dataset below data/private, or out of the repository"
        )
    if out_path.exists() and not force:
        raise DatasetError(f"{out_path} exists. Use --force to replace it")
    DatasetFile.model_validate(dataset_payload(cases))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(dataset_payload(cases), allow_unicode=True, sort_keys=False, width=100)
    descriptor = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    # The mode of `os.open` has no effect on a file that was there before.
    out_path.chmod(0o600)
    return out_path
