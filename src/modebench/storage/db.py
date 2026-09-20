"""The SQLite store: runs, requests and scores, and the tables of the speech suite.

One database file holds every run. A row is written when its request ends,
so a run that stops early keeps what it measured.
"""

import sqlite3
from collections.abc import Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from modebench.errors import StorageError
from modebench.storage.records import (
    RequestRecord,
    RunRecord,
    ScoreRecord,
    StoredRequest,
    SttFinalRecord,
    SttSessionRecord,
)

DB_NAME = "modebench.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  run_id TEXT PRIMARY KEY,
  suite TEXT NOT NULL,
  profile TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL,
  dry_run INTEGER NOT NULL,
  git_sha TEXT NOT NULL,
  git_dirty INTEGER NOT NULL,
  config_hash TEXT NOT NULL,
  dataset_hash TEXT NOT NULL,
  prompt_hash TEXT NOT NULL,
  timeout_s REAL NOT NULL,
  judge_model TEXT NOT NULL,
  modes_json TEXT NOT NULL,
  estimated_cost_usd REAL,
  actual_cost_usd REAL
);
CREATE TABLE IF NOT EXISTS requests(
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  seq INTEGER NOT NULL,
  mode_id TEXT NOT NULL,
  suite TEXT NOT NULL,
  item_id TEXT NOT NULL,
  case_id TEXT NOT NULL,
  variant_id TEXT NOT NULL,
  variant_kind TEXT NOT NULL,
  truncation_pct INTEGER NOT NULL,
  asr_wer REAL NOT NULL,
  noise_kind TEXT,
  duration_min INTEGER NOT NULL,
  repetition INTEGER NOT NULL,
  warmup INTEGER NOT NULL,
  scenario_id TEXT,
  click_index INTEGER,
  cache_state TEXT NOT NULL,
  expect_refusal INTEGER NOT NULL,
  private INTEGER NOT NULL,
  started_at TEXT NOT NULL,
  ok INTEGER NOT NULL,
  error_kind TEXT,
  error_message TEXT,
  http_status INTEGER,
  first_chunk_ms REAL,
  ttft_ms REAL,
  first_reasoning_ms REAL,
  ttfat_ms REAL,
  total_ms REAL NOT NULL,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  reasoning_tokens INTEGER,
  answer_tokens INTEGER,
  cached_tokens INTEGER,
  tok_per_s REAL,
  served_by TEXT,
  cost_usd REAL,
  cost_source TEXT NOT NULL,
  prompt_chars INTEGER NOT NULL,
  prompt_sha TEXT NOT NULL,
  answer TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_requests_run ON requests(run_id, mode_id);
CREATE TABLE IF NOT EXISTS scores(
  id INTEGER PRIMARY KEY,
  request_id INTEGER NOT NULL REFERENCES requests(id),
  scorer TEXT NOT NULL,
  name TEXT NOT NULL,
  value REAL,
  detail TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_scores_request ON scores(request_id);
CREATE TABLE IF NOT EXISTS stt_sessions(
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  provider TEXT NOT NULL,
  audio_id TEXT NOT NULL,
  repetition INTEGER NOT NULL,
  started_at TEXT NOT NULL,
  ok INTEGER NOT NULL,
  error_message TEXT,
  audio_ms REAL NOT NULL,
  session_ms REAL NOT NULL,
  connect_ms REAL,
  first_partial_ms REAL,
  first_partial_after_speech_ms REAL,
  final_latency_p50_ms REAL,
  final_latency_p95_ms REAL,
  finals INTEGER NOT NULL,
  wer REAL,
  cer REAL,
  speaker_accuracy REAL,
  max_send_lag_ms REAL NOT NULL,
  cost_usd REAL,
  hypothesis TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stt_finals(
  id INTEGER PRIMARY KEY,
  session_id INTEGER NOT NULL REFERENCES stt_sessions(id),
  utterance_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  speaker TEXT,
  received_ms REAL NOT NULL,
  audio_end_ms REAL,
  latency_ms REAL
);
"""

_REQUEST_BOOLS = ("warmup", "expect_refusal", "private", "ok")
_RUN_BOOLS = ("dry_run", "git_dirty")
_STT_BOOLS = ("ok",)


def _insert_sql(table: str, names: Sequence[str]) -> str:
    columns = ", ".join(names)
    values = ", ".join(f":{name}" for name in names)
    return f"INSERT INTO {table}({columns}) VALUES({values})"


def _from_row(row: sqlite3.Row, names: Sequence[str], bools: Sequence[str]) -> dict[str, Any]:
    values: dict[str, Any] = {name: row[name] for name in names}
    for name in bools:
        values[name] = bool(values[name])
    return values


class RunStore:
    """Reads and writes the database of the runs."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(db_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        """Close the database."""
        self._connection.close()

    def create_run(self, record: RunRecord) -> None:
        """Write the row of a run that starts."""
        names = [item.name for item in fields(RunRecord)]
        self._connection.execute(_insert_sql("runs", names), asdict(record))
        self._connection.commit()

    def finish_run(
        self, run_id: str, status: str, finished_at: str, actual_cost_usd: float | None
    ) -> None:
        """Write how a run ended."""
        self._connection.execute(
            "UPDATE runs SET status = ?, finished_at = ?, actual_cost_usd = ? WHERE run_id = ?",
            (status, finished_at, actual_cost_usd, run_id),
        )
        self._connection.commit()

    def insert_request(self, record: RequestRecord) -> int:
        """Write one request and return its identifier."""
        names = [item.name for item in fields(RequestRecord)]
        cursor = self._connection.execute(_insert_sql("requests", names), asdict(record))
        self._connection.commit()
        if cursor.lastrowid is None:
            raise StorageError("the database gave no identifier to a request")
        return cursor.lastrowid

    def insert_scores(self, request_id: int, scores: Sequence[ScoreRecord]) -> None:
        """Write the scores of one request."""
        rows = [
            (request_id, score.scorer, score.name, score.value, score.detail) for score in scores
        ]
        self._connection.executemany(
            "INSERT INTO scores(request_id, scorer, name, value, detail) VALUES(?, ?, ?, ?, ?)",
            rows,
        )
        self._connection.commit()

    def insert_stt_session(self, record: SttSessionRecord, finals: Sequence[SttFinalRecord]) -> int:
        """Write one replay and its settled utterances."""
        names = [item.name for item in fields(SttSessionRecord)]
        cursor = self._connection.execute(_insert_sql("stt_sessions", names), asdict(record))
        session_id = cursor.lastrowid
        if session_id is None:
            raise StorageError("the database gave no identifier to a session")
        final_names = [item.name for item in fields(SttFinalRecord)]
        sql = _insert_sql("stt_finals", ["session_id", *final_names])
        for final in finals:
            self._connection.execute(sql, {"session_id": session_id, **asdict(final)})
        self._connection.commit()
        return session_id

    def latest_run_id(self, suite: str | None = None) -> str:
        """Return the newest run, of one suite if `suite` is given."""
        if suite is None:
            row = self._connection.execute(
                "SELECT run_id FROM runs ORDER BY started_at DESC, run_id DESC LIMIT 1"
            ).fetchone()
        else:
            row = self._connection.execute(
                "SELECT run_id FROM runs WHERE suite = ? "
                "ORDER BY started_at DESC, run_id DESC LIMIT 1",
                (suite,),
            ).fetchone()
        if row is None:
            raise StorageError("the database has no run")
        return str(row["run_id"])

    def resolve_run_id(self, run: str, suite: str | None = None) -> str:
        """Return `run`, or the newest run if `run` is `latest`."""
        return self.latest_run_id(suite) if run == "latest" else run

    def load_run(self, run_id: str) -> RunRecord:
        """Return the row of one run."""
        row = self._connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise StorageError(f"run not found: {run_id}")
        names = [item.name for item in fields(RunRecord)]
        return RunRecord(**_from_row(row, names, _RUN_BOOLS))

    def load_requests(self, run_id: str, *, include_warmup: bool = False) -> list[StoredRequest]:
        """Return the requests of a run in the order in which they went out."""
        rows = self._connection.execute(
            "SELECT * FROM requests WHERE run_id = ? ORDER BY seq", (run_id,)
        ).fetchall()
        names = [item.name for item in fields(RequestRecord)]
        result: list[StoredRequest] = []
        for row in rows:
            record = RequestRecord(**_from_row(row, names, _REQUEST_BOOLS))
            if record.warmup and not include_warmup:
                continue
            result.append(StoredRequest(id=int(row["id"]), record=record))
        return result

    def load_scores(self, run_id: str) -> dict[int, dict[str, ScoreRecord]]:
        """Return the scores of a run: request identifier, then score name."""
        rows = self._connection.execute(
            "SELECT s.request_id, s.scorer, s.name, s.value, s.detail FROM scores s "
            "JOIN requests r ON r.id = s.request_id WHERE r.run_id = ? ORDER BY s.id",
            (run_id,),
        ).fetchall()
        result: dict[int, dict[str, ScoreRecord]] = {}
        for row in rows:
            score = ScoreRecord(
                scorer=str(row["scorer"]),
                name=str(row["name"]),
                value=None if row["value"] is None else float(row["value"]),
                detail=str(row["detail"]),
            )
            result.setdefault(int(row["request_id"]), {})[f"{score.scorer}.{score.name}"] = score
        return result

    def load_stt_final_latencies(self, run_id: str) -> dict[str, list[float]]:
        """Return the latency of each settled utterance of a run, for each provider."""
        rows = self._connection.execute(
            "SELECT s.provider, f.latency_ms FROM stt_finals f "
            "JOIN stt_sessions s ON s.id = f.session_id "
            "WHERE s.run_id = ? AND f.latency_ms IS NOT NULL ORDER BY f.id",
            (run_id,),
        ).fetchall()
        result: dict[str, list[float]] = {}
        for row in rows:
            result.setdefault(str(row["provider"]), []).append(float(row["latency_ms"]))
        return result

    def load_stt_sessions(self, run_id: str) -> list[SttSessionRecord]:
        """Return the replays of a run of the speech suite."""
        rows = self._connection.execute(
            "SELECT * FROM stt_sessions WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        names = [item.name for item in fields(SttSessionRecord)]
        return [SttSessionRecord(**_from_row(row, names, _STT_BOOLS)) for row in rows]
