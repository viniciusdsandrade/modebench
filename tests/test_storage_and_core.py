"""Tests for database storage (RunStore), JSONL output, hashing/git utilities, and secret redaction."""

from pathlib import Path

from helpers import REPO_ROOT
from modebench.hashing import git_dirty, git_sha, sha256_text, short_token, stable_seed
from modebench.redact import redact
from modebench.storage.db import RunStore
from modebench.storage.jsonl import JsonlWriter, read_jsonl
from modebench.storage.records import RequestRecord, RunRecord, ScoreRecord


def test_hashing_and_tokens() -> None:
    assert len(sha256_text("hello")) == 64
    assert sha256_text("hello") == sha256_text("hello")

    token1 = short_token("prefix-", "key-a")
    token2 = short_token("prefix-", "key-a")
    token3 = short_token("prefix-", "key-b")
    assert token1 == token2
    assert token1 != token3
    assert token1.startswith("prefix-")

    seed1 = stable_seed("seed-key")
    seed2 = stable_seed("seed-key")
    assert seed1 == seed2
    assert isinstance(seed1, int)


def test_git_info_returns_hash() -> None:
    sha = git_sha(REPO_ROOT)
    dirty = git_dirty(REPO_ROOT)
    assert len(sha) == 7 or len(sha) == 40 or sha == "unknown"
    assert isinstance(dirty, bool)


def test_redaction_hides_secrets() -> None:
    secret = "sk-ant-api03-very-secret-key-123456"
    text = f"Connecting with Authorization: Bearer {secret} to endpoint."
    redacted = redact(text, [secret])
    assert secret not in redacted
    assert "[REDACTED]" in redacted


def test_run_store_and_jsonl_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.db"
    store = RunStore(db_path)

    run_record = RunRecord(
        run_id="run-test-01",
        suite="analyze",
        profile="smoke",
        started_at="2026-09-01T12:00:00Z",
        status="completed",
        dry_run=False,
        git_sha="1234567",
        git_dirty=False,
        config_hash="cfg-hash",
        dataset_hash="ds-hash",
        prompt_hash="pr-hash",
        timeout_s=60.0,
        judge_model="fake-judge",
        modes_json='[{"id":"m1","provider":"local","model":"fake/m1"}]',
    )
    store.save_run(run_record)

    fetched_run = store.run("run-test-01")
    assert fetched_run is not None
    assert fetched_run.run_id == "run-test-01"
    assert fetched_run.profile == "smoke"
    assert store.latest_run_id("analyze") == "run-test-01"

    req = RequestRecord(
        seq=1,
        run_id="run-test-01",
        suite="analyze",
        mode_id="m1",
        item_id="item-01",
        repetition=1,
        warmup=False,
        cache_state="cold",
        ok=True,
        total_ms=1200.0,
        ttfat_ms=650.0,
        prompt_tokens=100,
        completion_tokens=50,
        cost_usd=0.0002,
        cost_source="usage",
        answer="Resposta completa.",
    )
    store.save_request(req)

    requests = store.requests("run-test-01")
    assert len(requests) == 1
    assert requests[0].item_id == "item-01"
    assert requests[0].ttfat_ms == 650.0

    score = ScoreRecord(
        run_id="run-test-01",
        request_id="req-01",
        scorer="deterministic",
        metric="non_empty",
        value=1.0,
        detail="true",
    )
    store.save_score(score)

    scores = store.scores("run-test-01")
    assert len(scores) == 1
    assert scores[0].metric == "non_empty"
    assert scores[0].value == 1.0

    # Test JSONL writer and reader
    jsonl_path = tmp_path / "events.jsonl"
    with JsonlWriter(jsonl_path) as writer:
        writer.write({"event": "start", "seq": 1})
        writer.write({"event": "token", "val": "abc"})

    lines = list(read_jsonl(jsonl_path))
    assert len(lines) == 2
    assert lines[0]["event"] == "start"
    assert lines[1]["val"] == "abc"
