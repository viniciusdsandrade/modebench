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
    assert len(token1) == 8

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
    store.create_run(run_record)

    fetched_run = store.load_run("run-test-01")
    assert fetched_run is not None
    assert fetched_run.run_id == "run-test-01"
    assert fetched_run.profile == "smoke"
    assert store.latest_run_id("analyze") == "run-test-01"

    req = RequestRecord(
        run_id="run-test-01",
        seq=1,
        mode_id="m1",
        suite="analyze",
        item_id="item-01",
        case_id="case-01",
        variant_id="var-01",
        variant_kind="truncation",
        truncation_pct=100,
        asr_wer=0.0,
        noise_kind=None,
        duration_min=5,
        repetition=1,
        warmup=False,
        scenario_id=None,
        click_index=None,
        cache_state="cold",
        expect_refusal=False,
        private=False,
        started_at="2026-09-01T12:00:01Z",
        ok=True,
        error_kind=None,
        error_message=None,
        http_status=200,
        first_chunk_ms=200.0,
        ttft_ms=300.0,
        first_reasoning_ms=None,
        ttfat_ms=650.0,
        total_ms=1200.0,
        prompt_tokens=100,
        completion_tokens=50,
        reasoning_tokens=None,
        answer_tokens=50,
        cached_tokens=None,
        tok_per_s=50.0,
        served_by="local",
        cost_usd=0.0002,
        cost_source="usage",
        prompt_chars=400,
        prompt_sha="sha256-prompt",
        answer="Resposta completa.",
    )
    req_id = store.insert_request(req)

    requests = store.load_requests("run-test-01")
    assert len(requests) == 1
    assert requests[0].id == req_id
    assert requests[0].record.item_id == "item-01"
    assert requests[0].record.ttfat_ms == 650.0

    score = ScoreRecord(
        scorer="deterministic",
        name="non_empty",
        value=1.0,
        detail="true",
    )
    store.insert_scores(req_id, [score])

    scores = store.load_scores("run-test-01")
    assert len(scores) == 1
    assert req_id in scores
    assert "deterministic.non_empty" in scores[req_id]
    assert scores[req_id]["deterministic.non_empty"].value == 1.0

    # Test JSONL writer and reader
    jsonl_path = tmp_path / "events.jsonl"
    with JsonlWriter(jsonl_path) as writer:
        writer.write({"event": "start", "seq": 1})
        writer.write({"event": "token", "val": "abc"})

    lines = list(read_jsonl(jsonl_path))
    assert len(lines) == 2
    assert lines[0]["event"] == "start"
    assert lines[1]["val"] == "abc"
