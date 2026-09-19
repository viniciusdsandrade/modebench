"""The rows of the database, as dataclasses."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One run. The hashes and the commit make the run reproducible."""

    run_id: str
    suite: str
    profile: str
    started_at: str
    status: str
    dry_run: bool
    git_sha: str
    git_dirty: bool
    config_hash: str
    dataset_hash: str
    prompt_hash: str
    timeout_s: float
    judge_model: str
    modes_json: str
    estimated_cost_usd: float | None = None
    actual_cost_usd: float | None = None
    finished_at: str | None = None


@dataclass(frozen=True, slots=True)
class RequestRecord:
    """One request of the Analyze suites, with its conditions and its measurements."""

    run_id: str
    seq: int
    mode_id: str
    suite: str
    item_id: str
    case_id: str
    variant_id: str
    variant_kind: str
    truncation_pct: int
    asr_wer: float
    noise_kind: str | None
    duration_min: int
    repetition: int
    warmup: bool
    scenario_id: str | None
    click_index: int | None
    cache_state: str
    expect_refusal: bool
    private: bool
    started_at: str
    ok: bool
    error_kind: str | None
    error_message: str | None
    http_status: int | None
    first_chunk_ms: float | None
    ttft_ms: float | None
    first_reasoning_ms: float | None
    ttfat_ms: float | None
    total_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    answer_tokens: int | None
    cached_tokens: int | None
    tok_per_s: float | None
    served_by: str | None
    cost_usd: float | None
    cost_source: str
    prompt_chars: int
    prompt_sha: str
    answer: str


@dataclass(frozen=True, slots=True)
class StoredRequest:
    """A request row with its database identifier."""

    id: int
    record: RequestRecord


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    """One score of one request. `scorer` is `deterministic` or `judge`."""

    scorer: str
    name: str
    value: float | None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class SttSessionRecord:
    """One replay of one audio file through one speech to text provider."""

    run_id: str
    provider: str
    audio_id: str
    repetition: int
    started_at: str
    ok: bool
    error_message: str | None
    audio_ms: float
    session_ms: float
    connect_ms: float | None
    first_partial_ms: float | None
    first_partial_after_speech_ms: float | None
    final_latency_p50_ms: float | None
    final_latency_p95_ms: float | None
    finals: int
    wer: float | None
    cer: float | None
    speaker_accuracy: float | None
    max_send_lag_ms: float
    cost_usd: float | None
    hypothesis: str


@dataclass(frozen=True, slots=True)
class SttFinalRecord:
    """One settled utterance of one replay."""

    utterance_index: int
    text: str
    speaker: str | None
    received_ms: float
    audio_end_ms: float | None
    latency_ms: float | None
