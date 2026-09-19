"""Turns the rows of a run into one summary for each mode.

The rule for failures is here. A request that failed, or that gave no
visible answer, has no latency of its own. It gets the timeout as its
latency, so that it moves the p95 up and never improves it by being absent.
"""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from modebench.config import StatsConfig
from modebench.hashing import stable_seed
from modebench.stats.percentiles import Estimate, bootstrap_mean, bootstrap_percentile, percentile
from modebench.storage.db import RunStore
from modebench.storage.records import RequestRecord, RunRecord, ScoreRecord, StoredRequest

SUMMARY_NAME = "summary.json"


class ModeSummary(BaseModel):
    """The aggregate of one mode in one run. Latencies are of the cold, single requests."""

    model_config = ConfigDict(frozen=True)

    mode_id: str
    requests: int
    errors: int
    error_rate: float
    ttft_p50_ms: Estimate
    ttft_p95_ms: Estimate
    ttfat_p50_ms: Estimate
    ttfat_p95_ms: Estimate
    total_p50_ms: Estimate
    total_p95_ms: Estimate
    quality: Estimate | None = None
    question_accuracy: Estimate | None = None
    utility_mean: float | None = None
    false_refusal_rate: float | None = None
    false_acceptance_rate: float | None = None
    empty_rate: float = 0.0
    preamble_rate: float | None = None
    judge_failures: int = 0
    cost_mean_usd: float | None = None
    cost_total_usd: float = 0.0
    tok_per_s_p50: float | None = None
    reasoning_tokens_mean: float | None = None
    answer_tokens_mean: float | None = None
    served_by: dict[str, int] = Field(default_factory=dict)
    meeting_cold_ttfat_p50_ms: Estimate | None = None
    meeting_warm_ttfat_p50_ms: Estimate | None = None
    meeting_warm_ttfat_p95_ms: Estimate | None = None
    meeting_warm_cached_share: float | None = None


class RunSummary(BaseModel):
    """The summary of a run. It holds numbers and identifiers, and no transcript text."""

    model_config = ConfigDict(frozen=True)

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
    modes: list[ModeSummary]

    def mode(self, mode_id: str) -> ModeSummary | None:
        """Return the summary of one mode, or None."""
        for summary in self.modes:
            if summary.mode_id == mode_id:
                return summary
        return None


def effective_latency(record: RequestRecord, value: float | None, timeout_ms: float) -> float:
    """Return the latency that a request counts for: its own, or the timeout if it failed."""
    if not record.ok or value is None:
        return timeout_ms
    return min(value, timeout_ms)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _score(scores: dict[str, ScoreRecord], key: str) -> float | None:
    record = scores.get(key)
    return None if record is None else record.value


def _rate(flags: Sequence[float]) -> float | None:
    return _mean(flags)


class _Bootstrap:
    """The bootstrap settings, with one seed for each mode and metric."""

    def __init__(self, config: StatsConfig, mode_id: str) -> None:
        self._config = config
        self._mode_id = mode_id

    def _seed(self, metric: str) -> int:
        return stable_seed(self._config.seed, self._mode_id, metric) % (2**32)

    def pct(self, values: Sequence[float], q: float, metric: str) -> Estimate:
        return bootstrap_percentile(
            values,
            q,
            resamples=self._config.bootstrap_resamples,
            confidence=self._config.confidence,
            seed=self._seed(metric),
        )

    def mean(self, values: Sequence[float], metric: str) -> Estimate | None:
        if not values:
            return None
        return bootstrap_mean(
            values,
            resamples=self._config.bootstrap_resamples,
            confidence=self._config.confidence,
            seed=self._seed(metric),
        )

    def pct_or_none(self, values: Sequence[float], q: float, metric: str) -> Estimate | None:
        return self.pct(values, q, metric) if values else None


def summarize_mode(
    mode_id: str,
    requests: Sequence[StoredRequest],
    scores: dict[int, dict[str, ScoreRecord]],
    timeout_ms: float,
    config: StatsConfig,
) -> ModeSummary:
    """Return the summary of one mode from its measured requests."""
    boot = _Bootstrap(config, mode_id)
    singles = [item for item in requests if item.record.suite == "single"]
    main = singles if singles else list(requests)
    records = [item.record for item in main]
    ttft = [effective_latency(record, record.ttft_ms, timeout_ms) for record in records]
    ttfat = [effective_latency(record, record.ttfat_ms, timeout_ms) for record in records]
    total = [effective_latency(record, record.total_ms, timeout_ms) for record in records]
    errors = sum(1 for record in records if not record.ok)
    quality: list[float] = []
    accuracy: list[float] = []
    utility: list[float] = []
    false_refusal: list[float] = []
    false_acceptance: list[float] = []
    preamble: list[float] = []
    judge_failures = 0
    for item in main:
        row = scores.get(item.id, {})
        value = _score(row, "derived.quality")
        if value is not None:
            quality.append(value)
        correct = _score(row, "derived.question_correct")
        if correct is not None:
            accuracy.append(correct)
        graded = _score(row, "judge.utility")
        if graded is not None and not item.record.expect_refusal:
            utility.append((graded - 1.0) / 4.0)
        if "judge.failure" in row:
            judge_failures += 1
        refused_wrongly = _score(row, "derived.false_refusal")
        accepted_wrongly = _score(row, "derived.false_acceptance")
        if item.record.ok and item.record.expect_refusal and accepted_wrongly is not None:
            false_acceptance.append(accepted_wrongly)
        if item.record.ok and not item.record.expect_refusal and refused_wrongly is not None:
            false_refusal.append(refused_wrongly)
        clean = _score(row, "deterministic.no_preamble")
        if item.record.ok and clean is not None:
            preamble.append(1.0 - clean)
    costs: list[float] = []
    speeds: list[float] = []
    reasoning: list[float] = []
    answers: list[float] = []
    served: Counter[str] = Counter()
    for record in records:
        if record.cost_usd is not None:
            costs.append(record.cost_usd)
        if record.tok_per_s is not None:
            speeds.append(record.tok_per_s)
        if record.reasoning_tokens is not None:
            reasoning.append(float(record.reasoning_tokens))
        if record.answer_tokens is not None:
            answers.append(float(record.answer_tokens))
        if record.served_by:
            served[record.served_by] += 1
    empties = sum(1 for record in records if record.error_kind == "empty_output")
    meeting = [item.record for item in requests if item.record.suite == "meeting"]
    cold = [
        effective_latency(record, record.ttfat_ms, timeout_ms)
        for record in meeting
        if record.cache_state == "cold"
    ]
    warm_records = [record for record in meeting if record.cache_state == "warm"]
    warm = [effective_latency(record, record.ttfat_ms, timeout_ms) for record in warm_records]
    cached_shares: list[float] = []
    for record in warm_records:
        if record.cached_tokens is not None and record.prompt_tokens:
            cached_shares.append(record.cached_tokens / record.prompt_tokens)
    return ModeSummary(
        mode_id=mode_id,
        requests=len(records),
        errors=errors,
        error_rate=errors / len(records) if records else 0.0,
        ttft_p50_ms=boot.pct(ttft, 50, "ttft_p50"),
        ttft_p95_ms=boot.pct(ttft, 95, "ttft_p95"),
        ttfat_p50_ms=boot.pct(ttfat, 50, "ttfat_p50"),
        ttfat_p95_ms=boot.pct(ttfat, 95, "ttfat_p95"),
        total_p50_ms=boot.pct(total, 50, "total_p50"),
        total_p95_ms=boot.pct(total, 95, "total_p95"),
        quality=boot.mean(quality, "quality"),
        question_accuracy=boot.mean(accuracy, "question_accuracy"),
        utility_mean=_mean(utility),
        false_refusal_rate=_rate(false_refusal),
        false_acceptance_rate=_rate(false_acceptance),
        empty_rate=empties / len(records) if records else 0.0,
        preamble_rate=_rate(preamble),
        judge_failures=judge_failures,
        cost_mean_usd=_mean(costs),
        cost_total_usd=sum(costs),
        tok_per_s_p50=percentile(speeds, 50) if speeds else None,
        reasoning_tokens_mean=_mean(reasoning),
        answer_tokens_mean=_mean(answers),
        served_by=dict(served),
        meeting_cold_ttfat_p50_ms=boot.pct_or_none(cold, 50, "meeting_cold_p50"),
        meeting_warm_ttfat_p50_ms=boot.pct_or_none(warm, 50, "meeting_warm_p50"),
        meeting_warm_ttfat_p95_ms=boot.pct_or_none(warm, 95, "meeting_warm_p95"),
        meeting_warm_cached_share=_mean(cached_shares),
    )


def summarize_run(store: RunStore, run_id: str, config: StatsConfig) -> RunSummary:
    """Return the summary of a run from the database."""
    run: RunRecord = store.load_run(run_id)
    requests = store.load_requests(run_id)
    scores = store.load_scores(run_id)
    by_mode: dict[str, list[StoredRequest]] = {}
    for item in requests:
        by_mode.setdefault(item.record.mode_id, []).append(item)
    timeout_ms = run.timeout_s * 1000.0
    modes = [
        summarize_mode(mode_id, items, scores, timeout_ms, config)
        for mode_id, items in by_mode.items()
    ]
    return RunSummary(
        run_id=run.run_id,
        suite=run.suite,
        profile=run.profile,
        started_at=run.started_at,
        status=run.status,
        dry_run=run.dry_run,
        git_sha=run.git_sha,
        git_dirty=run.git_dirty,
        config_hash=run.config_hash,
        dataset_hash=run.dataset_hash,
        prompt_hash=run.prompt_hash,
        timeout_s=run.timeout_s,
        judge_model=run.judge_model,
        modes=modes,
    )


def write_summary(summary: RunSummary, path: Path) -> Path:
    """Write the summary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_summary(path: Path) -> RunSummary:
    """Read a summary that `write_summary` wrote. A baseline is such a file."""
    return RunSummary.model_validate_json(path.read_text(encoding="utf-8"))
