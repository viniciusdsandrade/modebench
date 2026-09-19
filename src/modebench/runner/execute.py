"""Prepares and executes a run of the Analyze suites.

`prepare_run` reads the inputs, makes the plan, and applies the guards. It
sends nothing to a chat API. `execute_run` sends the requests one at a time
(concurrency is one), writes each result when it arrives, and then scores the
answers. A request that fails is a row like any other: it counts for the
error rate and it is never dropped.
"""

import json
import random
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from modebench.config import (
    BenchConfig,
    Mode,
    ModesFile,
    Price,
    Profile,
    ProviderConfig,
    resolve_path,
)
from modebench.dataset.loader import LoadedDataset, load_dataset, load_filler
from modebench.dataset.transcript import NOTHING_HERE, Labels, render_lines
from modebench.dataset.variants import Variant, build_variants
from modebench.errors import ConfigError, PrivacyViolation
from modebench.hashing import git_dirty, git_sha, sha256_files, sha256_text, stable_seed
from modebench.providers.base import ChatProvider, StreamOutcome
from modebench.providers.metrics import DerivedMetrics, derive_metrics
from modebench.providers.registry import build_providers, effective_provider_config
from modebench.quality.deterministic import score_answer, to_records
from modebench.quality.judge import FakeJudge, Judge, JudgeItem, LlmJudge, verdict_records
from modebench.quality.metrics import quality_records, request_quality
from modebench.runner.cost import JUDGE_ID, CostEstimate, estimate_cost
from modebench.runner.guard import (
    check_running_cost,
    enforce_cost_ceiling,
    enforce_privacy,
    privacy_problems,
)
from modebench.runner.plan import PlannedRequest, WorkItem, build_items, build_plan, render_request
from modebench.runner.preflight import run_preflight
from modebench.storage.db import RunStore
from modebench.storage.jsonl import RAW_NAME, RawLog
from modebench.storage.records import RequestRecord, RunRecord, ScoreRecord

SUITE = "analyze"
STATUS_COMPLETED = "completed"
STATUS_COST_STOP = "stopped_at_cost_ceiling"
Progress = Callable[[int, int, RequestRecord], None]


@dataclass(frozen=True, slots=True)
class RunSettings:
    """What the operator asked for."""

    root: Path
    bench: BenchConfig
    modes_file: ModesFile
    modes: tuple[Mode, ...]
    profile_name: str
    config_paths: tuple[Path, ...] = ()
    dry_run: bool = False
    judge_enabled: bool = True
    max_cost_usd: float | None = None
    skip_preflight: bool = False


@dataclass(frozen=True, slots=True)
class PreparedRun:
    """A run that passed the privacy guard and has a cost estimate."""

    settings: RunSettings
    profile: Profile
    datasets: list[LoadedDataset]
    variants: list[Variant]
    items: list[WorkItem]
    plan: list[PlannedRequest]
    preprompt: str
    judge_prompt: str
    labels: Labels
    prices: dict[str, Price | None]
    estimate: CostEstimate
    ceiling_usd: float
    judge_active: bool
    config_hash: str
    dataset_hash: str
    prompt_hash: str

    @property
    def has_private_data(self) -> bool:
        """Return True if a variant of the run comes from a private dataset."""
        return any(variant.private for variant in self.variants)


def utc_now() -> datetime:
    """Return the present time in UTC."""
    return datetime.now(UTC)


def new_run_id(moment: datetime) -> str:
    """Return a run identifier that sorts by time and does not collide."""
    return f"{moment.strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"


def _read_prompt(paths: list[Path]) -> str:
    try:
        texts = [path.read_text(encoding="utf-8").strip() for path in paths]
    except FileNotFoundError as exc:
        raise ConfigError(f"prompt file not found: {exc.filename}") from exc
    return "\n\n".join(text for text in texts if text)


def _routes(settings: RunSettings, judge_active: bool) -> list[tuple[Mode, ProviderConfig]]:
    """Return each mode that sees transcript text, the judge included, with its provider."""
    modes = list(settings.modes)
    if judge_active:
        modes.append(settings.bench.judge.mode)
    return [
        (mode, effective_provider_config(settings.modes_file, mode, dry_run=settings.dry_run))
        for mode in modes
    ]


def _check_judge(settings: RunSettings) -> None:
    judge = settings.bench.judge.mode
    if judge.provider not in settings.modes_file.providers:
        raise ConfigError(f"the judge names the unknown provider {judge.provider}")
    family = judge.resolved_family()
    same = [mode.id for mode in settings.modes if mode.resolved_family() == family]
    if same:
        raise ConfigError(
            f"the judge is of the family {family}, and so are these modes: {', '.join(same)}. "
            "A judge must be of a different family from each mode that it grades."
        )


def prepare_run(
    settings: RunSettings, env: Mapping[str, str], client: httpx.Client | None = None
) -> PreparedRun:
    """Read the inputs, make the plan, apply the privacy guard and estimate the cost."""
    bench = settings.bench
    root = settings.root
    profile = bench.profile(settings.profile_name)
    if not settings.modes:
        raise ConfigError("no mode is selected")
    dataset_paths = [resolve_path(root, path) for path in bench.paths.datasets]
    datasets = [load_dataset(root, path) for path in dataset_paths]
    filler_path = resolve_path(root, bench.paths.filler)
    filler = load_filler(filler_path)
    seed = bench.execution.seed
    variants = build_variants(datasets, profile, seed)
    items = build_items(variants, profile, filler, bench.transcript, bench.meeting, seed)
    mode_ids = [mode.id for mode in settings.modes]
    plan = build_plan(items, mode_ids, profile.repetitions, profile.warmup)
    prompt_paths = [resolve_path(root, path) for path in bench.paths.preprompt_paths]
    preprompt = _read_prompt(prompt_paths)
    judge_active = settings.judge_enabled and bench.judge.enabled
    judge_prompt = ""
    if judge_active:
        _check_judge(settings)
        judge_prompt = _read_prompt([resolve_path(root, bench.paths.judge_prompt_path)])
    has_private = any(variant.private for variant in variants)
    enforce_privacy(has_private, _routes(settings, judge_active))
    prices: dict[str, Price | None] = {mode.id: mode.price for mode in settings.modes}
    if judge_active:
        prices[JUDGE_ID] = bench.judge.mode.price
    if not settings.dry_run and not settings.skip_preflight:
        checked = [mode for mode, _ in _routes(settings, judge_active)]
        preflight_client = client if client is not None else httpx.Client()
        try:
            report = run_preflight(settings.modes_file, checked, env, preflight_client)
        finally:
            if client is None:
                preflight_client.close()
        for mode in settings.modes:
            if prices[mode.id] is None:
                prices[mode.id] = report.prices.get(mode.id)
        if judge_active and prices[JUDGE_ID] is None:
            prices[JUDGE_ID] = report.prices.get(bench.judge.mode.id)
    labels = Labels.from_config(bench.transcript)
    estimate = estimate_cost(
        plan,
        {mode.id: mode for mode in settings.modes},
        prices,
        preprompt,
        labels,
        bench.cost,
        judge_prompt_chars=len(judge_prompt) if judge_active else None,
    )
    ceiling = settings.max_cost_usd if settings.max_cost_usd is not None else profile.max_cost_usd
    return PreparedRun(
        settings=settings,
        profile=profile,
        datasets=datasets,
        variants=variants,
        items=items,
        plan=plan,
        preprompt=preprompt,
        judge_prompt=judge_prompt,
        labels=labels,
        prices=prices,
        estimate=estimate,
        ceiling_usd=ceiling,
        judge_active=judge_active,
        config_hash=sha256_files(settings.config_paths),
        dataset_hash=sha256_files([*dataset_paths, filler_path]),
        prompt_hash=sha256_text(preprompt + "\x1f" + judge_prompt),
    )


def _request_record(
    run_id: str,
    planned: PlannedRequest,
    started_at: str,
    prompt_text: str,
    outcome: StreamOutcome,
    metrics: DerivedMetrics,
) -> RequestRecord:
    item = planned.item
    variant = item.variant
    return RequestRecord(
        run_id=run_id,
        seq=planned.seq,
        mode_id=planned.mode_id,
        suite=item.suite,
        item_id=item.item_id,
        case_id=variant.case_id,
        variant_id=variant.variant_id,
        variant_kind=variant.kind,
        truncation_pct=variant.truncation_pct,
        asr_wer=variant.asr_wer,
        noise_kind=variant.noise_kind,
        duration_min=item.duration_min,
        repetition=planned.repetition,
        warmup=planned.warmup,
        scenario_id=item.scenario_id,
        click_index=item.click_index,
        cache_state=item.cache_state,
        expect_refusal=variant.expect_refusal,
        private=variant.private,
        started_at=started_at,
        ok=outcome.ok,
        error_kind=outcome.error_kind,
        error_message=outcome.error_message,
        http_status=outcome.http_status,
        first_chunk_ms=outcome.first_chunk_ms,
        ttft_ms=outcome.ttft_ms,
        first_reasoning_ms=outcome.first_reasoning_ms,
        ttfat_ms=outcome.ttfat_ms,
        total_ms=outcome.total_ms,
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
        reasoning_tokens=metrics.reasoning_tokens,
        answer_tokens=metrics.answer_tokens,
        cached_tokens=outcome.cached_tokens,
        tok_per_s=metrics.tok_per_s,
        served_by=outcome.served_by,
        cost_usd=metrics.cost_usd,
        cost_source=metrics.cost_source,
        prompt_chars=len(prompt_text),
        prompt_sha=sha256_text(prompt_text),
        answer=outcome.answer,
    )


def _raw_request(record: RequestRecord, request_id: int, outcome: StreamOutcome) -> dict[str, object]:
    return {
        "type": "request",
        "request_id": request_id,
        "run_id": record.run_id,
        "seq": record.seq,
        "mode_id": record.mode_id,
        "item_id": record.item_id,
        "repetition": record.repetition,
        "warmup": record.warmup,
        "ok": record.ok,
        "error_kind": record.error_kind,
        "error_message": record.error_message,
        "http_status": record.http_status,
        "ttft_ms": record.ttft_ms,
        "ttfat_ms": record.ttfat_ms,
        "total_ms": record.total_ms,
        "usage": {
            "prompt_tokens": outcome.prompt_tokens,
            "completion_tokens": outcome.completion_tokens,
            "reasoning_tokens": outcome.reasoning_tokens,
            "cached_tokens": outcome.cached_tokens,
            "total_tokens": outcome.total_tokens,
            "reported_cost_usd": outcome.reported_cost_usd,
        },
        "served_by": record.served_by,
        "malformed_chunks": outcome.malformed_chunks,
        "reasoning_chars": outcome.reasoning_chars,
        "events": [[event.offset_ms, event.kind, event.chars] for event in outcome.events],
        "prompt_sha": record.prompt_sha,
        "answer": record.answer,
    }


def _make_judge(
    prepared: PreparedRun, env: Mapping[str, str], client: httpx.Client | None
) -> Judge | None:
    settings = prepared.settings
    if not prepared.judge_active:
        return None
    if settings.dry_run:
        return FakeJudge()
    judge_mode = settings.bench.judge.mode
    providers = build_providers(settings.modes_file, [judge_mode], env, dry_run=False, client=client)
    return LlmJudge(
        providers[judge_mode.id],
        settings.modes_file.providers[judge_mode.provider],
        judge_mode,
        prepared.judge_prompt,
        price=prepared.prices.get(JUDGE_ID),
        max_attempts=settings.bench.judge.max_attempts,
        timeout_s=settings.bench.execution.timeout_s,
    )


def _judge_model_name(prepared: PreparedRun) -> str:
    if not prepared.judge_active:
        return ""
    if prepared.settings.dry_run:
        return "fake"
    judge_mode = prepared.settings.bench.judge.mode
    return f"{judge_mode.provider}:{judge_mode.model}"


def _score_request(
    record: RequestRecord,
    planned: PlannedRequest,
    prepared: PreparedRun,
    judge: Judge | None,
) -> tuple[list[ScoreRecord], float]:
    """Return the score rows of one measured request, and what the judge cost."""
    variant = planned.item.variant
    fixed = score_answer(record.answer, expect_refusal=variant.expect_refusal)
    rows = to_records(fixed)
    verdict = None
    judge_cost = 0.0
    if judge is not None and record.ok and fixed.non_empty:
        fresh_text = render_lines(variant.fresh, prepared.labels) if variant.fresh else NOTHING_HERE
        result = judge.judge(
            JudgeItem(
                fresh_text=fresh_text,
                expect_refusal=variant.expect_refusal,
                gold_question=variant.gold_question,
                key_points=variant.key_points,
                reference_answer=variant.reference_answer,
                answer=record.answer,
            )
        )
        judge_cost = result.cost_usd
        verdict = result.verdict
        if verdict is not None:
            rows.extend(verdict_records(verdict, len(variant.key_points)))
        else:
            rows.append(ScoreRecord("judge", "failure", 1.0, (result.error or "")[:500]))
    quality = request_quality(
        ok=record.ok,
        expect_refusal=variant.expect_refusal,
        key_points=len(variant.key_points),
        deterministic=fixed,
        verdict=verdict,
        weights=prepared.settings.bench.quality_weights,
    )
    rows.extend(quality_records(quality))
    return rows, judge_cost


def execute_run(
    prepared: PreparedRun,
    store: RunStore,
    env: Mapping[str, str],
    *,
    client: httpx.Client | None = None,
    providers: Mapping[str, ChatProvider] | None = None,
    judge: Judge | None = None,
    progress: Progress | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = utc_now,
) -> str:
    """Execute a prepared run and return its identifier.

    The two guards run again here. A real run stops when the money spent goes
    above the ceiling. The requests that it made stay in the database, and the
    status of the run says that it is not complete.
    """
    settings = prepared.settings
    bench = settings.bench
    route_list = _routes(settings, prepared.judge_active)
    routes = {mode.id: provider for mode, provider in route_list}
    enforce_privacy(prepared.has_private_data, route_list)
    if not settings.dry_run:
        enforce_cost_ceiling(
            prepared.estimate.total_usd,
            prepared.ceiling_usd,
            prepared.estimate.unknown_price_modes,
        )
    modes = {mode.id: mode for mode in settings.modes}
    chat = providers if providers is not None else build_providers(
        settings.modes_file, settings.modes, env, dry_run=settings.dry_run, client=client
    )
    active_judge = judge if judge is not None else _make_judge(prepared, env, client)
    run_id = new_run_id(now())
    store.create_run(
        RunRecord(
            run_id=run_id,
            suite=SUITE,
            profile=settings.profile_name,
            started_at=now().isoformat(),
            status="running",
            dry_run=settings.dry_run,
            git_sha=git_sha(settings.root),
            git_dirty=git_dirty(settings.root),
            config_hash=prepared.config_hash,
            dataset_hash=prepared.dataset_hash,
            prompt_hash=prepared.prompt_hash,
            timeout_s=bench.execution.timeout_s,
            judge_model=_judge_model_name(prepared),
            modes_json=json.dumps(
                [mode.model_dump(mode="json") for mode in settings.modes], ensure_ascii=False
            ),
            estimated_cost_usd=prepared.estimate.total_usd,
        )
    )
    runs_dir = resolve_path(settings.root, bench.paths.runs_dir)
    spent = 0.0
    status = STATUS_COMPLETED
    measured: list[tuple[int, PlannedRequest, RequestRecord]] = []
    with RawLog(runs_dir / run_id / RAW_NAME) as raw:
        for planned in prepared.plan:
            mode = modes[planned.mode_id]
            if planned.item.variant.private and privacy_problems(mode, routes[mode.id]):
                raise PrivacyViolation(f"{mode.id} must not receive the private item")
            request = render_request(planned, run_id, prepared.preprompt, prepared.labels)
            started_at = now().isoformat()
            outcome = chat[mode.id].stream_chat(mode, request, bench.execution.timeout_s)
            metrics = derive_metrics(outcome, routes[mode.id], prepared.prices.get(mode.id))
            record = _request_record(
                run_id, planned, started_at, request.system + "\x1f" + request.user, outcome, metrics
            )
            request_id = store.insert_request(record)
            raw.write(_raw_request(record, request_id, outcome))
            spent += metrics.cost_usd or 0.0
            if not planned.warmup:
                measured.append((request_id, planned, record))
            if progress is not None:
                progress(planned.seq + 1, len(prepared.plan), record)
            if not settings.dry_run:
                if check_running_cost(spent, prepared.ceiling_usd):
                    status = STATUS_COST_STOP
                    break
                pause = (
                    bench.execution.pause_between_requests_s
                    if outcome.ok
                    else bench.execution.pause_after_error_s
                )
                if pause > 0:
                    sleep(pause)
        order = list(range(len(measured)))
        random.Random(stable_seed(bench.execution.seed, "judge-order", run_id)).shuffle(order)
        for index in order:
            request_id, planned, record = measured[index]
            stopped = status == STATUS_COST_STOP
            rows, judge_cost = _score_request(
                record, planned, prepared, None if stopped else active_judge
            )
            store.insert_scores(request_id, rows)
            raw.write(
                {
                    "type": "scores",
                    "request_id": request_id,
                    "scores": [[row.scorer, row.name, row.value, row.detail] for row in rows],
                }
            )
            spent += judge_cost
            if not settings.dry_run and check_running_cost(spent, prepared.ceiling_usd):
                status = STATUS_COST_STOP
    store.finish_run(run_id, status, now().isoformat(), spent)
    return run_id
