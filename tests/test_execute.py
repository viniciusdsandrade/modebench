"""The executor of the Analyze suites: the life of a run, the cost ceiling and the guards."""

import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from helpers import REPO_ROOT, bench_for, fake_mode, local_modes_file, openrouter_mode
from modebench.config import BenchConfig, Mode, Price
from modebench.decision.decide import decide, modes_of_run
from modebench.errors import ConfigError, CostCeilingExceeded, PrivacyViolation
from modebench.providers.base import ChatRequest, StreamOutcome
from modebench.providers.fake import FakeProvider
from modebench.runner.execute import (
    STATUS_COMPLETED,
    STATUS_COST_STOP,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    PreparedRun,
    RunSettings,
    execute_run,
    prepare_run,
)
from modebench.stats.aggregate import summarize_run
from modebench.storage.db import DB_NAME, RunStore

PRIVATE_DATASET = """\
version: 1
visibility: private
cases:
  - id: private-case-01
    context:
      - speaker: system
        text: "O contrato do cliente vence em marco."
    question:
      speaker: mic
      text: "Quando vence o contrato do cliente?"
    key_points:
      - "marco"
"""


def settings_for(
    bench: BenchConfig,
    tmp_path: Path,
    modes: list[Mode],
    *,
    dry_run: bool = True,
    judge_enabled: bool = True,
    max_cost_usd: float | None = None,
    **path_updates: object,
) -> RunSettings:
    priced_judge = bench.judge.mode.model_copy(
        update={"price": Price(usd_per_mtok_in=1.0, usd_per_mtok_out=1.0)}
    )
    local = bench_for(bench, tmp_path / "runs", **path_updates)
    local = local.model_copy(
        update={"judge": local.judge.model_copy(update={"mode": priced_judge})}
    )
    return RunSettings(
        root=REPO_ROOT,
        bench=local,
        modes_file=local_modes_file(*modes),
        modes=tuple(modes),
        profile_name="smoke",
        dry_run=dry_run,
        judge_enabled=judge_enabled,
        max_cost_usd=max_cost_usd,
        skip_preflight=True,
    )


def open_store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "runs" / DB_NAME)


def no_sleep(seconds: float) -> None:
    return None


def test_a_dry_run_completes_and_grades_each_measured_request(
    seed_bench: BenchConfig, tmp_path: Path
) -> None:
    modes = [fake_mode("fast", ttft_ms=100, quality=1.0), fake_mode("slow", ttft_ms=900)]
    prepared = prepare_run(settings_for(seed_bench, tmp_path, modes), {})
    assert prepared.judge_active
    assert not prepared.has_private_data
    store = open_store(tmp_path)
    try:
        run_id = execute_run(prepared, store, {}, sleep=no_sleep)
        run = store.load_run(run_id)
        assert run.status == STATUS_COMPLETED
        assert run.finished_at is not None
        assert run.judge_model == "fake"
        everything = store.load_requests(run_id, include_warmup=True)
        measured = store.load_requests(run_id)
        assert len(everything) == len(prepared.plan)
        assert len(measured) == len([item for item in prepared.plan if not item.warmup])
        scores = store.load_scores(run_id)
        assert set(scores) == {item.id for item in measured}
        summary = summarize_run(store, run_id, prepared.settings.bench.stats)
    finally:
        store.close()
    assert {mode.mode_id for mode in summary.modes} == {"fast", "slow"}
    for mode in summary.modes:
        assert mode.quality is not None
        assert mode.judged > 0
        assert mode.unjudged == 0
    raw = tmp_path / "runs" / run_id / "raw.jsonl"
    assert '"new_connection": false' in raw.read_text(encoding="utf-8")


def test_a_run_with_no_judge_has_no_quality_estimate(
    seed_bench: BenchConfig, tmp_path: Path
) -> None:
    modes = [fake_mode("only", quality=1.0)]
    prepared = prepare_run(settings_for(seed_bench, tmp_path, modes, judge_enabled=False), {})
    assert not prepared.judge_active
    store = open_store(tmp_path)
    try:
        run_id = execute_run(prepared, store, {}, sleep=no_sleep)
        summary = summarize_run(store, run_id, prepared.settings.bench.stats)
        record = store.load_run(run_id)
    finally:
        store.close()
    mode = summary.modes[0]
    # The noise case and the failures have a value with no judge. They are no sample of the mode.
    assert mode.quality is None
    assert mode.question_accuracy is None
    assert mode.judged == 0
    assert mode.unjudged > 0
    decisions = decide(seed_bench.roles, summary, modes_of_run(record.modes_json))
    assert all(decision.winner is None for decision in decisions)
    assert all(
        reason == "no quality data (the run had no judge)"
        for decision in decisions
        for reason in decision.rejected.values()
    )


class Exploding:
    """A provider that raises after some requests, as a defect or the operator would."""

    def __init__(self, error: BaseException, after: int = 2) -> None:
        self._error = error
        self._after = after
        self._inner = FakeProvider()
        self.calls = 0

    def stream_chat(self, mode: Mode, request: ChatRequest, timeout_s: float) -> StreamOutcome:
        self.calls += 1
        if self.calls > self._after:
            raise self._error
        return self._inner.stream_chat(mode, request, timeout_s)


@pytest.mark.parametrize(
    ("error", "status"),
    [(RuntimeError("a defect"), STATUS_FAILED), (KeyboardInterrupt(), STATUS_INTERRUPTED)],
)
def test_a_run_that_stops_early_never_stays_running(
    seed_bench: BenchConfig, tmp_path: Path, error: BaseException, status: str
) -> None:
    modes = [fake_mode("only")]
    prepared = prepare_run(settings_for(seed_bench, tmp_path, modes), {})
    store = open_store(tmp_path)
    try:
        with pytest.raises(type(error)):
            execute_run(prepared, store, {}, providers={"only": Exploding(error)}, sleep=no_sleep)
        run = store.load_run(store.latest_run_id("analyze"))
        kept = store.load_requests(run.run_id, include_warmup=True)
    finally:
        store.close()
    assert run.status == status
    assert run.finished_at is not None
    assert len(kept) == 2


class NoUsage:
    """A provider whose answers have no usage block, so their cost is not known."""

    def stream_chat(self, mode: Mode, request: ChatRequest, timeout_s: float) -> StreamOutcome:
        return StreamOutcome(
            ok=True, total_ms=20.0, ttft_ms=5.0, ttfat_ms=5.0, answer="Pergunta: x\nResposta: y."
        )


class Overspends:
    """A provider whose usage gives one and a half times the estimate of a request."""

    def __init__(self, prepared: PreparedRun, mode_id: str) -> None:
        # The price of `fake_mode` is 1 USD for 1M input tokens.
        self._tokens = int(prepared.estimate.per_request_usd(mode_id) * 1.5 * 1_000_000)

    def stream_chat(self, mode: Mode, request: ChatRequest, timeout_s: float) -> StreamOutcome:
        return StreamOutcome(
            ok=True,
            total_ms=20.0,
            ttft_ms=5.0,
            ttfat_ms=5.0,
            answer="Pergunta: x\nResposta: y.",
            prompt_tokens=self._tokens,
            completion_tokens=0,
        )


def test_requests_with_no_usage_count_for_the_cost_ceiling(
    seed_bench: BenchConfig, tmp_path: Path
) -> None:
    modes = [fake_mode("silent"), fake_mode("costly")]
    first = prepare_run(
        settings_for(seed_bench, tmp_path, modes, dry_run=False, judge_enabled=False), {}
    )
    estimate = first.estimate.total_usd
    assert estimate > 0
    # The measured money alone (0.75 of the estimate) stays below this ceiling. With the
    # requests of no known cost at their estimate (1.25 of the estimate), the run stops.
    ceiling = estimate * 1.2
    prepared = prepare_run(
        settings_for(
            seed_bench, tmp_path, modes, dry_run=False, judge_enabled=False, max_cost_usd=ceiling
        ),
        {},
    )
    providers = {"silent": NoUsage(), "costly": Overspends(prepared, "costly")}
    store = open_store(tmp_path)
    try:
        run_id = execute_run(prepared, store, {}, providers=providers, sleep=no_sleep)
        run = store.load_run(run_id)
    finally:
        store.close()
    assert run.status == STATUS_COST_STOP
    assert run.actual_cost_usd is not None
    assert 0 < run.actual_cost_usd < ceiling


def test_an_estimate_above_the_ceiling_stops_a_real_run_before_it_starts(
    seed_bench: BenchConfig, tmp_path: Path
) -> None:
    modes = [fake_mode("only")]
    prepared = prepare_run(
        settings_for(
            seed_bench, tmp_path, modes, dry_run=False, judge_enabled=False, max_cost_usd=1e-9
        ),
        {},
    )
    store = open_store(tmp_path)
    try:
        with pytest.raises(CostCeilingExceeded, match="ceiling"):
            execute_run(prepared, store, {}, providers={"only": NoUsage()}, sleep=no_sleep)
        with pytest.raises(Exception, match="no run"):
            store.latest_run_id("analyze")
    finally:
        store.close()


def test_the_judge_identifier_is_reserved(seed_bench: BenchConfig, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="identifier judge"):
        prepare_run(settings_for(seed_bench, tmp_path, [fake_mode("judge")]), {})
    # With no judge there is no judge row in the estimate, and the name is free.
    free = settings_for(seed_bench, tmp_path, [fake_mode("judge")], judge_enabled=False)
    assert prepare_run(free, {}).estimate.requests > 0


def test_the_judge_must_be_of_another_family(seed_bench: BenchConfig, tmp_path: Path) -> None:
    same_family = fake_mode("claude-like").model_copy(update={"family": "anthropic"})
    with pytest.raises(ConfigError, match="different family"):
        prepare_run(settings_for(seed_bench, tmp_path, [same_family]), {})


def test_no_mode_and_an_unknown_profile_are_config_errors(
    seed_bench: BenchConfig, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match="no mode"):
        prepare_run(settings_for(seed_bench, tmp_path, []), {})
    settings = settings_for(seed_bench, tmp_path, [fake_mode("only")])
    unknown = RunSettings(
        root=settings.root,
        bench=settings.bench,
        modes_file=settings.modes_file,
        modes=settings.modes,
        profile_name="absent",
    )
    with pytest.raises(ConfigError, match="unknown profile"):
        prepare_run(unknown, {})


def test_private_data_stops_a_real_run_with_an_unsafe_route(
    seed_bench: BenchConfig, tmp_path: Path
) -> None:
    dataset = tmp_path / "sessions.yaml"
    dataset.write_text(PRIVATE_DATASET, encoding="utf-8")
    unsafe = [openrouter_mode("unsafe")]
    real = settings_for(
        seed_bench, tmp_path, unsafe, dry_run=False, judge_enabled=False, datasets=[dataset]
    )
    with pytest.raises(PrivacyViolation, match="data_collection"):
        prepare_run(real, {})
    # A dry run sends nothing to a network, so the same selection is permitted.
    dry = settings_for(seed_bench, tmp_path, unsafe, judge_enabled=False, datasets=[dataset])
    prepared = prepare_run(dry, {})
    assert prepared.has_private_data


def sse(*payloads: dict[str, object]) -> bytes:
    lines = [b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n" for payload in payloads]
    return b"".join(lines) + b"data: [DONE]\n\n"


USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cost": 0.0005}
JUDGE_MODEL = "anthropic/claude-sonnet-5"


def network_settings(
    bench: BenchConfig, tmp_path: Path, *, skip_preflight: bool, judge_price: Price | None
) -> RunSettings:
    # With no preflight, the price must come from the modes file.
    candidate = openrouter_mode(
        "candidate", provider={"data_collection": "deny"}, reasoning={"effort": "none"}
    ).model_copy(update={"family": "google", "price": judge_price})
    base = settings_for(bench, tmp_path, [candidate], dry_run=False)
    judge_mode = base.bench.judge.mode.model_copy(update={"price": judge_price})
    local = base.bench.model_copy(
        update={"judge": base.bench.judge.model_copy(update={"mode": judge_mode})}
    )
    return RunSettings(
        root=base.root,
        bench=local,
        modes_file=base.modes_file,
        modes=base.modes,
        profile_name="smoke",
        skip_preflight=skip_preflight,
    )


def test_the_preflight_of_a_real_run_gives_the_prices_that_the_modes_file_does_not_have(
    seed_bench: BenchConfig, tmp_path: Path
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/models"
        pricing = {"prompt": "0.000001", "completion": "0.000002"}
        entries = [
            {"id": "google/gemini-test", "pricing": pricing},
            {"id": JUDGE_MODEL, "pricing": pricing},
        ]
        return httpx.Response(200, json={"data": entries})

    settings = network_settings(seed_bench, tmp_path, skip_preflight=False, judge_price=None)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        prepared = prepare_run(settings, {"OPENROUTER_API_KEY": "sk-or-123456"}, client)
    assert prepared.prices["candidate"] == Price(usd_per_mtok_in=1.0, usd_per_mtok_out=2.0)
    assert prepared.prices["judge"] == Price(usd_per_mtok_in=1.0, usd_per_mtok_out=2.0)
    assert prepared.estimate.unknown_price_modes == []


def test_a_real_run_sends_each_request_and_records_a_judge_with_no_valid_verdict(
    seed_bench: BenchConfig, tmp_path: Path
) -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if body["model"] == JUDGE_MODEL:
            text = "I cannot grade this."
        elif len(bodies) == 1:
            return httpx.Response(500, text="upstream is down")
        else:
            text = "Pergunta: qual é o prazo?\nResposta: seis semanas."
        return httpx.Response(
            200, content=sse({"choices": [{"delta": {"content": text}}]}, {"usage": USAGE})
        )

    settings = network_settings(
        seed_bench,
        tmp_path,
        skip_preflight=True,
        judge_price=Price(usd_per_mtok_in=1, usd_per_mtok_out=1),
    )
    env = {"OPENROUTER_API_KEY": "sk-or-123456"}
    pauses: list[float] = []
    store = open_store(tmp_path)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        try:
            prepared = prepare_run(settings, env, client)
            run_id = execute_run(prepared, store, env, client=client, sleep=pauses.append)
            run = store.load_run(run_id)
            scores = store.load_scores(run_id)
            summary = summarize_run(store, run_id, settings.bench.stats)
        finally:
            store.close()
    assert run.status == STATUS_COMPLETED
    assert run.judge_model == f"openrouter:{JUDGE_MODEL}"
    assert run.actual_cost_usd is not None and run.actual_cost_usd > 0
    # The warm-up request failed, so one pause after an error came before the next request.
    assert 2.0 in pauses
    # The expected answer never goes to a candidate, and the judge never sees the mode.
    candidate_bodies = [body for body in bodies if body["model"] != JUDGE_MODEL]
    assert all("gold_question" not in json.dumps(body) for body in candidate_bodies)
    judge_bodies = [body for body in bodies if body["model"] == JUDGE_MODEL]
    assert judge_bodies and all("candidate" not in json.dumps(body) for body in judge_bodies)
    failures = [row for row in scores.values() if "judge.failure" in row]
    assert failures
    assert "no JSON object" in failures[0]["judge.failure"].detail
    mode = summary.modes[0]
    assert mode.judge_failures == len(failures)
    assert mode.judged == 0
    assert mode.quality is None


def test_the_guard_of_each_request_holds_for_a_run_that_was_made_by_hand(
    seed_bench: BenchConfig, tmp_path: Path
) -> None:
    dataset = tmp_path / "sessions.yaml"
    dataset.write_text(PRIVATE_DATASET, encoding="utf-8")
    price = Price(usd_per_mtok_in=1.0, usd_per_mtok_out=1.0)
    unsafe = [openrouter_mode("unsafe").model_copy(update={"price": price})]
    dry = settings_for(seed_bench, tmp_path, unsafe, judge_enabled=False, datasets=[dataset])
    prepared = prepare_run(dry, {})
    real = replace(dry, dry_run=False)
    # A caller hides the private variants from the guard of the run, and keeps them in the plan.
    forged = replace(prepared, settings=real, variants=[])
    assert not forged.has_private_data
    store = open_store(tmp_path)
    try:
        with pytest.raises(PrivacyViolation, match="must not receive the private item"):
            execute_run(forged, store, {}, providers={"unsafe": NoUsage()}, sleep=no_sleep)
        run = store.load_run(store.latest_run_id("analyze"))
    finally:
        store.close()
    assert run.status == STATUS_FAILED


def test_a_prompt_file_that_is_absent_and_a_judge_with_no_provider_are_config_errors(
    seed_bench: BenchConfig, tmp_path: Path
) -> None:
    modes = [fake_mode("only")]
    no_prompt = settings_for(seed_bench, tmp_path, modes, preprompt_paths=[Path("absent.md")])
    with pytest.raises(ConfigError, match="prompt file not found"):
        prepare_run(no_prompt, {})
    settings = settings_for(seed_bench, tmp_path, modes)
    stray = settings.bench.judge.mode.model_copy(update={"provider": "nowhere"})
    bench = settings.bench.model_copy(
        update={"judge": settings.bench.judge.model_copy(update={"mode": stray})}
    )
    with pytest.raises(ConfigError, match="unknown provider nowhere"):
        prepare_run(replace(settings, bench=bench), {})
