"""The replay of one speech session and the run of the speech suite, with no network."""

import asyncio
import wave
from collections.abc import Mapping
from pathlib import Path

import pytest

from helpers import TickClock
from modebench.config import StatsConfig
from modebench.errors import ConfigError, CostCeilingExceeded, DatasetError, PrivacyViolation
from modebench.runner.execute import STATUS_COMPLETED, STATUS_COST_STOP, STATUS_FAILED
from modebench.storage.db import DB_NAME, RunStore
from modebench.storage.records import RunRecord, SttSessionRecord
from modebench.stt.assemblyai import AssemblyAiAdapter
from modebench.stt.audio import PcmAudio, chunk_bytes, read_wav, silence
from modebench.stt.base import ERROR, FINAL, SttEvent, SttParser
from modebench.stt.config import (
    AudioItem,
    ReferenceUtterance,
    SttBilling,
    SttConfig,
    SttManifest,
    SttNormalization,
    SttProfile,
    SttProviderConfig,
    SttReplayConfig,
    load_manifest,
    load_stt_config,
)
from modebench.stt.fake import FakeSttAdapter, fake_connector
from modebench.stt.metrics import session_metrics
from modebench.stt.replay import (
    EARLY_END,
    ConnectionEnded,
    SessionTrace,
    VirtualTime,
    WsConnection,
    replay_session,
)
from modebench.stt.report import render_stt_report
from modebench.stt.suite import (
    SttRunSettings,
    estimate_stt_cost,
    manifest_is_private,
    run_stt_suite,
    select_providers,
    synthetic_manifest,
)

RATE = 16000
ITEM = synthetic_manifest().audios[0]
AUDIO = PcmAudio(sample_rate=RATE, samples=silence(RATE, 12000))
SESSION_BILLING = SttBilling(basis="session_seconds", usd_per_hour=3600.0)
AUDIO_BILLING = SttBilling(basis="audio_seconds", usd_per_hour=3600.0)


def replay(
    connection_or_error: WsConnection | BaseException,
    *,
    adapter: AssemblyAiAdapter | FakeSttAdapter | None = None,
    preroll_ms: int = 0,
) -> SessionTrace:
    async def connector(url: str, headers: Mapping[str, str], timeout_s: float) -> WsConnection:
        if isinstance(connection_or_error, BaseException):
            raise connection_or_error
        return connection_or_error

    virtual = VirtualTime()
    return asyncio.run(
        replay_session(
            adapter or FakeSttAdapter("fake"),
            AUDIO,
            connector,
            "key-123456",
            SttReplayConfig(drain_timeout_s=0.2),
            preroll_ms=preroll_ms,
            clock=virtual.clock,
            sleep=virtual.sleep,
        )
    )


class DropsEarly:
    """A server that closes the socket after some chunks, as a policy violation does."""

    def __init__(self, after: int = 10) -> None:
        self.sent = 0
        self._after = after
        self._closed = asyncio.Event()

    async def send(self, message: str | bytes) -> None:
        if self._closed.is_set():
            raise ConnectionEnded("received 1008 (policy violation) too many sessions key-123456")
        self.sent += 1
        if self.sent == self._after:
            self._closed.set()
        await asyncio.sleep(0)

    async def recv(self) -> str | bytes:
        await self._closed.wait()
        raise ConnectionEnded("received 1008 (policy violation) too many sessions key-123456")

    async def close(self) -> None:
        self._closed.set()


def test_a_complete_session_has_no_error_and_counts_the_audio_that_went_out() -> None:
    async def run() -> SessionTrace:
        connector = fake_connector(ITEM, RATE, {}, preroll_ms=2000)
        virtual = VirtualTime()
        return await replay_session(
            FakeSttAdapter("fake"),
            AUDIO,
            connector,
            "",
            SttReplayConfig(),
            preroll_ms=2000,
            clock=virtual.clock,
            sleep=virtual.sleep,
        )

    trace = asyncio.run(run())
    assert trace.error is None
    # 2 s of silence before, 12 s of audio and 3 s of silence after.
    assert trace.sent_ms == pytest.approx(17000.0)
    finals = [item for item in trace.events if item.event.kind == FINAL]
    assert len(finals) == len(ITEM.utterances)
    metrics = session_metrics(trace, ITEM, AUDIO_BILLING, SttNormalization())
    assert metrics.rates is not None and metrics.rates.wer == 0.0
    assert metrics.speaker_accuracy == 1.0
    # The final of an utterance comes 700 ms after its end, with or without silence before.
    assert metrics.final_latency_p50_ms == pytest.approx(700.0, abs=50.0)
    assert metrics.cost_usd == pytest.approx(17.0)


def test_a_session_that_the_provider_ends_early_is_a_failure() -> None:
    server = DropsEarly(after=10)
    trace = replay(server)
    assert server.sent == 10
    assert trace.error is not None
    assert trace.error.startswith(EARLY_END)
    # The reason of the close stays in the message, and the key does not.
    assert "1008" in trace.error
    assert "key-123456" not in trace.error
    assert trace.sent_ms == pytest.approx(500.0)


class ErrorsThenCloses:
    """A server that sends one message and then closes."""

    def __init__(self, message: str) -> None:
        self._messages: asyncio.Queue[str | None] = asyncio.Queue()
        self._messages.put_nowait(message)
        self._messages.put_nowait(None)

    async def send(self, message: str | bytes) -> None:
        await asyncio.sleep(0)

    async def recv(self) -> str | bytes:
        item = await self._messages.get()
        if item is None:
            raise ConnectionEnded("")
        return item

    async def close(self) -> None:
        self._messages.put_nowait(None)


def test_an_error_message_of_the_provider_is_the_error_of_the_session() -> None:
    server = ErrorsThenCloses('{"type":"Error","error_code":"3005","error":"session expired"}')
    trace = replay(server, adapter=AssemblyAiAdapter({}))
    assert trace.error == "3005: session expired"


class BrokenParser:
    def parse(self, raw: str | bytes) -> list[SttEvent]:
        raise KeyError("words")


class BrokenDialect(FakeSttAdapter):
    def new_parser(self) -> SttParser:
        return BrokenParser()


def test_a_parser_defect_is_the_error_of_one_session() -> None:
    trace = replay(ErrorsThenCloses('{"type":"final"}'), adapter=BrokenDialect("broken"))
    assert trace.error is not None
    assert "the parser failed: KeyError" in trace.error
    assert [item.event.kind for item in trace.events] == [ERROR]


class RecvFails:
    async def send(self, message: str | bytes) -> None:
        await asyncio.sleep(0)

    async def recv(self) -> str | bytes:
        raise OSError("socket is gone")

    async def close(self) -> None:
        return None


def test_a_receiver_defect_is_the_error_of_one_session() -> None:
    trace = replay(RecvFails())
    assert trace.error is not None
    assert "the receiver failed: OSError: socket is gone" in trace.error


def test_a_session_that_cannot_connect_costs_nothing() -> None:
    trace = replay(OSError("no route to host"))
    assert trace.error == "connect failed: OSError: no route to host"
    assert trace.sent_ms == 0.0
    metrics = session_metrics(trace, ITEM, AUDIO_BILLING, SttNormalization())
    assert metrics.cost_usd == 0.0
    assert metrics.rates is not None and metrics.rates.wer == 1.0
    assert metrics.first_partial_ms is None


class NeverConfirms:
    """A server that takes each message and never says that the session ended."""

    def __init__(self) -> None:
        self._closed = asyncio.Event()

    async def send(self, message: str | bytes) -> None:
        await asyncio.sleep(0)

    async def recv(self) -> str | bytes:
        await self._closed.wait()
        raise ConnectionEnded("")

    async def close(self) -> None:
        self._closed.set()


def test_a_provider_that_does_not_confirm_the_end_is_a_failure() -> None:
    trace = replay(NeverConfirms(), adapter=AssemblyAiAdapter({}))
    assert trace.error == "the provider did not confirm the end of the session"
    assert trace.sent_ms == pytest.approx(15000.0)


def write_wav(path: Path, *, channels: int = 1, width: int = 2, seconds: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(RATE)
        handle.writeframes(bytes(int(RATE * seconds) * channels * width))
    return path


def test_read_wav_mixes_channels_and_refuses_other_sample_widths(tmp_path: Path) -> None:
    mono = read_wav(write_wav(tmp_path / "mono.wav"))
    assert mono.sample_rate == RATE
    assert mono.duration_ms == pytest.approx(1000.0)
    stereo = read_wav(write_wav(tmp_path / "stereo.wav", channels=2))
    assert stereo.duration_ms == pytest.approx(1000.0)
    with pytest.raises(DatasetError, match="16 bit"):
        read_wav(write_wav(tmp_path / "eight.wav", width=1))
    with pytest.raises(DatasetError, match="not found"):
        read_wav(tmp_path / "absent.wav")
    (tmp_path / "text.wav").write_text("not audio", encoding="utf-8")
    with pytest.raises(DatasetError, match="not a WAV file"):
        read_wav(tmp_path / "text.wav")
    assert len(chunk_bytes(silence(RATE, 120), RATE, 50)) == 3
    with pytest.raises(ValueError, match="shorter than one sample"):
        chunk_bytes(b"", 10, 10)


MANIFEST = """\
version: 1
visibility: public
audios:
  - id: call-1
    wav: call-1.wav
    utterances:
      - start_ms: 100
        end_ms: 900
        speaker: A
        text: "bom dia"
"""


def fake_provider(billing: SttBilling = SESSION_BILLING) -> SttProviderConfig:
    return SttProviderConfig(kind="fake", billing=billing)


def suite_config(tmp_path: Path, manifest: Path, **providers: SttProviderConfig) -> SttConfig:
    return SttConfig(
        manifest=manifest,
        runs_dir=tmp_path / "runs",
        providers=providers,
        profiles={"smoke": SttProfile(repetitions=2, max_cost_usd=1000.0)},
    )


def public_manifest(directory: Path) -> Path:
    write_wav(directory / "call-1.wav")
    manifest = directory / "manifest.yaml"
    manifest.write_text(MANIFEST, encoding="utf-8")
    return manifest


def test_a_dry_run_of_the_suite_needs_no_audio_and_completes(tmp_path: Path) -> None:
    config = suite_config(
        tmp_path, tmp_path / "absent.yaml", one=fake_provider(), two=fake_provider()
    )
    settings = SttRunSettings(root=tmp_path, config=config, profile_name="smoke", dry_run=True)
    store = RunStore(tmp_path / "runs" / DB_NAME)
    try:
        run_id = run_stt_suite(settings, store, {})
        run = store.load_run(run_id)
        sessions = store.load_stt_sessions(run_id)
        latencies = store.load_stt_final_latencies(run_id)
    finally:
        store.close()
    assert run.status == STATUS_COMPLETED
    assert len(sessions) == 4
    assert all(session.ok and session.wer == 0.0 for session in sessions)
    assert set(latencies) == {"one", "two"}
    report = render_stt_report(run, sessions, latencies, StatsConfig(bootstrap_resamples=50))
    assert "Dry run" in report
    assert "`one`" in report and "`two`" in report
    assert "Failed sessions" not in report


def test_a_real_run_of_the_suite_stops_at_the_cost_ceiling(tmp_path: Path) -> None:
    manifest = public_manifest(tmp_path / "audio")
    config = suite_config(tmp_path, manifest, one=fake_provider())
    settings = SttRunSettings(
        root=tmp_path, config=config, profile_name="smoke", max_cost_usd=100.0
    )
    store = RunStore(tmp_path / "runs" / DB_NAME)
    try:
        # A clock that moves one second for each reading makes a session much longer, and thus
        # much more expensive, than the estimate of 12 USD.
        run_id = run_stt_suite(settings, store, {}, clock=TickClock(step_ms=1000))
        run = store.load_run(run_id)
        sessions = store.load_stt_sessions(run_id)
    finally:
        store.close()
    assert run.status == STATUS_COST_STOP
    assert len(sessions) == 1
    assert run.actual_cost_usd is not None and run.actual_cost_usd > 100.0


def test_a_defect_in_the_suite_gives_the_status_failed(tmp_path: Path) -> None:
    manifest = public_manifest(tmp_path / "audio")
    config = suite_config(tmp_path, manifest, one=fake_provider())
    settings = SttRunSettings(root=tmp_path, config=config, profile_name="smoke")

    async def broken(url: str, headers: Mapping[str, str], timeout_s: float) -> WsConnection:
        raise RuntimeError("a defect")

    store = RunStore(tmp_path / "runs" / DB_NAME)
    try:
        with pytest.raises(RuntimeError, match="a defect"):
            run_stt_suite(settings, store, {}, connector=broken)
        run = store.load_run(store.latest_run_id("stt"))
    finally:
        store.close()
    assert run.status == STATUS_FAILED
    assert run.finished_at is not None


def test_a_manifest_in_the_private_directory_is_private(tmp_path: Path) -> None:
    manifest_path = public_manifest(tmp_path / "data" / "private" / "stt")
    manifest = load_manifest(manifest_path)
    assert manifest.visibility == "public"
    assert manifest_is_private(tmp_path, manifest_path, manifest)
    assert not manifest_is_private(tmp_path / "other", manifest_path, manifest)
    hidden = SttManifest(visibility="private", audios=manifest.audios)
    assert manifest_is_private(tmp_path / "other", manifest_path, hidden)

    remote = SttProviderConfig(
        kind="assemblyai", api_key_env="ASSEMBLYAI_API_KEY", billing=AUDIO_BILLING
    )
    config = suite_config(tmp_path, manifest_path, remote=remote, local=fake_provider())
    settings = SttRunSettings(root=tmp_path, config=config, profile_name="smoke")
    store = RunStore(tmp_path / "runs" / DB_NAME)
    try:
        with pytest.raises(PrivacyViolation, match="remote"):
            run_stt_suite(settings, store, {"ASSEMBLYAI_API_KEY": "key-123456"})
    finally:
        store.close()


def test_the_guards_of_the_suite_stop_a_run_with_no_key_or_a_high_estimate(tmp_path: Path) -> None:
    manifest = public_manifest(tmp_path / "audio")
    remote = SttProviderConfig(
        kind="assemblyai", api_key_env="ASSEMBLYAI_API_KEY", billing=AUDIO_BILLING
    )
    store = RunStore(tmp_path / "runs" / DB_NAME)
    try:
        config = suite_config(tmp_path, manifest, remote=remote)
        settings = SttRunSettings(root=tmp_path, config=config, profile_name="smoke")
        with pytest.raises(Exception, match="ASSEMBLYAI_API_KEY"):
            run_stt_suite(settings, store, {})
        cheap = SttRunSettings(
            root=tmp_path, config=config, profile_name="smoke", max_cost_usd=0.0001
        )
        with pytest.raises(CostCeilingExceeded, match="ceiling"):
            run_stt_suite(cheap, store, {"ASSEMBLYAI_API_KEY": "key-123456"})
    finally:
        store.close()


def test_select_providers_and_the_estimate() -> None:
    providers = {
        "on": fake_provider(AUDIO_BILLING),
        "off": SttProviderConfig(kind="fake", billing=AUDIO_BILLING, enabled=False),
    }
    config = SttConfig(providers=providers, profiles={"smoke": SttProfile(max_cost_usd=1.0)})
    root = Path(".")
    enabled = select_providers(SttRunSettings(root=root, config=config, profile_name="smoke"))
    assert list(enabled) == ["on"]
    named = SttRunSettings(root=root, config=config, profile_name="smoke", providers=("off",))
    assert list(select_providers(named)) == ["off"]
    unknown = SttRunSettings(root=root, config=config, profile_name="smoke", providers=("x",))
    with pytest.raises(ConfigError, match="unknown speech to text providers: x"):
        select_providers(unknown)
    none = SttConfig(providers={"off": providers["off"]}, profiles=config.profiles)
    with pytest.raises(ConfigError, match="no speech to text provider"):
        select_providers(SttRunSettings(root=root, config=none, profile_name="smoke"))
    with pytest.raises(ConfigError, match="unknown profile"):
        config.profile("absent")
    # 12 s of audio and 3 s of silence, two repetitions, 1 USD for each second.
    assert estimate_stt_cost({"on": providers["on"]}, [AUDIO], 2, 3000) == pytest.approx(30.0)


def test_the_seed_config_of_the_suite_is_valid(repo_root: Path, tmp_path: Path) -> None:
    config = load_stt_config(repo_root / "configs" / "stt.toml")
    assert set(config.providers) == {"assemblyai", "elevenlabs"}
    assert config.providers["assemblyai"].billing.total_usd_per_hour == pytest.approx(0.57)
    with pytest.raises(ConfigError, match="not found"):
        load_stt_config(tmp_path / "absent.toml")
    with pytest.raises(DatasetError, match="manifest not found"):
        load_manifest(tmp_path / "absent.yaml")
    (tmp_path / "bad.yaml").write_text("audios: []\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="not a valid manifest"):
        load_manifest(tmp_path / "bad.yaml")
    with pytest.raises(ValueError, match="end after it starts"):
        ReferenceUtterance(start_ms=500, end_ms=400, text="x")
    item = AudioItem(
        id="a",
        wav=Path("a.wav"),
        utterances=[
            ReferenceUtterance(start_ms=900, end_ms=1000, text="depois"),
            ReferenceUtterance(start_ms=0, end_ms=500, text="antes"),
        ],
    )
    assert item.reference_text == "antes depois"


class ClosesAtTheEnd:
    """A server that takes the audio and drops the message that ends the session."""

    def __init__(self) -> None:
        self._closed = asyncio.Event()

    async def send(self, message: str | bytes) -> None:
        if isinstance(message, str):
            raise ConnectionEnded("sent 1000 (OK)")
        await asyncio.sleep(0)

    async def recv(self) -> str | bytes:
        await self._closed.wait()
        raise ConnectionEnded("")

    async def close(self) -> None:
        self._closed.set()


def test_a_connection_that_closes_after_the_audio_is_a_failure_of_the_end_only() -> None:
    trace = replay(ClosesAtTheEnd())
    assert trace.error == "the provider closed the connection before the session ended"
    assert trace.sent_ms == pytest.approx(15000.0)


class NeverReturns:
    """A server whose receive call does not end, also after the close."""

    async def send(self, message: str | bytes) -> None:
        await asyncio.sleep(0)

    async def recv(self) -> str | bytes:
        await asyncio.Event().wait()
        return ""

    async def close(self) -> None:
        return None


def test_a_receiver_that_does_not_end_is_cancelled() -> None:
    trace = replay(NeverReturns())
    assert trace.error == "the provider did not confirm the end of the session"


def test_the_speech_report_lists_the_failed_sessions_and_flags_a_run_that_is_not_complete() -> None:
    run = RunRecord(
        run_id="20260901T000000Z-abc123",
        suite="stt",
        profile="smoke",
        started_at="2026-09-01T00:00:00+00:00",
        status="stopped_at_cost_ceiling",
        dry_run=False,
        git_sha="abcdef1",
        git_dirty=False,
        config_hash="c" * 64,
        dataset_hash="d" * 64,
        prompt_hash="p" * 64,
        timeout_s=8.0,
        judge_model="",
        modes_json='["one"]',
    )
    failed = SttSessionRecord(
        run_id=run.run_id,
        provider="one",
        audio_id="call-1",
        repetition=1,
        started_at=run.started_at,
        ok=False,
        error_message="the provider ended the session | [link](http://x)",
        audio_ms=0.0,
        session_ms=10.0,
        connect_ms=None,
        first_partial_ms=None,
        first_partial_after_speech_ms=None,
        final_latency_p50_ms=None,
        final_latency_p95_ms=None,
        finals=0,
        wer=None,
        cer=None,
        speaker_accuracy=None,
        max_send_lag_ms=0.0,
        cost_usd=None,
        hypothesis="",
    )
    text = render_stt_report(run, [failed], {}, StatsConfig(bootstrap_resamples=50))
    assert "**Run not complete** (status `stopped_at_cost_ceiling`)" in text
    assert "## Failed sessions" in text
    assert r"the provider ended the session \| \[link\](http://x)" in text
    assert "| `one` | 1 | 1 | n/a | n/a | n/a | n/a | n/a | not supported | n/a |" in text
