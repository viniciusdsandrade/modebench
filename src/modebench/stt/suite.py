"""Runs the speech to text suite: each audio file through each provider, in turns.

The providers take turns on each audio file, and the provider that goes first
changes from file to file, as in the Analyze suites. One session runs at a
time. A session that fails is a row with its error, and it is never dropped.
"""

import asyncio
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from modebench.config import resolve_path
from modebench.errors import ConfigError, CostCeilingExceeded, PreflightError, PrivacyViolation
from modebench.hashing import git_dirty, git_sha, sha256_files, sha256_text
from modebench.runner.execute import new_run_id, utc_now
from modebench.storage.db import RunStore
from modebench.storage.jsonl import RAW_NAME, RawLog
from modebench.storage.records import RunRecord, SttFinalRecord, SttSessionRecord
from modebench.stt.assemblyai import AssemblyAiAdapter
from modebench.stt.audio import PcmAudio, read_wav, silence
from modebench.stt.base import SttAdapter
from modebench.stt.config import (
    AudioItem,
    ReferenceUtterance,
    SttConfig,
    SttManifest,
    SttProviderConfig,
    load_manifest,
)
from modebench.stt.elevenlabs import ElevenLabsAdapter
from modebench.stt.fake import FakeSttAdapter, fake_connector
from modebench.stt.metrics import SessionMetrics, session_cost, session_metrics
from modebench.stt.replay import (
    Clock,
    Connector,
    SessionTrace,
    Sleep,
    VirtualTime,
    replay_session,
)

SUITE = "stt"
SESSION_OVERHEAD_MS = 2000.0
_DRY_RUN_RATE = 16000


@dataclass(frozen=True, slots=True)
class SttRunSettings:
    """What the operator asked for."""

    root: Path
    config: SttConfig
    profile_name: str
    config_paths: tuple[Path, ...] = ()
    providers: tuple[str, ...] | None = None
    dry_run: bool = False
    max_cost_usd: float | None = None


def synthetic_manifest() -> SttManifest:
    """Return the manifest that a dry run uses when the machine has no audio yet."""
    utterances = [
        ReferenceUtterance(
            start_ms=1000, end_ms=3000, speaker="A", text="bom dia vamos começar a reunião"
        ),
        ReferenceUtterance(
            start_ms=4000, end_ms=6500, speaker="B", text="qual é o prazo para a entrega do módulo"
        ),
        ReferenceUtterance(
            start_ms=8000, end_ms=11000, speaker="A", text="a entrega fica para o dia 23 de outubro"
        ),
    ]
    item = AudioItem(id="synthetic-1", wav=Path("synthetic-1.wav"), utterances=utterances)
    return SttManifest(visibility="public", audios=[item])


def build_adapter(name: str, provider: SttProviderConfig, *, dry_run: bool) -> SttAdapter:
    """Return the adapter of a provider. In a dry run, each provider gets the fake dialect."""
    if dry_run or provider.kind == "fake":
        return FakeSttAdapter(name)
    if provider.kind == "assemblyai":
        return AssemblyAiAdapter(provider.query, provider.url)
    return ElevenLabsAdapter(provider.query, provider.url)


def _load_audio(manifest_dir: Path, item: AudioItem, *, dry_run: bool) -> PcmAudio:
    path = manifest_dir / item.wav
    if dry_run and not path.is_file():
        length_ms = max(utterance.end_ms for utterance in item.utterances) + 1000
        return PcmAudio(sample_rate=_DRY_RUN_RATE, samples=silence(_DRY_RUN_RATE, length_ms))
    return read_wav(path)


def select_providers(settings: SttRunSettings) -> dict[str, SttProviderConfig]:
    """Return the providers of the run: those named, or each enabled provider."""
    config = settings.config
    if settings.providers is None:
        chosen = {name: item for name, item in config.providers.items() if item.enabled}
    else:
        unknown = [name for name in settings.providers if name not in config.providers]
        if unknown:
            raise ConfigError(f"unknown speech to text providers: {', '.join(unknown)}")
        chosen = {name: config.providers[name] for name in settings.providers}
    if not chosen:
        raise ConfigError("no speech to text provider is selected")
    return chosen


def estimate_stt_cost(
    providers: Mapping[str, SttProviderConfig],
    audios: Sequence[PcmAudio],
    repetitions: int,
    tail_silence_ms: int,
) -> float:
    """Return the estimated cost of a run, with the tail silence and a connect overhead."""
    total = 0.0
    for provider in providers.values():
        for audio in audios:
            audio_ms = audio.duration_ms + tail_silence_ms + provider.preroll_silence_ms
            session_ms = audio_ms + SESSION_OVERHEAD_MS
            total += repetitions * session_cost(provider.billing, audio_ms, session_ms)
    return total


def _enforce_guards(
    settings: SttRunSettings,
    manifest: SttManifest,
    providers: Mapping[str, SttProviderConfig],
    env: Mapping[str, str],
    estimate_usd: float,
    ceiling_usd: float,
) -> None:
    if settings.dry_run:
        return
    if manifest.visibility == "private":
        unsafe = [name for name, item in providers.items() if not item.private_data_ok]
        if unsafe:
            raise PrivacyViolation(
                "The manifest is private, and these providers do not set "
                f"private_data_ok = true: {', '.join(unsafe)}."
            )
    missing = [
        item.api_key_env
        for item in providers.values()
        if item.kind != "fake" and not env.get(item.api_key_env, "")
    ]
    if missing:
        names = ", ".join(sorted(set(missing)))
        raise PreflightError(f"missing keys in the environment or in .env: {names}")
    if estimate_usd > ceiling_usd:
        raise CostCeilingExceeded(
            f"The estimate is {estimate_usd:.2f} USD and the ceiling is {ceiling_usd:.2f} USD."
        )


def _session_record(
    run_id: str,
    name: str,
    item: AudioItem,
    repetition: int,
    started_at: str,
    trace: SessionTrace,
    metrics: SessionMetrics,
) -> SttSessionRecord:
    return SttSessionRecord(
        run_id=run_id,
        provider=name,
        audio_id=item.id,
        repetition=repetition,
        started_at=started_at,
        ok=trace.error is None,
        error_message=trace.error,
        audio_ms=trace.audio_ms,
        session_ms=trace.session_ms,
        connect_ms=trace.connect_ms,
        first_partial_ms=metrics.first_partial_ms,
        first_partial_after_speech_ms=metrics.first_partial_after_speech_ms,
        final_latency_p50_ms=metrics.final_latency_p50_ms,
        final_latency_p95_ms=metrics.final_latency_p95_ms,
        finals=len(metrics.finals),
        wer=metrics.rates.wer if metrics.rates is not None else None,
        cer=metrics.rates.cer if metrics.rates is not None else None,
        speaker_accuracy=metrics.speaker_accuracy,
        max_send_lag_ms=trace.max_send_lag_ms,
        cost_usd=metrics.cost_usd,
        hypothesis=metrics.hypothesis,
    )


def run_stt_suite(
    settings: SttRunSettings,
    store: RunStore,
    env: Mapping[str, str],
    *,
    connector: Connector | None = None,
    clock: Clock | None = None,
    sleep: Sleep | None = None,
    now: Callable[[], datetime] = utc_now,
) -> str:
    """Run the suite and return the run identifier.

    A session with the fake server runs on a virtual clock, so a dry run takes
    seconds. A session with a real provider runs on the real clock, at the
    speed of real time.
    """
    config = settings.config
    profile = config.profile(settings.profile_name)
    manifest_path = resolve_path(settings.root, config.manifest)
    if settings.dry_run and not manifest_path.is_file():
        manifest = synthetic_manifest()
    else:
        manifest = load_manifest(manifest_path)
    items = manifest.audios if profile.audios is None else manifest.audios[: profile.audios]
    audios = [_load_audio(manifest_path.parent, item, dry_run=settings.dry_run) for item in items]
    providers = select_providers(settings)
    ceiling = settings.max_cost_usd if settings.max_cost_usd is not None else profile.max_cost_usd
    estimate = estimate_stt_cost(
        providers, audios, profile.repetitions, config.replay.tail_silence_ms
    )
    _enforce_guards(settings, manifest, providers, env, estimate, ceiling)
    wav_paths = [manifest_path.parent / item.wav for item in items]
    hashed = [path for path in [manifest_path, *wav_paths] if path.is_file()]
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
            config_hash=sha256_files(settings.config_paths),
            dataset_hash=sha256_files(hashed),
            prompt_hash=sha256_text(""),
            timeout_s=config.replay.drain_timeout_s,
            judge_model="",
            modes_json=json.dumps(sorted(providers)),
            estimated_cost_usd=estimate,
        )
    )
    names = list(providers)
    spent = 0.0
    runs_dir = resolve_path(settings.root, config.runs_dir)
    with RawLog(runs_dir / run_id / RAW_NAME) as raw:
        for repetition in range(1, profile.repetitions + 1):
            for index, (item, audio) in enumerate(zip(items, audios, strict=True)):
                offset = (index + repetition) % len(names)
                for name in names[offset:] + names[:offset]:
                    provider = providers[name]
                    adapter = build_adapter(name, provider, dry_run=settings.dry_run)
                    use_fake = settings.dry_run or provider.kind == "fake"
                    connect = connector
                    if connect is None:
                        connect = _default_connector(item, audio, provider, use_fake)
                    virtual = VirtualTime()
                    session_clock = clock or (virtual.clock if use_fake else time.perf_counter_ns)
                    session_sleep = sleep or (virtual.sleep if use_fake else asyncio.sleep)
                    started_at = now().isoformat()
                    trace = asyncio.run(
                        replay_session(
                            adapter,
                            audio,
                            connect,
                            env.get(provider.api_key_env, ""),
                            config.replay,
                            preroll_ms=provider.preroll_silence_ms,
                            clock=session_clock,
                            sleep=session_sleep,
                        )
                    )
                    metrics = session_metrics(trace, item, provider.billing, config.normalization)
                    record = _session_record(
                        run_id, name, item, repetition, started_at, trace, metrics
                    )
                    finals = [
                        SttFinalRecord(
                            utterance_index=final.index,
                            text=final.text,
                            speaker=final.speaker,
                            received_ms=final.received_ms,
                            audio_end_ms=final.audio_end_ms,
                            latency_ms=final.latency_ms,
                        )
                        for final in metrics.finals
                    ]
                    store.insert_stt_session(record, finals)
                    raw.write(
                        {
                            "type": "stt_session",
                            "run_id": run_id,
                            "provider": name,
                            "audio_id": item.id,
                            "repetition": repetition,
                            "error": trace.error,
                            "events": [
                                [event.at_ms, event.event.kind, event.event.utterance]
                                for event in trace.events
                            ],
                        }
                    )
                    spent += metrics.cost_usd
    store.finish_run(run_id, "completed", now().isoformat(), spent)
    return run_id


def _default_connector(
    item: AudioItem, audio: PcmAudio, provider: SttProviderConfig, use_fake: bool
) -> Connector:
    if use_fake:
        return fake_connector(item, audio.sample_rate, provider.query)
    from modebench.stt.connector import websockets_connector

    return websockets_connector
