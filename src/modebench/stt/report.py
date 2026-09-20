"""The Markdown report of a run of the speech to text suite."""

from collections.abc import Mapping, Sequence

from modebench.config import StatsConfig
from modebench.hashing import stable_seed
from modebench.stats.percentiles import Estimate, bootstrap_percentile, percentile
from modebench.storage.records import RunRecord, SttSessionRecord

STT_REPORT_NAME = "stt_report.md"


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value: float | None, digits: int = 0, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def _interval(estimate: Estimate | None) -> str:
    if estimate is None:
        return "n/a"
    return f"{estimate.value:.0f} ({estimate.ci_low:.0f} to {estimate.ci_high:.0f})"


def _latency(
    values: Sequence[float], q: float, provider: str, stats: StatsConfig
) -> Estimate | None:
    if not values:
        return None
    return bootstrap_percentile(
        values,
        q,
        resamples=stats.bootstrap_resamples,
        confidence=stats.confidence,
        seed=stable_seed(stats.seed, provider, q) % (2**32),
    )


def render_stt_report(
    run: RunRecord,
    sessions: Sequence[SttSessionRecord],
    latencies: Mapping[str, Sequence[float]],
    stats: StatsConfig,
) -> str:
    """Return the report of a speech to text run."""
    lines = [f"# modebench speech to text report: {run.run_id}", ""]
    if run.dry_run:
        lines += ["> **Dry run.** The fake server made these numbers.", ""]
    lines += [
        f"Profile `{run.profile}`, commit `{run.git_sha}`, config `{run.config_hash[:16]}`, "
        f"dataset `{run.dataset_hash[:16]}`.",
        "",
        "The audio goes out at the speed of real time, in chunks of 50 ms. The final latency "
        "is the time from the end of the audio of an utterance to the first message that "
        "settled it. The error rates are computed after the Portuguese normalisation. "
        "Intervals are 95% bootstrap intervals.",
        "",
    ]
    headers = [
        "Provider",
        "Sessions",
        "Failed",
        "First partial after speech p50 ms",
        "Final latency p50 ms",
        "Final latency p95 ms",
        "WER",
        "CER",
        "Speaker accuracy",
        "USD for one hour of audio",
    ]
    rows: list[list[str]] = []
    for provider in sorted({session.provider for session in sessions}):
        own = [session for session in sessions if session.provider == provider]
        firsts: list[float] = []
        wers: list[float] = []
        cers: list[float] = []
        speakers: list[float] = []
        for session in own:
            if session.first_partial_after_speech_ms is not None:
                firsts.append(session.first_partial_after_speech_ms)
            if session.wer is not None:
                wers.append(session.wer)
            if session.cer is not None:
                cers.append(session.cer)
            if session.speaker_accuracy is not None:
                speakers.append(session.speaker_accuracy)
        hours = sum(session.audio_ms for session in own) / 3_600_000.0
        cost = sum(session.cost_usd or 0.0 for session in own)
        values = latencies.get(provider, [])
        accuracy = _mean(speakers)
        rows.append(
            [
                f"`{provider}`",
                str(len(own)),
                str(sum(1 for session in own if not session.ok)),
                _fmt(percentile(firsts, 50) if firsts else None),
                _interval(_latency(values, 50, provider, stats)),
                _interval(_latency(values, 95, provider, stats)),
                _fmt(_mean(wers), 3),
                _fmt(_mean(cers), 3),
                "not supported" if accuracy is None else f"{accuracy:.3f}",
                _fmt(cost / hours if hours > 0 else None, 3),
            ]
        )
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(" --- " for _ in headers) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    failed = [session for session in sessions if not session.ok]
    if failed:
        lines += ["", "## Failed sessions", ""]
        for session in failed:
            lines.append(
                f"- `{session.provider}`, `{session.audio_id}`, repetition {session.repetition}: "
                f"{session.error_message}"
            )
    return "\n".join(lines) + "\n"
