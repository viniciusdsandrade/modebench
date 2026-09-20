"""The command line: `modebench run`, `decide`, `report` and `compare`.

Exit codes: 0 is success, 1 is a regression that `compare` found, 2 is an
error that stopped the command, and 3 is a run that stopped at the cost
ceiling.
"""

import logging
import shutil
from pathlib import Path
from typing import Annotated

import httpx
import typer

from modebench.config import (
    BenchConfig,
    ModesFile,
    load_bench_config,
    load_modes_file,
    resolve_path,
)
from modebench.dataset.importer import (
    ANALYZE_TASK,
    ImportOptions,
    import_cases,
    write_private_dataset,
)
from modebench.decision.decide import (
    RECOMMENDED_NAME,
    STATUS_OK,
    decide,
    modes_of_run,
    recommended_document,
    write_recommended,
)
from modebench.decision.regression import compare_summaries
from modebench.env import load_env
from modebench.errors import ConfigError, ModebenchError
from modebench.redact import install_redaction
from modebench.report.markdown import REPORT_NAME, mode_rows, render_report
from modebench.report.plots import plot_latency, plot_pareto
from modebench.runner.execute import (
    STATUS_COMPLETED,
    STATUS_COST_STOP,
    RunSettings,
    execute_run,
    prepare_run,
    utc_now,
)
from modebench.runner.preflight import run_preflight
from modebench.stats.aggregate import (
    SUMMARY_NAME,
    RunSummary,
    load_summary,
    summarize_run,
    write_summary,
)
from modebench.storage.db import DB_NAME, RunStore
from modebench.storage.records import RequestRecord
from modebench.stt.config import load_stt_config
from modebench.stt.report import STT_REPORT_NAME, render_stt_report
from modebench.stt.suite import SttRunSettings, run_stt_suite

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Select the provider, model and reasoning level of each role, with your own data.",
)

RootOption = Annotated[Path, typer.Option(help="The directory from which relative paths start.")]
ConfigOption = Annotated[Path, typer.Option(help="The bench file.")]
RunOption = Annotated[str, typer.Option("--run", help="A run identifier, or `latest`.")]

ANALYZE_SUITE = "analyze"
STT_SUITE = "stt"
EXIT_REGRESSION = 1
EXIT_ERROR = 2
EXIT_COST_STOP = 3
PROGRESS_EVERY = 10


def _fail(error: ModebenchError) -> typer.Exit:
    typer.secho(f"error: {error}", fg=typer.colors.RED, err=True)
    return typer.Exit(EXIT_ERROR)


def _setup_logging(secrets: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    install_redaction(secrets)


def _load(root: Path, config: Path) -> tuple[BenchConfig, ModesFile, Path, Path]:
    bench_path = resolve_path(root, config)
    bench = load_bench_config(bench_path)
    modes_path = resolve_path(root, bench.paths.modes_file)
    return bench, load_modes_file(modes_path), bench_path, modes_path


def _store(root: Path, runs_dir: Path) -> RunStore:
    return RunStore(resolve_path(root, runs_dir) / DB_NAME)


def _require_usable(summary: RunSummary, *, allow_dry_run: bool = False) -> None:
    """Stop a command that must read a real and complete run of the Analyze suites.

    A decision, a baseline and a regression gate are facts that other people
    and other programs use. Fake numbers, or the numbers of a run that stopped
    early, must not become such a fact without a clear statement.
    """
    if summary.suite != ANALYZE_SUITE:
        raise ConfigError(
            f"run {summary.run_id} is of the suite {summary.suite}, "
            "and this command reads a run of the Analyze suites"
        )
    if summary.status != STATUS_COMPLETED:
        raise ConfigError(
            f"run {summary.run_id} is not complete (status {summary.status}). "
            "Its numbers come from a part of the plan, so they are not used"
        )
    if summary.dry_run and not allow_dry_run:
        raise ConfigError(
            f"run {summary.run_id} is a dry run. Its numbers are fake, so they are not used. "
            "Use --allow-dry-run only to test the pipeline"
        )


def _progress(done: int, total: int, record: RequestRecord) -> None:
    if done % PROGRESS_EVERY == 0 or done == total or not record.ok:
        state = "ok" if record.ok else f"FAILED ({record.error_kind})"
        typer.echo(f"[{done}/{total}] {record.mode_id} {record.item_id}: {state}")


def _run_stt(
    root: Path, stt_config: Path, profile: str, dry_run: bool, max_cost: float | None
) -> None:
    config_path = resolve_path(root, stt_config)
    config = load_stt_config(config_path)
    env = load_env(root)
    keys = [env.get(item.api_key_env, "") for item in config.providers.values()]
    _setup_logging([key for key in keys if key])
    settings = SttRunSettings(
        root=root,
        config=config,
        profile_name=profile,
        config_paths=(config_path,),
        dry_run=dry_run,
        max_cost_usd=max_cost,
    )
    store = _store(root, config.runs_dir)
    try:
        run_id = run_stt_suite(settings, store, env)
        status = store.load_run(run_id).status
    finally:
        store.close()
    typer.echo(f"run {run_id}: {status}. Make the report with: modebench report --run {run_id}")
    if status == STATUS_COST_STOP:
        raise typer.Exit(EXIT_COST_STOP)


@app.command()
def run(
    suite: Annotated[str, typer.Option(help="`analyze` or `stt`.")] = "analyze",
    profile: Annotated[str, typer.Option(help="`smoke`, `quick` or `full`.")] = "smoke",
    config: ConfigOption = Path("configs/bench.toml"),
    stt_config: Annotated[Path, typer.Option(help="The speech to text file.")] = Path(
        "configs/stt.toml"
    ),
    modes: Annotated[str | None, typer.Option(help="Mode identifiers with commas.")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Use the fake provider. No request leaves.")
    ] = False,
    estimate_only: Annotated[
        bool, typer.Option("--estimate-only", help="Print the plan and the estimate, and stop.")
    ] = False,
    no_judge: Annotated[bool, typer.Option("--no-judge", help="Do not grade the answers.")] = False,
    max_cost: Annotated[
        float | None, typer.Option("--max-cost", help="The cost ceiling of this run, in USD.")
    ] = None,
    skip_preflight: Annotated[
        bool, typer.Option("--skip-preflight", help="Do not check the model identifiers.")
    ] = False,
    root: RootOption = Path("."),
) -> None:
    """Run a suite and write its results."""
    try:
        if suite == STT_SUITE:
            _run_stt(root, stt_config, profile, dry_run, max_cost)
            return
        if suite != ANALYZE_SUITE:
            raise ConfigError(f"unknown suite {suite}. Use analyze or stt")
        bench, modes_file, bench_path, modes_path = _load(root, config)
        env = load_env(root)
        named = [name.strip() for name in modes.split(",") if name.strip()] if modes else None
        # A name given two times must not send each of its requests two times.
        chosen = modes_file.select(list(dict.fromkeys(named)) if named else None)
        keys = [env.get(item.api_key_env, "") for item in modes_file.providers.values()]
        _setup_logging([key for key in keys if key])
        settings = RunSettings(
            root=root,
            bench=bench,
            modes_file=modes_file,
            modes=tuple(chosen),
            profile_name=profile,
            config_paths=(bench_path, modes_path),
            dry_run=dry_run,
            judge_enabled=not no_judge,
            max_cost_usd=max_cost,
            skip_preflight=skip_preflight,
        )
        prepared = prepare_run(settings, env)
        typer.echo(
            f"plan: {len(prepared.items)} items, {len(chosen)} modes, "
            f"{prepared.estimate.requests} requests with the judge"
        )
        for item in prepared.estimate.modes:
            cost = "no price" if item.cost_usd is None else f"{item.cost_usd:.4f} USD"
            typer.echo(
                f"  {item.mode_id}: {item.requests} requests, {item.input_tokens} tokens in, "
                f"{item.output_tokens} tokens out, {cost}"
            )
        typer.echo(
            f"estimate: {prepared.estimate.total_usd:.2f} USD, "
            f"ceiling: {prepared.ceiling_usd:.2f} USD"
        )
        if estimate_only:
            return
        store = _store(root, bench.paths.runs_dir)
        try:
            run_id = execute_run(prepared, store, env, progress=_progress)
            summary = summarize_run(store, run_id, bench.stats)
        finally:
            store.close()
        run_dir = resolve_path(root, bench.paths.runs_dir) / run_id
        write_summary(summary, run_dir / SUMMARY_NAME)
        for line in mode_rows(summary.modes):
            typer.echo(line)
        typer.echo(f"run {run_id}: {summary.status}")
        if summary.status != STATUS_COMPLETED:
            raise typer.Exit(EXIT_COST_STOP)
    except ModebenchError as error:
        raise _fail(error) from error


@app.command("decide")
def decide_command(
    run_id: RunOption = "latest",
    config: ConfigOption = Path("configs/bench.toml"),
    out: Annotated[
        Path | None,
        typer.Option(
            help="Where the recommendations go. The default is recommended_modes.toml at the "
            "root. A dry run with no --out writes only in its run directory."
        ),
    ] = None,
    allow_dry_run: Annotated[
        bool, typer.Option("--allow-dry-run", help="Decide from fake numbers, for a test.")
    ] = False,
    root: RootOption = Path("."),
) -> None:
    """Select the mode of each role and write `recommended_modes.toml`."""
    try:
        bench, modes_file, _, _ = _load(root, config)
        store = _store(root, bench.paths.runs_dir)
        try:
            resolved = store.resolve_run_id(run_id, ANALYZE_SUITE)
            record = store.load_run(resolved)
            summary = summarize_run(store, resolved, bench.stats)
        finally:
            store.close()
        _require_usable(summary, allow_dry_run=allow_dry_run)
        run_modes = modes_of_run(record.modes_json)
        decisions = decide(bench.roles, summary, run_modes)
        document = recommended_document(
            decisions, summary, run_modes, modes_file, utc_now().isoformat()
        )
        run_dir = resolve_path(root, bench.paths.runs_dir) / resolved
        target = write_recommended(document, run_dir / RECOMMENDED_NAME)
        if out is not None:
            target = write_recommended(document, resolve_path(root, out))
        elif not summary.dry_run:
            # The application reads this file, so fake numbers never go there by default.
            target = write_recommended(document, resolve_path(root, Path(RECOMMENDED_NAME)))
        for decision in decisions:
            winner = decision.winner if decision.status == STATUS_OK else "no mode"
            typer.echo(f"{decision.role}: {winner} ({decision.reason})")
        typer.echo(f"wrote {target}")
    except ModebenchError as error:
        raise _fail(error) from error


@app.command()
def report(
    run_id: RunOption = "latest",
    config: ConfigOption = Path("configs/bench.toml"),
    baseline: Annotated[
        Path | None, typer.Option(help="A summary file to compare with, if there is one.")
    ] = None,
    root: RootOption = Path("."),
) -> None:
    """Write the Markdown report and the charts of a run."""
    try:
        bench, _, _, _ = _load(root, config)
        store = _store(root, bench.paths.runs_dir)
        try:
            resolved = store.resolve_run_id(run_id)
            record = store.load_run(resolved)
            run_dir = resolve_path(root, bench.paths.runs_dir) / resolved
            if record.suite == STT_SUITE:
                text = render_stt_report(
                    record,
                    store.load_stt_sessions(resolved),
                    store.load_stt_final_latencies(resolved),
                    bench.stats,
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                target = run_dir / STT_REPORT_NAME
                target.write_text(text, encoding="utf-8")
                typer.echo(f"wrote {target}")
                return
            summary = summarize_run(store, resolved, bench.stats)
            requests = store.load_requests(resolved)
            scores = store.load_scores(resolved)
        finally:
            store.close()
        run_modes = modes_of_run(record.modes_json)
        decisions = decide(bench.roles, summary, run_modes)
        images: dict[str, str] = {}
        if plot_pareto(summary.modes, bench.roles, run_dir / "pareto.png") is not None:
            images["pareto"] = "pareto.png"
        if plot_latency(summary.modes, bench.roles, run_dir / "latency.png") is not None:
            images["latency"] = "latency.png"
        baseline_summary = None
        comparison = None
        if baseline is not None:
            if not resolve_path(root, baseline).is_file():
                raise ConfigError(f"baseline not found: {resolve_path(root, baseline)}")
            baseline_summary = load_summary(resolve_path(root, baseline))
            comparison = compare_summaries(
                summary, baseline_summary, bench.regression.p95_increase_ratio
            )
        text = render_report(
            summary,
            decisions,
            bench.roles,
            run_modes,
            requests,
            scores,
            images=images,
            baseline=baseline_summary,
            comparison=comparison,
        )
        write_summary(summary, run_dir / SUMMARY_NAME)
        target = run_dir / REPORT_NAME
        target.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {target}")
    except ModebenchError as error:
        raise _fail(error) from error


@app.command()
def compare(
    run_id: RunOption = "latest",
    baseline: Annotated[Path, typer.Option(help="The summary file of the baseline.")] = Path(
        "baselines/baseline.json"
    ),
    config: ConfigOption = Path("configs/bench.toml"),
    allow_dry_run: Annotated[
        bool, typer.Option("--allow-dry-run", help="Compare fake numbers, for a test.")
    ] = False,
    root: RootOption = Path("."),
) -> None:
    """Compare a run with the baseline. The exit code is 1 if there is a regression."""
    try:
        bench, _, _, _ = _load(root, config)
        baseline_path = resolve_path(root, baseline)
        if not baseline_path.is_file():
            raise ConfigError(
                f"baseline not found: {baseline_path}. Make one with: modebench baseline"
            )
        store = _store(root, bench.paths.runs_dir)
        try:
            resolved = store.resolve_run_id(run_id, ANALYZE_SUITE)
            summary = summarize_run(store, resolved, bench.stats)
        finally:
            store.close()
        _require_usable(summary, allow_dry_run=allow_dry_run)
        result = compare_summaries(
            summary, load_summary(baseline_path), bench.regression.p95_increase_ratio
        )
        for warning in result.warnings:
            typer.secho(f"warning: {warning}", fg=typer.colors.YELLOW, err=True)
        for item in result.regressions:
            typer.echo(
                f"REGRESSION {item.mode_id} {item.metric}: {item.baseline:.3f} before, "
                f"{item.current:.3f} now, {item.detail}"
            )
        if result.failed:
            raise typer.Exit(EXIT_REGRESSION)
        typer.echo(f"run {resolved}: no regression against {baseline_path}")
    except ModebenchError as error:
        raise _fail(error) from error


@app.command("baseline")
def baseline_command(
    run_id: RunOption = "latest",
    out: Annotated[Path, typer.Option(help="Where the baseline goes.")] = Path(
        "baselines/baseline.json"
    ),
    config: ConfigOption = Path("configs/bench.toml"),
    root: RootOption = Path("."),
) -> None:
    """Make a run the baseline of later comparisons."""
    try:
        bench, _, _, _ = _load(root, config)
        store = _store(root, bench.paths.runs_dir)
        try:
            resolved = store.resolve_run_id(run_id, ANALYZE_SUITE)
            summary = summarize_run(store, resolved, bench.stats)
        finally:
            store.close()
        _require_usable(summary)
        run_summary = resolve_path(root, bench.paths.runs_dir) / resolved / SUMMARY_NAME
        write_summary(summary, run_summary)
        target = resolve_path(root, out)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_summary, target)
        typer.echo(f"wrote {target}")
    except ModebenchError as error:
        raise _fail(error) from error


@app.command("preflight")
def preflight_command(
    config: ConfigOption = Path("configs/bench.toml"),
    root: RootOption = Path("."),
) -> None:
    """Check each model identifier against the catalogue of its provider."""
    try:
        bench, modes_file, _, _ = _load(root, config)
        env = load_env(root)
        checked = [*modes_file.select(), bench.judge.mode]
        with httpx.Client() as client:
            result = run_preflight(modes_file, checked, env, client)
        for mode_id, key in sorted(result.matched.items()):
            price = result.prices.get(mode_id)
            text = "no price"
            if price is not None:
                text = f"{price.usd_per_mtok_in:g} in, {price.usd_per_mtok_out:g} out"
            typer.echo(f"{mode_id}: {key} ({text} USD for 1M tokens)")
    except ModebenchError as error:
        raise _fail(error) from error


@app.command("import-sessions")
def import_sessions(
    db: Annotated[Path, typer.Option(help="The SQLite file of the stt-bridge application.")],
    out: Annotated[Path, typer.Option(help="The dataset to write.")] = Path(
        "data/private/sttbridge_sessions.yaml"
    ),
    all_tasks: Annotated[
        bool, typer.Option("--all-tasks", help="Import each task, not only the Analyze button.")
    ] = False,
    include_unwindowed: Annotated[
        bool, typer.Option("--include-unwindowed", help="Import the answers of the terminal too.")
    ] = False,
    meeting_id: Annotated[int | None, typer.Option(help="Import one meeting only.")] = None,
    limit: Annotated[int | None, typer.Option(help="The largest number of cases.")] = None,
    force: Annotated[bool, typer.Option("--force", help="Replace the output file.")] = False,
    root: RootOption = Path("."),
) -> None:
    """Import sessions of the stt-bridge application into a private dataset."""
    try:
        options = ImportOptions(
            task=None if all_tasks else ANALYZE_TASK,
            windowed_only=not include_unwindowed,
            meeting_id=meeting_id,
            limit=limit,
        )
        cases = import_cases(resolve_path(root, db), options)
        target = write_private_dataset(cases, resolve_path(root, out), force=force, root=root)
        typer.echo(f"wrote {len(cases)} cases to {target}. Add it to paths.datasets to use it.")
    except ModebenchError as error:
        raise _fail(error) from error
