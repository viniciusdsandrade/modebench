"""The command line. Each test that runs a suite uses a copy of the inputs below `tmp_path`."""

import re
import sqlite3
import stat
from pathlib import Path

from typer.testing import CliRunner, Result

from modebench.cli import EXIT_ERROR, app
from modebench.config import load_bench_config
from modebench.stats.aggregate import summarize_run, write_summary
from modebench.storage.db import DB_NAME, RunStore

runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "160"})


def plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


def invoke(root: Path, command: str, *arguments: str) -> Result:
    return runner.invoke(app, [command, "--root", str(root), *arguments])


def output(result: Result) -> str:
    return plain(result.output)


def dry_run(root: Path, *arguments: str) -> str:
    """Do a smoke dry run in `root` and return its run identifier."""
    result = invoke(root, "run", "--dry-run", "--profile", "smoke", *arguments)
    assert result.exit_code == 0, output(result)
    match = re.search(r"run (\S+): completed", output(result))
    assert match is not None
    return match.group(1)


def test_cli_help_displays_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = plain(result.stdout)
    assert "run" in out
    assert "decide" in out
    assert "report" in out
    assert "compare" in out


def test_cli_run_help() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    out = plain(result.stdout)
    assert "--dry-run" in out
    assert "--profile" in out
    assert "--estimate-only" in out


def test_cli_compare_help_names_its_options() -> None:
    result = runner.invoke(app, ["compare", "--help"])
    assert result.exit_code == 0
    out = plain(result.stdout)
    assert "--baseline" in out
    assert "--run" in out
    assert "--allow-dry-run" in out


def test_cli_unknown_suite_fails_with_exit_error() -> None:
    result = runner.invoke(app, ["run", "--suite", "unknown-suite"])
    assert result.exit_code == EXIT_ERROR
    assert "unknown suite" in output(result)


def test_cli_estimate_only_writes_nothing(isolated_root: Path) -> None:
    result = invoke(
        isolated_root, "run", "--dry-run", "--estimate-only", "--profile", "smoke", "--no-judge"
    )
    assert result.exit_code == 0
    assert "plan:" in output(result)
    assert not (isolated_root / "runs").exists()


def test_cli_a_mode_named_two_times_runs_one_time(isolated_root: Path) -> None:
    result = invoke(
        isolated_root,
        "run",
        "--dry-run",
        "--estimate-only",
        "--no-judge",
        "--modes",
        "glm-5.3-flash-or, glm-5.3-flash-or",
    )
    assert result.exit_code == 0
    assert "1 modes" in output(result)


def test_cli_dry_run_then_decide_and_report(isolated_root: Path, tmp_path: Path) -> None:
    run_id = dry_run(isolated_root)
    run_dir = isolated_root / "runs" / run_id

    refused = invoke(isolated_root, "decide")
    assert refused.exit_code == EXIT_ERROR
    assert "dry run" in output(refused)

    decided = invoke(isolated_root, "decide", "--allow-dry-run")
    assert decided.exit_code == 0, output(decided)
    assert (run_dir / "recommended_modes.toml").is_file()
    # Fake numbers never go to the file that the application reads.
    assert not (isolated_root / "recommended_modes.toml").exists()

    chosen = tmp_path / "rec.toml"
    explicit = invoke(isolated_root, "decide", "--allow-dry-run", "--out", str(chosen))
    assert explicit.exit_code == 0
    assert "dry_run = true" in chosen.read_text(encoding="utf-8")

    reported = invoke(isolated_root, "report")
    assert reported.exit_code == 0, output(reported)
    text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Dry run" in text
    assert "## Modes" in text


def test_cli_a_run_with_no_judge_has_no_quality_and_no_decision(isolated_root: Path) -> None:
    run_id = dry_run(isolated_root, "--no-judge", "--modes", "glm-5.3-flash-or")
    decided = invoke(isolated_root, "decide", "--allow-dry-run")
    assert decided.exit_code == 0, output(decided)
    text = (isolated_root / "runs" / run_id / "recommended_modes.toml").read_text(encoding="utf-8")
    assert "no quality data" in text
    assert "question accuracy" not in text


def test_cli_compare_and_baseline_refuse_a_dry_run(isolated_root: Path, tmp_path: Path) -> None:
    run_id = dry_run(isolated_root, "--modes", "glm-5.3-flash-or")
    bench = load_bench_config(isolated_root / "configs" / "bench.toml")
    store = RunStore(isolated_root / "runs" / DB_NAME)
    try:
        summary = summarize_run(store, run_id, bench.stats)
    finally:
        store.close()
    baseline = write_summary(summary, tmp_path / "baseline.json")

    refused = invoke(isolated_root, "compare", "--baseline", str(baseline))
    assert refused.exit_code == EXIT_ERROR
    assert "dry run" in output(refused)

    same = invoke(isolated_root, "compare", "--baseline", str(baseline), "--allow-dry-run")
    assert same.exit_code == 0, output(same)
    assert "no regression" in output(same)

    absent = invoke(isolated_root, "compare", "--baseline", str(tmp_path / "absent.json"))
    assert absent.exit_code == EXIT_ERROR
    assert "baseline not found" in output(absent)

    made = invoke(isolated_root, "baseline", "--out", str(tmp_path / "made.json"))
    assert made.exit_code == EXIT_ERROR
    assert "dry run" in output(made)

    no_file = invoke(isolated_root, "report", "--baseline", str(tmp_path / "absent.json"))
    assert no_file.exit_code == EXIT_ERROR
    assert "baseline not found" in output(no_file)

    with_baseline = invoke(isolated_root, "report", "--baseline", str(baseline))
    assert with_baseline.exit_code == 0, output(with_baseline)
    report = (isolated_root / "runs" / run_id / "report.md").read_text(encoding="utf-8")
    assert "Difference from the baseline" in report


def test_cli_stt_dry_run_report_and_suite_guard(isolated_root: Path) -> None:
    result = invoke(isolated_root, "run", "--suite", "stt", "--dry-run", "--profile", "smoke")
    assert result.exit_code == 0, output(result)
    match = re.search(r"run (\S+): completed", output(result))
    assert match is not None
    run_id = match.group(1)

    reported = invoke(isolated_root, "report", "--run", run_id)
    assert reported.exit_code == 0, output(reported)
    text = (isolated_root / "runs" / run_id / "stt_report.md").read_text(encoding="utf-8")
    assert "speech to text report" in text

    wrong_suite = invoke(isolated_root, "decide", "--run", run_id, "--allow-dry-run")
    assert wrong_suite.exit_code == EXIT_ERROR
    assert "suite stt" in output(wrong_suite)


def _application_database(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE segments (id INTEGER PRIMARY KEY, meeting_id INTEGER, source TEXT, "
        "ts_start_ms INTEGER, text TEXT)"
    )
    connection.execute(
        "CREATE TABLE interpretations (id INTEGER PRIMARY KEY, meeting_id INTEGER, task TEXT, "
        "up_to_ts_ms INTEGER, up_to_segment_id INTEGER, model TEXT, text TEXT, status TEXT, "
        "windowed INTEGER)"
    )
    connection.execute("INSERT INTO segments VALUES (1, 1, 'mic', 1000, 'Qual e o prazo')")
    connection.execute(
        "INSERT INTO interpretations VALUES "
        "(10, 1, 'recent_question', 2500, 1, 'gemini', 'Seis semanas.', 'completed', 1)"
    )
    connection.commit()
    connection.close()
    return path


def test_cli_import_sessions_keeps_private_data_out_of_tracked_directories(
    isolated_root: Path, tmp_path: Path
) -> None:
    database = _application_database(tmp_path / "app.db")

    tracked = invoke(
        isolated_root, "import-sessions", "--db", str(database), "--out", "data/public/leak.yaml"
    )
    assert tracked.exit_code == EXIT_ERROR
    assert "data/private" in output(tracked)
    assert not (isolated_root / "data" / "public" / "leak.yaml").exists()

    imported = invoke(isolated_root, "import-sessions", "--db", str(database))
    assert imported.exit_code == 0, output(imported)
    written = isolated_root / "data" / "private" / "sttbridge_sessions.yaml"
    assert "visibility: private" in written.read_text(encoding="utf-8")
    assert stat.S_IMODE(written.stat().st_mode) == 0o600

    again = invoke(isolated_root, "import-sessions", "--db", str(database))
    assert again.exit_code == EXIT_ERROR
    assert "--force" in output(again)
