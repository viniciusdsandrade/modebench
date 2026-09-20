import re
from pathlib import Path

from typer.testing import CliRunner

from helpers import REPO_ROOT
from modebench.cli import (
    EXIT_ERROR,
    app,
)

runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "160"})


def plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


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


def test_cli_decide_help() -> None:
    result = runner.invoke(app, ["decide", "--help"])
    assert result.exit_code == 0
    out = plain(result.stdout)
    assert "--run" in out


def test_cli_report_help() -> None:
    result = runner.invoke(app, ["report", "--help"])
    assert result.exit_code == 0
    out = plain(result.stdout)
    assert "--run" in out


def test_cli_compare_help() -> None:
    result = runner.invoke(app, ["compare", "--help"])
    assert result.exit_code == 0
    out = plain(result.stdout)
    assert "--baseline" in out
    assert "--p95-ratio" in out


def test_cli_unknown_suite_fails_with_exit_error() -> None:
    result = runner.invoke(app, ["run", "--suite", "unknown-suite"])
    assert result.exit_code == EXIT_ERROR
    out = plain(result.stderr or result.stdout)
    assert "unknown suite" in out


def test_cli_run_estimate_only_smoke_dry_run() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--root",
            str(REPO_ROOT),
            "--dry-run",
            "--estimate-only",
            "--profile",
            "smoke",
            "--skip-preflight",
            "--no-judge",
        ],
    )
    assert result.exit_code == 0
    assert "plan:" in plain(result.stdout)


def test_cli_full_dry_run_and_reporting(tmp_path: Path) -> None:
    rec_out = tmp_path / "rec.toml"
    result_run = runner.invoke(
        app,
        [
            "run",
            "--root",
            str(REPO_ROOT),
            "--dry-run",
            "--profile",
            "smoke",
            "--skip-preflight",
            "--no-judge",
        ],
    )
    assert result_run.exit_code == 0
    assert "plan:" in plain(result_run.stdout)

    result_decide = runner.invoke(
        app,
        [
            "decide",
            "--root",
            str(REPO_ROOT),
            "--allow-dry-run",
            "--out",
            str(rec_out),
        ],
    )
    assert result_decide.exit_code == 0
    assert rec_out.is_file()

    result_report = runner.invoke(
        app,
        [
            "report",
            "--root",
            str(REPO_ROOT),
        ],
    )
    assert result_report.exit_code == 0


def test_cli_stt_dry_run() -> None:
    result_stt = runner.invoke(
        app,
        [
            "run",
            "--suite",
            "stt",
            "--root",
            str(REPO_ROOT),
            "--dry-run",
            "--profile",
            "smoke",
        ],
    )
    assert result_stt.exit_code == 0
    assert "run" in plain(result_stt.stdout)
