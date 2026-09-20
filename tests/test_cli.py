"""Tests for command-line interface: help, subcommands, options, and exit codes."""

from typer.testing import CliRunner

from helpers import REPO_ROOT
from modebench.cli import (
    EXIT_ERROR,
    app,
)

runner = CliRunner()


def test_cli_help_displays_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "decide" in result.stdout
    assert "report" in result.stdout
    assert "compare" in result.stdout


def test_cli_run_help() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.stdout
    assert "--profile" in result.stdout
    assert "--estimate-only" in result.stdout


def test_cli_decide_help() -> None:
    result = runner.invoke(app, ["decide", "--help"])
    assert result.exit_code == 0
    assert "--run" in result.stdout


def test_cli_report_help() -> None:
    result = runner.invoke(app, ["report", "--help"])
    assert result.exit_code == 0
    assert "--run" in result.stdout


def test_cli_compare_help() -> None:
    result = runner.invoke(app, ["compare", "--help"])
    assert result.exit_code == 0
    assert "--baseline" in result.stdout
    assert "--p95-ratio" in result.stdout


def test_cli_unknown_suite_fails_with_exit_error() -> None:
    result = runner.invoke(app, ["run", "--suite", "unknown-suite"])
    assert result.exit_code == EXIT_ERROR
    assert "unknown suite" in result.stderr or "unknown suite" in result.stdout


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
    assert "plan:" in result.stdout
