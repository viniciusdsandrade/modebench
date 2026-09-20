"""The configuration models: what they accept, what they refuse, and the TOML writer."""

import math
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from helpers import openrouter_mode
from modebench.config import (
    MeetingConfig,
    Mode,
    ModesFile,
    Profile,
    ProviderConfig,
    QualityWeights,
    load_bench_config,
    load_modes_file,
    load_toml,
    resolve_path,
)
from modebench.decision.toml_writer import dumps, format_value
from modebench.errors import ConfigError

FAKE = ProviderConfig(kind="fake", privacy="local")


def test_the_seed_files_are_valid(repo_root: Path) -> None:
    bench = load_bench_config(repo_root / "configs" / "bench.toml")
    modes = load_modes_file(repo_root / "configs" / "modes.toml")
    assert set(bench.profiles) == {"smoke", "quick", "full"}
    assert bench.profile("smoke").repetitions == 1
    assert len(modes.select()) == len(modes.modes)
    assert [mode.id for mode in modes.select(["glm-5.3-flash-or"])] == ["glm-5.3-flash-or"]
    with pytest.raises(ConfigError, match="unknown mode"):
        modes.select(["absent"])
    with pytest.raises(ConfigError, match="unknown profile"):
        bench.profile("absent")
    assert resolve_path(repo_root, Path("configs")) == repo_root / "configs"
    assert resolve_path(repo_root, Path("/etc")) == Path("/etc")


def test_config_files_that_are_absent_or_not_valid_are_config_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_toml(tmp_path / "absent.toml")
    (tmp_path / "broken.toml").write_text("[unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_toml(tmp_path / "broken.toml")
    (tmp_path / "empty.toml").write_text("unknown_key = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a valid bench file"):
        load_bench_config(tmp_path / "empty.toml")
    with pytest.raises(ConfigError, match="not a valid modes file"):
        load_modes_file(tmp_path / "empty.toml")


def test_a_modes_file_checks_its_references() -> None:
    mode = Mode(id="a", provider="local", model="fake/a")
    with pytest.raises(ValidationError, match="duplicate mode ids: a"):
        ModesFile(providers={"local": FAKE}, modes=[mode, mode])
    with pytest.raises(ValidationError, match="unknown provider elsewhere"):
        ModesFile(
            providers={"local": FAKE}, modes=[mode.model_copy(update={"provider": "elsewhere"})]
        )
    with pytest.raises(ValidationError, match="unknown mode absent"):
        ModesFile(
            providers={"local": FAKE}, modes=[mode.model_copy(update={"overhead_of": "absent"})]
        )
    with pytest.raises(ValidationError, match="needs base_url and api_key_env"):
        ProviderConfig(kind="openai_compat")
    # The identifier goes into Markdown tables as it is.
    for bad in ("with space", "pipe|name", "tick`name"):
        with pytest.raises(ValidationError):
            Mode(id=bad, provider="local", model="fake/a")


def test_the_family_and_the_reasoning_level_of_a_mode() -> None:
    assert openrouter_mode("a", model="Google/gemini-test").resolved_family() == "google"
    assert Mode(id="a", provider="google", model="models/gemini-3").resolved_family() == "google"
    assert Mode(id="a", provider="direct", model="claude-x").resolved_family() == "anthropic"
    assert Mode(id="a", provider="Direct", model="unknown-model").resolved_family() == "direct"
    assert Mode(id="a", provider="x", model="m", family="Own").resolved_family() == "own"

    assert openrouter_mode("a", reasoning={"effort": "high"}).reasoning_effort() == "high"
    assert openrouter_mode("a", reasoning={"enabled": False}).reasoning_effort() == "none"
    assert openrouter_mode("a", reasoning_effort="low").reasoning_effort() == "low"
    assert openrouter_mode("a").reasoning_effort() == "default"
    assert openrouter_mode("a").serves_role("anything")
    limited = openrouter_mode("a").model_copy(update={"roles": ["summary"]})
    assert limited.serves_role("summary") and not limited.serves_role("analysis")


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"truncations": [0]}, "percentage from 1 to 100"),
        ({"durations_min": [0], "baseline_duration_min": 0}, "minutes above 0"),
        ({"asr_wers": [0.6]}, "at most 0.5"),
        ({"baseline_duration_min": 15}, "must be one of durations_min"),
    ],
)
def test_a_profile_checks_its_ranges(fields: dict[str, object], message: str) -> None:
    base: dict[str, object] = {
        "truncations": [100],
        "durations_min": [5],
        "baseline_duration_min": 5,
        "max_cost_usd": 1.0,
    }
    with pytest.raises(ValidationError, match=message):
        Profile.model_validate({**base, **fields})


def test_the_weights_add_up_to_one_and_the_clicks_increase() -> None:
    with pytest.raises(ValidationError, match="add up to 1"):
        QualityWeights(question=0.5, utility=0.5, key_points=0.5)
    with pytest.raises(ValidationError, match="must increase"):
        MeetingConfig(click_minutes=[15, 5])


def test_the_toml_writer_gives_a_document_that_the_standard_reader_reads() -> None:
    document = {
        "name": 'a "quoted" \\ value\nwith a line and a tab\t and a bell \x07',
        "count": 3,
        "ratio": 0.25,
        "whole": 2.0,
        "tiny": 1e-07,
        "flag": True,
        "absent": None,
        "list": [1, None, "two", {"inline": {"deep": 1}, "skip": None}],
        "empty inline": [{}],
        "roles": {"key with space": {"mode": "m1", "params": {"reasoning": {"effort": "low"}}}},
    }
    text = dumps(document, header="Generated.\nDo not edit.")
    assert text.startswith("# Generated.\n# Do not edit.\n")
    parsed = tomllib.loads(text)
    assert parsed["name"] == document["name"]
    assert parsed["whole"] == 2.0 and isinstance(parsed["whole"], float)
    assert parsed["tiny"] == 1e-07
    assert "absent" not in parsed
    assert parsed["list"] == [1, "two", {"inline": {"deep": 1}}]
    assert parsed["empty inline"] == [{}]
    assert parsed["roles"]["key with space"]["params"]["reasoning"] == {"effort": "low"}
    assert format_value(math.inf) == "inf"
    assert format_value(-math.inf) == "-inf"
    assert format_value(math.nan) == "nan"
    with pytest.raises(TypeError, match="cannot hold"):
        format_value(object())
