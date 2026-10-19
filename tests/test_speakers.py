"""The speaker score: gold set, hypothesis, match, WDER, names, echo and the command line."""

import itertools
import json
import random
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from modebench.cli import EXIT_ERROR, app
from modebench.errors import DatasetError
from modebench.speakers.external import diarization_rates, permutation_rates
from modebench.speakers.gold import (
    GoldMeeting,
    Turn,
    WordSegment,
    load_gold,
    parse_rttm,
    parse_seglst,
    parse_uem,
)
from modebench.speakers.hypothesis import (
    Hypothesis,
    HypothesisName,
    load_hypothesis,
    read_hypothesis,
)
from modebench.speakers.mapping import best_mapping, mapped_total
from modebench.speakers.metrics import (
    echo_rate,
    fold_name,
    name_accuracy,
    name_checks,
    operator_confusion,
    overlap_table,
    time_mapping,
    word_diarization,
)
from modebench.speakers.score import render_score, score_meeting, write_score

runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "160"})

RTTM = """\
# who spoke when in meeting m1
SPKR-INFO m1 1 <NA> <NA> <NA> unknown mic-op <NA> <NA>
SPEAKER m1 1 0.00 2.00 <NA> <NA> mic-op <NA> <NA>
SPEAKER m1 1 2.00 3.00 <NA> <NA> sys-ana <NA> <NA>
SPEAKER m1 1 5.00 2.00 <NA> <NA> sys-pedro <NA> <NA>
"""

UEM = "m1 1 0.00 7.00\n"

WORDS = [
    {
        "session_id": "m1",
        "speaker": "mic-op",
        "start_time": 0.0,
        "end_time": 2.0,
        "words": "bom dia a todos",
    },
    {
        "session_id": "m1",
        "speaker": "sys-ana",
        "start_time": 2.0,
        "end_time": 5.0,
        "words": "aqui é a Ana obrigada",
    },
    {
        "session_id": "m1",
        "speaker": "sys-pedro",
        "start_time": 5.0,
        "end_time": 7.0,
        "words": "valeu Ana até logo",
    },
]


def segment(
    number: int,
    source: str,
    start_ms: int,
    end_ms: int,
    text: str,
    label: str | None,
    origin: str | None,
    speaker: str | None = None,
) -> dict[str, Any]:
    """Return one line of the detail of stt-bridge."""
    line: dict[str, Any] = {
        "id": number,
        "source": source,
        "ts_start_ms": start_ms,
        "ts_end_ms": end_ms,
        "text": text,
        "provider": "assemblyai" if source == "system" else "elevenlabs",
        "translations": [],
    }
    if speaker is not None:
        line["speaker"] = speaker
    if label is not None:
        line["speaker_label"] = label
    if origin is not None:
        line["speaker_origin"] = origin
    return line


def detail(pedro_label: str = "B") -> dict[str, Any]:
    """Return a meeting recorded with fusion: the operator, Ana with her echo, and Pedro."""
    return {
        "id": 12,
        "title": "planning",
        "started_at": "2026-10-01T13:00:00Z",
        "ended_at": "2026-10-01T13:40:00Z",
        "segments": [
            segment(1, "mic", 0, 2000, "Bom dia a todos.", "me", "channel"),
            segment(2, "system", 2000, 5000, "Aqui é a Ana, obrigada.", "A", "provider", "Ana"),
            segment(3, "mic", 2100, 5000, "aqui é a Ana obrigada", "A", "echo", "Ana"),
            segment(4, "system", 5000, 7000, "Valeu, Ana. Até logo.", pedro_label, "provider"),
        ],
        "interpretations": [],
        "participants": [],
        "speakers": [
            {"label": "me", "kind": "local", "name": None, "name_origin": None},
            {
                "label": "A",
                "kind": "remote",
                "name": "Ana",
                "name_origin": "rule",
                "name_confidence_pct": 85,
            },
            {"label": "B", "kind": "remote", "name": None, "name_origin": None},
        ],
    }


def unattributed() -> dict[str, Any]:
    """Return the same meeting recorded with attribution off."""
    return {
        "id": 12,
        "segments": [
            segment(1, "mic", 0, 2000, "Bom dia a todos.", None, None),
            segment(2, "system", 2000, 5000, "Aqui é a Ana, obrigada.", None, None, "A"),
            segment(3, "mic", 2100, 5000, "aqui é a Ana obrigada", None, None, "A"),
            segment(4, "system", 5000, 7000, "Valeu, Ana. Até logo.", None, None, "B"),
        ],
    }


def gold(**changes: Any) -> GoldMeeting:
    """Return the gold set of meeting m1."""
    values: dict[str, Any] = {
        "id": "m1",
        "turns": parse_rttm(RTTM, "m1.rttm"),
        "scored": [(0.0, 7.0)],
        "words": parse_seglst(json.dumps(WORDS), "m1.seglst.json"),
        "names": {"sys-ana": ("Ana Souza", "Ana"), "sys-pedro": ("Pedro Lima",)},
    }
    values.update(changes)
    return GoldMeeting(**values)


def write_gold_set(base: Path, index: str | None = None) -> Path:
    """Write the gold set of meeting m1 below `base` and return its index."""
    base.mkdir(parents=True, exist_ok=True)
    (base / "m1.rttm").write_text(RTTM, encoding="utf-8")
    (base / "m1.uem").write_text(UEM, encoding="utf-8")
    (base / "m1.seglst.json").write_text(json.dumps(WORDS), encoding="utf-8")
    path = base / "gold.yaml"
    path.write_text(
        index
        or (
            "meetings:\n"
            "  - id: m1\n"
            "    rttm: m1.rttm\n"
            "    uem: m1.uem\n"
            "    words: m1.seglst.json\n"
            "    names:\n"
            "      sys-ana: [Ana Souza, Ana]\n"
            "      sys-pedro: Pedro Lima\n"
            "    offset_s: 0.5\n"
        ),
        encoding="utf-8",
    )
    return path


# The gold set.


def test_rttm_reads_the_speaker_lines_and_ignores_the_others() -> None:
    turns = parse_rttm(RTTM, "m1.rttm")
    assert turns == [
        Turn("mic-op", 0.0, 2.0),
        Turn("sys-ana", 2.0, 5.0),
        Turn("sys-pedro", 5.0, 7.0),
    ]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("SPEAKER m1 1 0 1 <NA> <NA> a <NA>\nSPEAKER m2 1 1 1 <NA> <NA> b <NA>\n", "more than one"),
        ("SPEAKER m1 1 -1 1 <NA> <NA> a <NA>\n", "negative"),
        ("SPEAKER m1 1 zero 1 <NA> <NA> a <NA>\n", "not a number"),
        ("SPEAKER m1 1 0 1\n", "8 fields"),
        ("# nothing\n", "no SPEAKER line"),
    ],
)
def test_rttm_refuses_a_file_that_is_not_one_meeting_of_turns(text: str, message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        parse_rttm(text, "bad.rttm")


def test_uem_reads_the_scored_parts() -> None:
    assert parse_uem("# scored\nm1 1 0.0 3.5\nm1 1 4.0 7.0\n", "m1.uem") == [
        (0.0, 3.5),
        (4.0, 7.0),
    ]


@pytest.mark.parametrize(
    ("text", "message"),
    [("m1 1 3.0 2.0\n", "not after"), ("m1 1 0.0\n", "4 fields"), ("\n", "no scored part")],
)
def test_uem_refuses_a_part_that_ends_before_it_starts(text: str, message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        parse_uem(text, "bad.uem")


def test_seglst_reads_the_word_segments() -> None:
    segments = parse_seglst(json.dumps(WORDS), "m1.seglst.json")
    assert segments[1] == WordSegment("sys-ana", 2.0, 5.0, "aqui é a Ana obrigada")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{", "not valid JSON"),
        ('{"words": "a"}', "not a list"),
        ('["a"]', "is an object"),
        ('[{"speaker": "a", "words": "b"}]', "are needed"),
        ('[{"speaker": 1, "start_time": 0, "end_time": 1, "words": "b"}]', "are text"),
        (
            '[{"session_id": "m1", "speaker": "a", "start_time": 0, "end_time": 1, "words": "b"},'
            ' {"session_id": "m2", "speaker": "a", "start_time": 1, "end_time": 2, "words": "c"}]',
            "more than one meeting",
        ),
    ],
)
def test_seglst_refuses_a_segment_without_its_fields(text: str, message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        parse_seglst(text, "bad.json")


def test_gold_set_reads_a_meeting_of_the_index(tmp_path: Path) -> None:
    meeting = load_gold(write_gold_set(tmp_path / "gold"), "m1")

    assert [turn.speaker for turn in meeting.turns] == ["mic-op", "sys-ana", "sys-pedro"]
    assert meeting.scored == [(0.0, 7.0)]
    assert len(meeting.words) == 3
    assert meeting.names == {"sys-ana": ("Ana Souza", "Ana"), "sys-pedro": ("Pedro Lima",)}
    assert meeting.offset_s == 0.5


def test_gold_set_without_a_uem_scores_from_the_first_turn_to_the_last(tmp_path: Path) -> None:
    index = "meetings:\n  - id: m1\n    rttm: m1.rttm\n"
    meeting = load_gold(write_gold_set(tmp_path / "gold", index), "m1")

    assert meeting.scored == [(0.0, 7.0)]
    assert meeting.words == []
    assert meeting.names == {}


@pytest.mark.parametrize(
    ("index", "meeting", "message"),
    [
        ("meetings:\n  - id: m1\n    rttm: m1.rttm\n", "m9", "not in .*The index has: m1"),
        (
            "meetings:\n  - id: m1\n    rttm: m1.rttm\n    names:\n      sys-zé: Zé\n",
            "m1",
            "not a speaker",
        ),
        (
            "meetings:\n  - id: m1\n    rttm: m1.rttm\n    names:\n      sys-ana: []\n",
            "m1",
            "empty",
        ),
        ("meetings:\n  - id: m1\n    rttm: gone.rttm\n", "m1", "not found"),
        (
            "meetings:\n  - id: m1\n    rttm: m1.rttm\n  - id: m1\n    rttm: m1.rttm\n",
            "m1",
            "two times",
        ),
        ("meetings: [", "m1", "not valid YAML"),
        ("meetings:\n  - id: m1\n    rttm: m1.rttm\n    colour: red\n", "m1", "not a valid"),
    ],
)
def test_gold_set_refuses_an_index_that_does_not_hold_together(
    tmp_path: Path, index: str, meeting: str, message: str
) -> None:
    with pytest.raises(DatasetError, match=message):
        load_gold(write_gold_set(tmp_path / "gold", index), meeting)


def test_gold_set_refuses_words_of_a_speaker_the_turns_do_not_have(tmp_path: Path) -> None:
    base = tmp_path / "gold"
    index = write_gold_set(
        base, "meetings:\n  - id: m1\n    rttm: m1.rttm\n    words: extra.json\n"
    )
    (base / "extra.json").write_text(
        json.dumps([{"speaker": "sys-zoe", "start_time": 0, "end_time": 1, "words": "oi"}]),
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="sys-zoe"):
        load_gold(index, "m1")


# The hypothesis.


def test_hypothesis_takes_the_meeting_label_and_the_names_of_the_speakers() -> None:
    document = detail()
    document["speakers"].append("not a speaker")
    hypothesis = read_hypothesis(document, mode="fusion")

    assert [line.speaker for line in hypothesis.lines] == ["me", "A", "A", "B"]
    assert [line.origin for line in hypothesis.lines] == ["channel", "provider", "echo", "provider"]
    assert hypothesis.names == {"A": HypothesisName("Ana", "rule", 85)}
    assert hypothesis.local == frozenset({"me"})
    assert hypothesis.speakers() == ["me", "A", "B"]
    assert hypothesis.meeting_id == 12
    assert hypothesis.lines[2].start == pytest.approx(2.1)


def test_hypothesis_reads_the_export_of_the_page_and_moves_the_times() -> None:
    hypothesis = read_hypothesis({"meeting": detail(), "cost": None}, mode="native", offset_s=1.5)

    assert hypothesis.lines[0].start == pytest.approx(1.5)
    assert hypothesis.lines[0].end == pytest.approx(3.5)


def test_hypothesis_without_attribution_keeps_each_label_on_its_channel() -> None:
    hypothesis = read_hypothesis(unattributed(), mode="off")

    assert [line.speaker for line in hypothesis.lines] == [
        "mic:?",
        "system:A",
        "mic:A",
        "system:B",
    ]
    assert hypothesis.names == {}


def test_hypothesis_puts_an_untimed_line_at_its_start() -> None:
    document = unattributed()
    document["segments"][0]["ts_start_ms"] = 900
    document["segments"][0]["ts_end_ms"] = 0

    line = next(
        item for item in read_hypothesis(document, mode="off").lines if item.segment_id == 1
    )
    assert line.end == line.start


@pytest.mark.parametrize(
    ("document", "mode", "message"),
    [
        (unattributed(), "native", "use --mode off"),
        (detail(), "off", "use --mode native or --mode fusion"),
        (detail(), "loud", "unknown mode"),
        ({"segments": "none"}, "off", "no segments"),
        ({"segments": ["line"]}, "off", "not an object"),
        ({"segments": [{"source": "radio"}]}, "off", "unknown channel"),
        ({"segments": [{"source": "mic", "id": 1}]}, "off", "no id or no times"),
    ],
)
def test_hypothesis_refuses_what_is_not_the_detail_of_a_meeting(
    document: object, mode: str, message: str
) -> None:
    with pytest.raises(DatasetError, match=message):
        read_hypothesis(document, mode=mode)


def test_a_directory_is_not_a_file_of_the_gold_set_or_a_hypothesis(tmp_path: Path) -> None:
    base = tmp_path / "gold"
    index = write_gold_set(base, "meetings:\n  - id: m1\n    rttm: folder\n")
    (base / "folder").mkdir()

    with pytest.raises(DatasetError, match="cannot be read"):
        load_gold(index, "m1")
    with pytest.raises(DatasetError, match="cannot be read"):
        load_hypothesis(base / "folder", mode="off")


def test_hypothesis_file_must_be_there_and_be_json(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_hypothesis(tmp_path / "meeting-12.json", mode="off")

    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(DatasetError, match="not valid JSON"):
        load_hypothesis(broken, mode="off")

    good = tmp_path / "meeting-12.json"
    good.write_text(json.dumps(detail()), encoding="utf-8")
    assert len(load_hypothesis(good, mode="fusion").lines) == 4


# The match of speakers.


def test_match_takes_the_largest_total_and_not_the_largest_cell() -> None:
    # A greedy match takes A to x for 5 and is left with 0. The best match is 3 + 4.
    table = {"A": {"x": 5.0, "y": 3.0}, "B": {"x": 4.0}}

    assert best_mapping(table) == {"A": "y", "B": "x"}
    assert mapped_total(table, best_mapping(table)) == 7.0


def test_match_leaves_out_a_speaker_with_no_agreement() -> None:
    assert best_mapping({"A": {"x": 2.0}, "B": {"x": 3.0}, "C": {"y": 1.0}}) == {
        "B": "x",
        "C": "y",
    }
    assert best_mapping({"A": {"x": 0.0}}) == {}
    assert best_mapping({}) == {}


def _brute_force(table: dict[str, dict[str, float]]) -> float:
    rows = sorted(table)
    columns = sorted({column for row in table.values() for column in row})
    slots: list[str | None] = [*columns, *([None] * len(rows))]
    best = 0.0
    for chosen in itertools.permutations(slots, len(rows)):
        total = sum(
            table[row].get(column, 0.0)
            for row, column in zip(rows, chosen, strict=True)
            if column is not None
        )
        best = max(best, total)
    return best


@pytest.mark.parametrize("seed", range(40))
def test_match_is_as_good_as_every_match_tried_one_by_one(seed: int) -> None:
    chance = random.Random(seed)
    rows = [f"h{index}" for index in range(chance.randint(1, 4))]
    columns = [f"r{index}" for index in range(chance.randint(1, 4))]
    table = {
        row: {column: float(chance.randint(0, 9)) for column in columns if chance.random() < 0.7}
        for row in rows
    }

    assert mapped_total(table, best_mapping(table)) == _brute_force(table)


# WDER.


def test_wder_is_zero_when_each_word_is_with_its_speaker() -> None:
    measured = word_diarization(gold().words, read_hypothesis(detail(), mode="fusion").lines)

    assert measured is not None
    assert measured.wder == 0.0
    assert measured.wer == 0.0
    assert measured.correct == 13
    assert measured.mapping == {"A": "sys-ana", "B": "sys-pedro", "me": "mic-op"}


def test_wder_counts_the_words_of_two_people_under_one_label() -> None:
    measured = word_diarization(gold().words, read_hypothesis(detail("A"), mode="fusion").lines)

    assert measured is not None
    # Pedro's four words are under Ana's label, and Ana keeps the label.
    assert measured.wrong_speaker == 4
    assert measured.wder == pytest.approx(4 / 13)


def test_wder_reads_a_substitution_as_a_word_of_its_speaker() -> None:
    document = detail()
    document["segments"][0]["text"] = "bom dia a todas"
    measured = word_diarization(gold().words, read_hypothesis(document, mode="fusion").lines)

    assert measured is not None
    assert measured.substitutions == 1
    assert measured.wder == 0.0
    assert measured.wer == pytest.approx(1 / 13)


def test_wder_leaves_deletions_to_the_wer() -> None:
    document = detail()
    document["segments"][0]["text"] = "bom dia todos"
    measured = word_diarization(gold().words, read_hypothesis(document, mode="fusion").lines)

    assert measured is not None
    assert measured.deletions == 1
    assert measured.wder == 0.0
    assert measured.wer == pytest.approx(1 / 13)


def test_wer_counts_an_echo_that_was_not_marked() -> None:
    measured = word_diarization(gold().words, read_hypothesis(unattributed(), mode="off").lines)

    assert measured is not None
    assert measured.insertions == 5
    assert measured.wer == pytest.approx(5 / 13)


def test_wder_needs_reference_words_and_says_when_nothing_was_heard() -> None:
    assert word_diarization([], read_hypothesis(detail(), mode="fusion").lines) is None

    measured = word_diarization(gold().words, [])
    assert measured is not None
    assert measured.wder is None
    assert measured.wer == 1.0
    assert measured.deletions == 13


# Names and echo.


def test_time_match_follows_the_shared_seconds() -> None:
    lines = read_hypothesis(detail(), mode="fusion").lines
    table = overlap_table(gold().turns, lines)

    assert table["A"]["sys-ana"] == pytest.approx(5.9)
    assert time_mapping(gold().turns, lines) == {"A": "sys-ana", "B": "sys-pedro", "me": "mic-op"}


def test_names_are_counted_from_both_sides() -> None:
    hypothesis = read_hypothesis(detail(), mode="fusion")
    accuracy = name_accuracy(gold().names, hypothesis, time_mapping(gold().turns, hypothesis.lines))

    # Ana is named and right; Pedro has no name. The one name given is right.
    assert accuracy.recall == 0.5
    assert accuracy.precision == 1.0
    assert accuracy.named_reference == 2
    assert accuracy.named_hypothesis == 1


def test_a_wrong_name_counts_against_both_sides() -> None:
    hypothesis = Hypothesis(
        meeting_id=1, lines=[], names={"A": HypothesisName("Pedro Lima", "model", 80)}
    )
    accuracy = name_accuracy(gold().names, hypothesis, {"A": "sys-ana"})

    assert accuracy.recall == 0.0
    assert accuracy.precision == 0.0
    assert name_accuracy({}, Hypothesis(1, [], {}), {}).recall is None


def test_names_are_compared_without_accents_and_case() -> None:
    assert fold_name("  João   da SILVA ") == "joao da silva"
    hypothesis = Hypothesis(meeting_id=1, lines=[], names={"A": HypothesisName("joao", "rule")})
    accuracy = name_accuracy({"sys-joao": ("João",)}, hypothesis, {"A": "sys-joao"})
    assert accuracy.recall == 1.0


def test_a_name_the_application_was_given_measures_nothing() -> None:
    # The configuration names the operator, and the gold set agrees: that is
    # no name the application found.
    given = Hypothesis(
        meeting_id=1, lines=[], names={"me": HypothesisName("Marina", "config", 100)}
    )
    accuracy = name_accuracy({"mic-op": ("Marina",)}, given, {"me": "mic-op"})
    assert accuracy.recall is None
    assert accuracy.precision is None

    # A found name whose speaker the gold set does not name cannot be judged.
    unjudged = Hypothesis(meeting_id=1, lines=[], names={"B": HypothesisName("Pedro", "rule", 80)})
    accuracy = name_accuracy(gold().names, unjudged, {"B": "mic-op"})
    assert accuracy.precision is None
    checks = name_checks(gold().names, unjudged, {"B": "mic-op"})
    assert checks[0].correct is None
    assert checks[0].reference == "mic-op"


def test_each_name_is_shown_with_its_origin_its_confidence_and_its_verdict() -> None:
    hypothesis = read_hypothesis(detail(), mode="fusion")
    checks = name_checks(gold().names, hypothesis, time_mapping(gold().turns, hypothesis.lines))

    assert len(checks) == 1
    assert checks[0].speaker == "A"
    assert checks[0].origin == "rule"
    assert checks[0].confidence_pct == 85
    assert checks[0].reference == "sys-ana"
    assert checks[0].correct is True


def test_operator_confusion_counts_microphone_speech_on_the_wrong_side() -> None:
    fused = operator_confusion(gold().turns, read_hypothesis(detail(), mode="fusion"))
    assert fused.rate == 0.0
    assert fused.microphone_s == pytest.approx(4.9)

    # Without attribution the echo is the operator's, for 2.9 of 4.9 seconds.
    plain = operator_confusion(gold().turns, read_hypothesis(unattributed(), mode="off"))
    assert plain.remote_as_operator_s == pytest.approx(2.9)
    assert plain.rate == pytest.approx(2.9 / 4.9)

    nobody = operator_confusion([Turn("ana", 0.0, 9.0)], read_hypothesis(detail(), mode="fusion"))
    assert nobody.rate is None


def test_echo_rate_counts_the_microphone_words_that_the_call_has() -> None:
    lines = read_hypothesis(detail(), mode="fusion").lines

    exact = echo_rate(lines, tolerance_s=0.0)
    assert exact.microphone_words == 9
    assert exact.rate == pytest.approx(5 / 9)
    assert exact.marked_rate == pytest.approx(5 / 9)

    # One second either way also takes the "a" of "bom dia a todos", which the
    # call says a moment later: the rate is an upper bound.
    assert echo_rate(lines).rate == pytest.approx(6 / 9)
    assert echo_rate([]).rate is None


# The score.


def test_score_gives_each_measure_that_the_files_can_give() -> None:
    score = score_meeting(
        gold(), read_hypothesis(detail(), mode="fusion"), mode="fusion", external=False
    )

    assert score.der is None
    assert score.wder == 0.0
    assert score.wer == 0.0
    assert score.name_recall == 0.5
    assert score.name_precision == 1.0
    assert score.echo_marked_rate == pytest.approx(5 / 9)
    assert score.reference_speakers == 3
    assert score.hypothesis_speakers == 3
    assert score.loopback_der is None
    assert score.operator_confusion == 0.0
    assert score.name_threshold_pct == 75
    assert [check.name for check in score.names] == ["Ana"]
    assert "DER, JER and loopback DER: not attempted" in score.not_measured
    assert "cpWER and tcpWER: not attempted" in score.not_measured


def test_score_says_what_the_gold_set_lacks() -> None:
    score = score_meeting(
        gold(words=[], names={}),
        read_hypothesis(detail(), mode="fusion"),
        mode="fusion",
        external=False,
    )

    assert score.wder is None
    assert score.name_recall is None
    assert any("no words" in reason for reason in score.not_measured)
    assert any("names nobody" in reason for reason in score.not_measured)


def test_score_is_written_as_json_and_shown_as_lines(tmp_path: Path) -> None:
    score = score_meeting(
        gold(), read_hypothesis(detail(), mode="fusion"), mode="fusion", external=False
    )
    target = tmp_path / "runs" / "m1.score.json"
    write_score(score, target)

    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["wder"] == 0.0
    assert written["mapping"] == {"A": "sys-ana", "B": "sys-pedro", "me": "mic-op"}
    assert written["names"][0]["correct"] is True
    shown = "\n".join(render_score(score))
    assert "WDER 0.00 %" in shown
    assert "DER not measured" in shown
    assert "operator confusion 0.00 %" in shown
    assert "names: recall 50.00 %" in shown
    assert "name A: Ana (rule, 85 %, threshold 75 %): right" in shown


# The two libraries, when they are there.


def test_diarization_rates_need_pyannote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyannote.core", None)
    assert diarization_rates(gold().turns, [], [(0.0, 7.0)], "m1") is None


def test_permutation_rates_need_meeteval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "meeteval.wer", None)
    assert permutation_rates(gold().words, [], "m1") is None


def test_der_is_zero_for_the_gold_turns() -> None:
    pytest.importorskip("pyannote.metrics")
    rates = diarization_rates(
        gold().turns, read_hypothesis(detail(), mode="fusion").lines, [(0.0, 7.0)], "m1"
    )

    assert rates is not None
    assert rates.der == pytest.approx(0.0)
    assert rates.jer == pytest.approx(0.0)


def test_cpwer_is_zero_for_the_gold_words() -> None:
    pytest.importorskip("meeteval")
    rates = permutation_rates(gold().words, read_hypothesis(detail(), mode="fusion").lines, "m1")

    assert rates is not None
    assert rates.cpwer == pytest.approx(0.0)
    assert rates.tcpwer == pytest.approx(0.0)


# The command line.


def test_command_scores_a_meeting_and_writes_the_score(tmp_path: Path) -> None:
    index = write_gold_set(tmp_path / "data" / "private" / "speakers")
    hypothesis = tmp_path / "m1.json"
    hypothesis.write_text(json.dumps(detail()), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "speakers",
            "score",
            "--root",
            str(tmp_path),
            "--gold",
            str(index),
            "--hypothesis",
            str(hypothesis),
            "--mode",
            "fusion",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "WDER" in result.output
    written = tmp_path / "runs" / "speakers" / "fusion" / "m1.score.json"
    assert json.loads(written.read_text(encoding="utf-8"))["meeting"] == "m1"


def test_command_writes_where_it_is_told_and_takes_the_meeting_by_name(tmp_path: Path) -> None:
    index = write_gold_set(tmp_path / "gold")
    hypothesis = tmp_path / "meeting-12.json"
    hypothesis.write_text(json.dumps(detail()), encoding="utf-8")
    out = tmp_path / "score.json"

    result = runner.invoke(
        app,
        [
            "speakers",
            "score",
            "--gold",
            str(index),
            "--hypothesis",
            str(hypothesis),
            "--meeting",
            "m1",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text(encoding="utf-8"))["mode"] == "native"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [(["--mode", "loud"], "unknown mode"), (["--meeting", "m9"], "not in")],
)
def test_command_stops_with_a_message(tmp_path: Path, arguments: list[str], message: str) -> None:
    index = write_gold_set(tmp_path / "gold")
    hypothesis = tmp_path / "m1.json"
    hypothesis.write_text(json.dumps(detail()), encoding="utf-8")

    result = runner.invoke(
        app,
        ["speakers", "score", "--gold", str(index), "--hypothesis", str(hypothesis), *arguments],
    )

    assert result.exit_code == EXIT_ERROR
    assert message in result.output
