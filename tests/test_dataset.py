"""Tests for dataset loading, variant generation (truncation, ASR noise, noise cases), meeting scenarios, and SQLite import."""

import sqlite3
import stat
from pathlib import Path

import pytest

from helpers import REPO_ROOT
from modebench.config import MeetingConfig, Profile, TranscriptConfig
from modebench.dataset.importer import ImportOptions, import_cases, write_private_dataset
from modebench.dataset.loader import LoadedDataset, is_private_path, load_dataset, load_filler
from modebench.dataset.meeting import build_scenarios
from modebench.dataset.schema import Case, DatasetFile, FillerBlock, FillerFile, Line
from modebench.dataset.transcript import (
    EARLIER_HEADING,
    NEW_HEADING,
    Labels,
    nonce_line,
    render_window,
)
from modebench.dataset.variants import (
    RenderLine,
    Variant,
    apply_asr_noise,
    build_variants,
    generate_variants,
    truncate_words,
)
from modebench.errors import DatasetError
from modebench.runner.plan import build_items


def test_seed_public_dataset_loads_valid_cases_and_filler() -> None:
    cases_path = REPO_ROOT / "data" / "public" / "analyze" / "cases.yaml"
    filler_path = REPO_ROOT / "data" / "public" / "analyze" / "filler.yaml"

    cases_file = load_dataset(REPO_ROOT, cases_path).data
    filler_file = load_filler(filler_path)

    assert len(cases_file.cases) == 20
    assert len(filler_file.blocks) > 0
    assert cases_file.cases[0].question is not None
    assert cases_file.cases[0].question.text != ""
    assert len(cases_file.cases[0].key_points) > 0


def test_truncate_words_cuts_at_percentages() -> None:
    text = "um dois tres quatro cinco seis sete oito nove dez"
    # 10 words total
    assert truncate_words(text, 100) == text
    assert truncate_words(text, 50) == "um dois tres quatro cinco"
    assert truncate_words(text, 70) == "um dois tres quatro cinco seis sete"
    assert truncate_words(text, 80) == "um dois tres quatro cinco seis sete oito"
    assert truncate_words(text, 90) == "um dois tres quatro cinco seis sete oito nove"


def test_truncate_words_strips_trailing_punctuation() -> None:
    text = "Qual é o prazo, por favor?"
    # If cut before the question mark, punctuation at cut boundary is stripped
    res = truncate_words(text, 80)
    assert not res.endswith(",")
    assert not res.endswith("?")


def test_apply_asr_noise_is_deterministic_and_respects_wer() -> None:
    text = "A reunião de hoje tem o objetivo de fechar o prazo do cliente."
    clean = apply_asr_noise(text, 0.0, seed=42)
    assert clean == text

    noisy1 = apply_asr_noise(text, 0.20, seed=42)
    noisy2 = apply_asr_noise(text, 0.20, seed=42)
    assert noisy1 == noisy2
    # With 20% WER, some words should be substituted or corrupted
    assert noisy1 != text


def test_generate_variants_creates_baseline_and_noise() -> None:
    case = Case(
        id="test-01",
        context=[Line(speaker="mic", text="Qual é o valor final do contrato?")],
        question=Line(speaker="mic", text="Qual é o valor final do contrato?"),
        key_points=["cem mil reais", "parcelado em três vezes"],
        reference_answer="O valor final é cem mil reais.",
    )

    profile = Profile(
        design="star",
        durations_min=[5, 15],
        baseline_duration_min=5,
        truncations=[100, 50],
        asr_wers=[0.2],
        noise_kinds=["empty", "two_words"],
        repetitions=1,
        max_cost_usd=10.0,
    )

    variants = generate_variants([case], profile, seed=123)

    kinds = {v.kind for v in variants}
    assert "truncation" in kinds
    assert "asr_noise" in kinds
    assert "noise" in kinds
    assert any(v.is_baseline for v in variants)

    # Noise variants expect refusal
    noise_vars = [v for v in variants if v.kind == "noise"]
    assert len(noise_vars) >= 2
    assert all(v.expect_refusal for v in noise_vars)


def test_render_window_formats_transcript_with_nonces() -> None:
    nonce = nonce_line("custom-nonce-key")
    earlier = (nonce, RenderLine(speaker="system", text="Bom dia a todos."))
    fresh = (RenderLine(speaker="mic", text="Quando sai o relatório?", partial=False),)
    labels = Labels(mic="MIC", system="SYSTEM", partial_marker="*")

    rendered = render_window(earlier, fresh, labels)
    assert EARLIER_HEADING in rendered
    assert "Bom dia a todos." in rendered
    assert NEW_HEADING in rendered
    assert "Quando sai o relatório?" in rendered
    assert "Registro da sessão custom-nonce-key." in rendered


def test_meeting_scenario_chains_clicks_with_growing_history() -> None:
    filler = FillerFile(
        blocks=[
            FillerBlock(
                topic="reuniao",
                lines=[
                    Line(speaker="system", text="Discussão sobre a meta trimestral."),
                    Line(speaker="mic", text="Alinhamos os pontos pendentes."),
                ],
            )
        ]
    )

    v1 = Variant(
        variant_id="c1-b",
        case_id="c1",
        kind="truncation",
        truncation_pct=100,
        asr_wer=0.0,
        noise_kind=None,
        expect_refusal=False,
        private=False,
        earlier=(),
        fresh=(RenderLine(speaker="mic", text="Primeira pergunta?"),),
        gold_question="Pergunta 1",
        key_points=("p1",),
        reference_answer="Resp 1",
    )

    v2 = Variant(
        variant_id="c2-b",
        case_id="c2",
        kind="truncation",
        truncation_pct=100,
        asr_wer=0.0,
        noise_kind=None,
        expect_refusal=False,
        private=False,
        earlier=(),
        fresh=(RenderLine(speaker="mic", text="Segunda pergunta?"),),
        gold_question="Pergunta 2",
        key_points=("p2",),
        reference_answer="Resp 2",
    )

    scenarios = build_scenarios(
        variants=[v1, v2],
        filler=filler,
        click_minutes=[5, 15],
        scenarios=1,
        words_per_minute=120,
        seed=42,
    )

    assert len(scenarios) == 1
    clicks = scenarios[0]
    assert len(clicks) == 2

    # First click is cold cache
    assert clicks[0].cache_state == "cold"
    assert clicks[0].minute == 5

    # Second click is warm cache and includes settled speech from click 1
    assert clicks[1].cache_state == "warm"
    assert clicks[1].minute == 15
    prior_texts = [line.text for line in clicks[1].prior]
    assert any("Primeira pergunta?" in text for text in prior_texts)


def test_import_cases_from_sqlite(tmp_path: Path) -> None:
    db_file = tmp_path / "stt_bridge_test.db"
    conn = sqlite3.connect(db_file)
    conn.execute(
        """
        CREATE TABLE segments (
            id INTEGER PRIMARY KEY,
            meeting_id INTEGER,
            source TEXT,
            ts_start_ms INTEGER,
            text TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE interpretations (
            id INTEGER PRIMARY KEY,
            meeting_id INTEGER,
            task TEXT,
            up_to_ts_ms INTEGER,
            up_to_segment_id INTEGER,
            model TEXT,
            text TEXT,
            status TEXT,
            windowed INTEGER
        )
        """
    )

    conn.execute(
        "INSERT INTO segments VALUES (1, 1, 'system', 1000, 'Contexto inicial da reuniao')"
    )
    conn.execute("INSERT INTO segments VALUES (2, 1, 'mic', 2000, 'Pergunta sobre os custos')")
    conn.execute(
        """
        INSERT INTO interpretations VALUES (
            10, 1, 'recent_question', 2500, 2, 'gemini',
            'O custo estimado e dez mil reais.', 'completed', 1
        )
        """
    )
    conn.commit()
    conn.close()

    cases = import_cases(db_file, ImportOptions(task="recent_question", windowed_only=True))
    assert len(cases) == 1
    case = cases[0]
    assert case.id == "m1-i10"
    assert case.reference_answer == "O custo estimado e dez mil reais."
    assert any("Pergunta sobre os custos" in line.text for line in case.context)


def test_a_private_path_is_found_through_links_and_letter_case(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "other-disk" / "sessions"
    outside.mkdir(parents=True)
    (outside / "real.yaml").write_text("x", encoding="utf-8")
    (root / "data" / "public").mkdir(parents=True)
    # The private directory is a link to another disk.
    (root / "data" / "private").symlink_to(outside, target_is_directory=True)
    assert is_private_path(root, root / "data" / "private" / "real.yaml")
    # A link in a public directory that points into the private one.
    (root / "data" / "public" / "alias.yaml").symlink_to(root / "data" / "private" / "real.yaml")
    assert not is_private_path(root, root / "data" / "public" / "other.yaml")
    assert is_private_path(root, root / "data" / "private" / ".." / "private" / "real.yaml")
    assert is_private_path(root, root / "Data" / "Private" / "real.yaml")
    assert not is_private_path(root, tmp_path / "elsewhere" / "data" / "private" / "x.yaml")


def test_a_dataset_is_private_by_its_header_or_by_its_directory(tmp_path: Path) -> None:
    text = "cases:\n  - id: c1\n    question: {speaker: mic, text: 'Qual o prazo?'}\n"
    public = tmp_path / "data" / "public" / "cases.yaml"
    private_dir = tmp_path / "data" / "private" / "cases.yaml"
    marked = tmp_path / "data" / "public" / "marked.yaml"
    for path in (public, private_dir, marked):
        path.parent.mkdir(parents=True, exist_ok=True)
    public.write_text(text, encoding="utf-8")
    private_dir.write_text(text, encoding="utf-8")
    marked.write_text("visibility: private\n" + text, encoding="utf-8")
    assert not load_dataset(tmp_path, public).private
    assert load_dataset(tmp_path, private_dir).private
    assert load_dataset(tmp_path, marked).private

    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path, tmp_path / "absent.yaml")
    with pytest.raises(DatasetError, match="cannot be read"):
        load_dataset(tmp_path, tmp_path / "data")
    (tmp_path / "broken.yaml").write_text("cases: [unclosed", encoding="utf-8")
    with pytest.raises(DatasetError, match="not valid YAML"):
        load_dataset(tmp_path, tmp_path / "broken.yaml")
    (tmp_path / "empty.yaml").write_text("cases: []\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="not a valid dataset"):
        load_dataset(tmp_path, tmp_path / "empty.yaml")
    with pytest.raises(DatasetError, match="not a valid filler"):
        load_filler(tmp_path / "empty.yaml")


def test_equal_identifiers_in_two_datasets_are_an_error() -> None:
    case = Case(id="shared-01", question=Line(speaker="mic", text="Qual o prazo do projeto?"))
    noise_name = Case(id="noise-empty-1", question=Line(speaker="mic", text="Quem aprova?"))
    profile = Profile(
        durations_min=[5],
        baseline_duration_min=5,
        truncations=[100],
        noise_kinds=["empty"],
        max_cost_usd=1.0,
    )
    first = LoadedDataset(Path("a.yaml"), False, DatasetFile(cases=[case, noise_name]))
    second = LoadedDataset(Path("b.yaml"), True, DatasetFile(cases=[case]))
    assert len(build_variants([first], profile, seed=1)) == 3
    with pytest.raises(DatasetError, match=r"shared-01\|trunc100"):
        build_variants([first, second], profile, seed=1)
    with pytest.raises(ValueError, match="duplicate case ids"):
        DatasetFile(cases=[case, case])
    with pytest.raises(ValueError, match="no question and no context"):
        Case(id="empty-case")


def test_the_importer_refuses_tracked_directories_and_protects_the_file(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    case = Case(id="m1-i1", context=[Line(speaker="mic", text="Qual o prazo?")])
    with pytest.raises(DatasetError, match="not below data/private"):
        write_private_dataset([case], root / "data" / "public" / "leak.yaml", root=root)
    with pytest.raises(DatasetError, match="no completed answer"):
        write_private_dataset([], root / "data" / "private" / "x.yaml", root=root)

    inside = write_private_dataset([case], root / "data" / "private" / "x.yaml", root=root)
    outside = write_private_dataset([case], tmp_path / "vault" / "x.yaml", root=root)
    for written in (inside, outside):
        assert stat.S_IMODE(written.stat().st_mode) == 0o600
        assert load_dataset(root, written).private
    with pytest.raises(DatasetError, match="--force"):
        write_private_dataset([case], inside, root=root)
    inside.chmod(0o644)
    write_private_dataset([case], inside, force=True, root=root)
    assert stat.S_IMODE(inside.stat().st_mode) == 0o600


def test_the_importer_reports_a_database_that_is_not_of_the_application(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="database not found"):
        import_cases(tmp_path / "absent.db")
    no_table = tmp_path / "empty.db"
    sqlite3.connect(no_table).close()
    with pytest.raises(DatasetError, match="no interpretations table"):
        import_cases(no_table)
    half = tmp_path / "half.db"
    connection = sqlite3.connect(half)
    connection.execute("CREATE TABLE interpretations (id INTEGER, status TEXT)")
    connection.commit()
    connection.close()
    with pytest.raises(DatasetError, match="not a database of the application"):
        import_cases(half)


def test_a_click_after_a_private_case_is_private_too() -> None:
    filler = FillerFile(
        blocks=[FillerBlock(topic="geral", lines=[Line(speaker="system", text="Conversa geral.")])]
    )

    def clean_variant(case_id: str, *, private: bool) -> Variant:
        return Variant(
            variant_id=f"{case_id}|trunc100",
            case_id=case_id,
            kind="truncation",
            truncation_pct=100,
            asr_wer=0.0,
            noise_kind=None,
            expect_refusal=False,
            private=private,
            earlier=(),
            fresh=(RenderLine(speaker="mic", text=f"Pergunta de {case_id}?"),),
            gold_question="Pergunta?",
            key_points=(),
            reference_answer="",
        )

    variants = [
        clean_variant("public-1", private=False),
        clean_variant("secret-2", private=True),
        clean_variant("public-3", private=False),
    ]
    clicks = build_scenarios(variants, filler, [5, 10, 15], 1, 100, seed=1)[0]
    # The third click is a public case, and its earlier stretch holds the private question.
    assert [click.private for click in clicks] == [False, True, True]
    assert any("secret-2" in line.text for line in clicks[2].prior)

    profile = Profile(
        durations_min=[5],
        baseline_duration_min=5,
        truncations=[100],
        meeting_scenarios=1,
        max_cost_usd=1.0,
    )
    items = build_items(
        variants, profile, filler, TranscriptConfig(), MeetingConfig(click_minutes=[5, 10, 15]), 1
    )
    flags = {item.item_id: item.private for item in items}
    assert flags["public-3|trunc100|5m"] is False
    assert flags["secret-2|trunc100|5m"] is True
    assert flags["meeting-1|click2"] is True
