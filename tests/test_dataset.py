"""Tests for dataset loading, variant generation (truncation, ASR noise, noise cases), meeting scenarios, and SQLite import."""

import sqlite3
from pathlib import Path

from helpers import REPO_ROOT
from modebench.config import Profile
from modebench.dataset.importer import ImportOptions, import_cases
from modebench.dataset.loader import load_dataset, load_filler
from modebench.dataset.meeting import build_scenarios
from modebench.dataset.schema import Case, FillerBlock, FillerFile, Line
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
    generate_variants,
    truncate_words,
)


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
