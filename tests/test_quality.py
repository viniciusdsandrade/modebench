"""Tests for deterministic scorers (refusal, preamble, emptiness) and the blind LLM judge protocol."""

import pytest

from modebench.quality.deterministic import (
    has_incomplete_remark,
    has_preamble,
    is_refusal,
    score_deterministic,
)
from modebench.quality.judge import (
    JudgeItem,
    JudgeVerdict,
    parse_verdict,
    render_judge_message,
    score_records,
)


def test_is_refusal_matches_noise_markers_and_checks_length() -> None:
    assert is_refusal("Não há nenhuma pergunta.")
    assert is_refusal("Apenas ruído na gravação.")
    assert is_refusal("Não contém uma pergunta nova.")
    assert is_refusal("No question identified.")

    # A real answer is not a refusal
    assert not is_refusal("O prazo de entrega foi acordado para o dia quinze.")

    # Over 40 words is not treated as a clean refusal even with markers
    long_text = "não há nenhuma pergunta " + ("palavra " * 45)
    assert not is_refusal(long_text)


def test_has_preamble_detects_conversational_fillers() -> None:
    assert has_preamble("Claro! Com base na transcrição, o prazo é terça.")
    assert has_preamble("Aqui está a resposta: o projeto começa em maio.")
    assert has_preamble("De acordo com a transcrição da reunião, fechamos o valor.")
    assert has_preamble("Sure, here is the answer: we decided on option B.")

    # Direct answer has no preamble
    assert not has_preamble("O projeto começa na segunda-feira às nove.")


def test_has_incomplete_remark_detects_complaints_about_audio() -> None:
    assert has_incomplete_remark("A transcrição parece incompleta neste ponto.")
    assert has_incomplete_remark("A fala foi cortada antes de terminar.")
    assert not has_incomplete_remark("O cliente pediu mais informações.")


def test_score_deterministic_evaluates_noise_and_answer() -> None:
    # 1. Clean answer on normal case
    clean_scores = score_deterministic("O valor total é de vinte mil reais.", expect_refusal=False)
    assert clean_scores.non_empty is True
    assert clean_scores.is_refusal is False
    assert clean_scores.refusal_only_on_noise is True  # Did not refuse when refusal wasn't expected
    assert clean_scores.no_preamble is True

    # 2. Refusal on noise case
    noise_scores = score_deterministic("Não há pergunta identificável.", expect_refusal=True)
    assert noise_scores.non_empty is True
    assert noise_scores.is_refusal is True
    assert noise_scores.refusal_only_on_noise is True  # Refused when noise was expected

    # 3. False refusal on normal case
    false_refusal = score_deterministic("Não há pergunta.", expect_refusal=False)
    assert false_refusal.is_refusal is True
    assert false_refusal.refusal_only_on_noise is False

    # 4. Empty answer
    empty_scores = score_deterministic("", expect_refusal=False)
    assert empty_scores.non_empty is False


def test_parse_verdict_validates_json_and_clamps_points() -> None:
    valid_json = """
    ```json
    {
        "refused": false,
        "inferred_question_correct": true,
        "key_points_covered": 4,
        "utility": 5,
        "hallucination": false,
        "rationale": "Respondeu com precisão e clareza."
    }
    ```
    """
    verdict = parse_verdict(valid_json, key_points=2)
    assert verdict.refused is False
    assert verdict.inferred_question_correct is True
    # Clamped to available key_points = 2
    assert verdict.key_points_covered == 2
    assert verdict.utility == 5
    assert verdict.hallucination is False


def test_parse_verdict_raises_on_invalid_or_missing_json() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        parse_verdict("Desculpe, não consegui avaliar.", key_points=1)

    with pytest.raises(ValueError, match="invalid verdict"):
        parse_verdict('{"utility": 99}', key_points=1)


def test_render_judge_message_keeps_evaluation_blind() -> None:
    item = JudgeItem(
        fresh_text="CHAMADA: quando lançamos o produto?",
        expect_refusal=False,
        gold_question="Quando lançamos o produto?",
        key_points=("no fim do mês", "versão beta"),
        reference_answer="O lançamento da versão beta será no fim do mês.",
        answer="Lançaremos a versão beta no final deste mês.",
    )

    prompt = render_judge_message(item)
    assert "<new_stretch>" in prompt
    assert "quando lançamos o produto?" in prompt
    assert "<expected>" in prompt
    assert "no fim do mês" in prompt
    assert "<candidate>" in prompt
    assert "Lançaremos a versão beta" in prompt
    # Blind: ensure prompt does not carry provider tags
    assert "openrouter" not in prompt.lower()
    assert "gemini" not in prompt.lower()


def test_score_records_transforms_verdict_into_records() -> None:
    verdict = JudgeVerdict(
        refused=False,
        inferred_question_correct=True,
        key_points_covered=2,
        utility=4,
        hallucination=False,
        rationale="Boa resposta.",
    )

    records = score_records(
        run_id="run-1",
        request_id="req-1",
        verdict=verdict,
        expect_refusal=False,
        key_points=2,
    )

    record_map = {r.metric: r.value for r in records}
    assert record_map["question_accuracy"] == 1.0
    assert record_map["utility"] == 4.0
    assert record_map["false_refusal"] == 0.0
    assert record_map["false_acceptance"] == 0.0
    assert record_map["quality"] > 0.0
