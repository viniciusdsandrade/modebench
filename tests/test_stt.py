"""Tests for STT Portuguese normalization, WER/CER metrics, dialects (AssemblyAI, ElevenLabs), and fake STT server."""

import json
from pathlib import Path

import pytest

from modebench.stt.assemblyai import AssemblyAiAdapter, AssemblyAiParser
from modebench.stt.base import CLOSED, ERROR, FINAL, OPEN, PARTIAL, SttEvent
from modebench.stt.config import (
    AudioItem,
    ReferenceUtterance,
    SttBilling,
    SttNormalization,
)
from modebench.stt.elevenlabs import ElevenLabsAdapter, ElevenLabsParser
from modebench.stt.fake import FakeSttAdapter
from modebench.stt.metrics import collect_finals, session_cost, session_metrics
from modebench.stt.normalize import normalize_pt, number_to_words_pt, ordinal_to_words_pt
from modebench.stt.replay import ReceivedEvent, SessionTrace
from modebench.stt.speakers import FinalUtterance, match_reference, speaker_accuracy
from modebench.stt.wer import error_rates


def test_normalize_pt_lowercases_and_strips_punctuation() -> None:
    text = "Olá! Vamos fechar a meta de R$ 100 mil, né?"
    normalized = normalize_pt(text)
    assert "!" not in normalized
    assert "," not in normalized
    assert "?" not in normalized
    assert normalized == "olá vamos fechar a meta de cem mil reais não é"


def test_wer_and_cer_computations() -> None:
    # Identical
    rates_exact = error_rates("o prazo e amanha", "o prazo e amanha")
    assert rates_exact is not None
    assert rates_exact.wer == 0.0
    assert rates_exact.cer == 0.0
    assert rates_exact.reference_words == 4

    # One substitution ("hoje" instead of "amanha")
    rates_sub = error_rates("o prazo e amanha", "o prazo e hoje")
    assert rates_sub is not None
    assert rates_sub.wer == 0.25
    assert rates_sub.substitutions == 1

    # Empty hypothesis (all deleted)
    rates_empty = error_rates("o prazo e amanha", "")
    assert rates_empty is not None
    assert rates_empty.wer == 1.0
    assert rates_empty.deletions == 4

    # Empty reference
    assert error_rates("", "qualquer texto") is None


def test_session_cost_distinguishes_audio_vs_session_seconds() -> None:
    # $1.50 per hour
    billing_audio = SttBilling(usd_per_hour=1.50, basis="audio_seconds")
    billing_session = SttBilling(usd_per_hour=1.50, basis="session_seconds")

    # 1 hour audio (3,600,000 ms), 2 hours session (7,200,000 ms)
    audio_ms = 3_600_000.0
    session_ms = 7_200_000.0

    assert session_cost(billing_audio, audio_ms, session_ms) == pytest.approx(1.50)
    assert session_cost(billing_session, audio_ms, session_ms) == pytest.approx(3.00)


def test_fake_stt_adapter_and_parser() -> None:
    adapter = FakeSttAdapter(name="fake-test")
    assert adapter.connect_url(16000) == "fake://stt?sample_rate=16000"
    assert adapter.headers("secret") == {}
    assert adapter.audio_message(b"\x00\x01", 16000) == b"\x00\x01"

    parser = adapter.new_parser()
    events = parser.parse(
        '{"type":"final","text":"palavra teste","start_ms":100,"end_ms":500,"speaker":"spk1"}'
    )
    assert len(events) == 1
    ev = events[0]
    assert ev.kind == FINAL
    assert ev.text == "palavra teste"
    assert ev.audio_start_ms == 100.0
    assert ev.audio_end_ms == 500.0
    assert ev.speaker == "spk1"


def test_assemblyai_adapter_and_parser() -> None:
    adapter = AssemblyAiAdapter(query={"speech_model": "nano", "speakers": True})
    url = adapter.connect_url(16000)
    assert "sample_rate=16000" in url
    assert "speech_model=nano" in url
    assert "encoding=pcm_s16le" in url

    headers = adapter.headers("aai-secret-key")
    assert headers["Authorization"] == "aai-secret-key"
    assert not headers["Authorization"].startswith("Bearer")

    parser = AssemblyAiParser()
    # 1. Begin event
    assert parser.parse('{"type":"Begin"}')[0].kind == OPEN

    # 2. Interim turn (partial)
    partial_msg = json.dumps(
        {
            "type": "Turn",
            "transcript": "Bom dia",
            "end_of_turn": False,
            "turn_order": 1,
            "speaker_label": "A",
        }
    )
    events = parser.parse(partial_msg)
    assert len(events) == 1
    assert events[0].kind == PARTIAL
    assert events[0].text == "Bom dia"
    assert events[0].speaker == "A"

    # 3. Final turn
    final_msg = json.dumps(
        {
            "type": "Turn",
            "transcript": "Bom dia a todos.",
            "end_of_turn": True,
            "turn_order": 1,
            "speaker_label": "A",
            "words": [
                {"text": "Bom", "start": 100, "end": 200},
                {"text": "todos.", "start": 300, "end": 500},
            ],
        }
    )
    events = parser.parse(final_msg)
    assert events[0].kind == FINAL
    assert events[0].audio_start_ms == 100.0
    assert events[0].audio_end_ms == 500.0

    # 4. Termination
    assert parser.parse('{"type":"Termination"}')[0].kind == CLOSED

    # 5. Error
    err = parser.parse('{"type":"Error","error_code":"4001","error":"Invalid token"}')
    assert err[0].kind == ERROR
    assert "Invalid token" in (err[0].message or "")


def test_elevenlabs_adapter_and_parser() -> None:
    adapter = ElevenLabsAdapter(query={"model_id": "scribe_v1"})
    headers = adapter.headers("xi-secret-key")
    assert headers["xi-api-key"] == "xi-secret-key"

    audio_msg = adapter.audio_message(b"\x01\x02\x03\x04", 16000)
    parsed_audio = json.loads(audio_msg)
    assert parsed_audio["message_type"] == "input_audio_chunk"
    assert parsed_audio["audio_base_64"] == "AQIDBA=="

    parser = ElevenLabsParser()

    # Partial transcript
    part_events = parser.parse(
        json.dumps({"message_type": "partial_transcript", "text": "Boa tarde"})
    )
    assert len(part_events) == 1
    assert part_events[0].kind == PARTIAL
    assert part_events[0].text == "Boa tarde"

    # Committed transcript with timestamps in seconds
    commit_msg = json.dumps(
        {
            "message_type": "committed_transcript_with_timestamps",
            "words": [
                {"text": "Boa", "start": 0.5, "end": 0.8, "speaker_id": "spk_1"},
                {"text": "tarde", "start": 0.9, "end": 1.2, "speaker_id": "spk_1"},
            ],
        }
    )
    final_events = parser.parse(commit_msg)
    assert len(final_events) == 1
    fe = final_events[0]
    assert fe.kind == FINAL
    assert fe.text == "Boa tarde"
    assert fe.audio_start_ms == 500.0
    assert fe.audio_end_ms == 1200.0
    assert fe.speaker == "spk_1"


def test_collect_finals_calculates_latency_against_reference() -> None:
    audio = AudioItem(
        id="aud-1",
        wav=Path("test.wav"),
        utterances=[
            ReferenceUtterance(start_ms=200, end_ms=800, text="Olá mundo", speaker="A"),
        ],
    )

    trace = SessionTrace(
        provider="fake",
        audio_ms=1000.0,
        session_ms=1200.0,
        events=[
            ReceivedEvent(
                at_ms=1100.0,
                event=SttEvent(
                    kind=FINAL,
                    text="Olá mundo",
                    utterance=0,
                    audio_start_ms=200.0,
                    audio_end_ms=800.0,
                    speaker="A",
                ),
            )
        ],
    )

    finals = collect_finals(trace, audio)
    assert len(finals) == 1
    f = finals[0]
    assert f.text == "Olá mundo"
    assert f.received_ms == 1100.0
    assert f.audio_end_ms == 800.0
    assert f.latency_ms == 300.0  # 1100 - 800


def commit(kind: str, text: str, **fields: object) -> str:
    return json.dumps({"message_type": kind, "text": text, **fields})


def test_elevenlabs_commits_of_one_segment_come_in_any_order() -> None:
    timed = "committed_transcript_with_timestamps"
    plain = "committed_transcript"
    words = [{"text": "oi", "start": 1.0, "end": 1.5}]

    usual = ElevenLabsParser()
    events = [
        usual.parse(commit("partial_transcript", "bom"))[0],
        usual.parse(commit(plain, "bom dia"))[0],
        usual.parse(commit(timed, "bom dia", words=words))[0],
        usual.parse(commit("partial_transcript", "qual"))[0],
        usual.parse(commit(plain, "qual o prazo"))[0],
    ]
    assert [event.utterance for event in events] == [0, 0, 0, 1, 1]

    # The twin with timestamps comes first: the plain commit is still the same segment.
    swapped = ElevenLabsParser()
    events = [
        swapped.parse(commit(timed, "bom dia", words=words))[0],
        swapped.parse(commit(plain, "bom dia"))[0],
        swapped.parse(commit(timed, "qual o prazo", words=words))[0],
        swapped.parse(commit(plain, "qual o prazo"))[0],
    ]
    assert [event.utterance for event in events] == [0, 0, 1, 1]
    assert all(event.kind == FINAL for event in events)

    # Two commits of one type, with no text between them, are two segments.
    repeated = ElevenLabsParser()
    events = [
        repeated.parse(commit(plain, "um"))[0],
        repeated.parse(commit(plain, "dois"))[0],
    ]
    assert [event.utterance for event in events] == [0, 1]


def test_elevenlabs_other_messages_and_the_connect_url() -> None:
    parser = ElevenLabsParser()
    assert parser.parse('{"message_type":"session_started"}')[0].kind == OPEN
    error = parser.parse('{"message_type":"quota_exceeded","error":"no credit"}')[0]
    assert error.kind == ERROR and error.message == "quota_exceeded: no credit"
    assert parser.parse('{"message_type":"ping"}')[0].kind == "other"
    assert parser.parse(b"not json")[0].message == "not a JSON object"
    final = parser.parse(commit("final_transcript", "", words="not a list"))[0]
    assert final.kind == FINAL and final.text == "" and final.audio_end_ms is None

    adapter = ElevenLabsAdapter({"model_id": "scribe", "keyterms": ["a b", "c"], "vad": True})
    url = adapter.connect_url(16000)
    assert url.startswith("wss://api.elevenlabs.io/")
    assert "audio_format=pcm_16000" in url
    assert "keyterms=a%20b&keyterms=c" in url
    assert "vad=true" in url
    assert adapter.closing_messages() == []
    assert isinstance(adapter.new_parser(), ElevenLabsParser)


def test_assemblyai_other_messages_and_the_connect_url() -> None:
    parser = AssemblyAiParser()
    assert parser.parse("[1]")[0].message == "not a JSON object"
    assert parser.parse('{"type":"Pong"}')[0].message == "Pong"
    words = [
        {"text": "bom", "start": 10, "end": 20},
        "junk",
        {"text": "dia", "start": 30, "end": 40},
    ]
    turn = parser.parse(json.dumps({"type": "Turn", "transcript": "", "words": words}))[0]
    assert turn.kind == PARTIAL and turn.text == "bom dia" and turn.utterance is None

    adapter = AssemblyAiAdapter(
        {"language_codes": ["pt", "en"], "speaker_labels": True}, "wss://x/ws"
    )
    url = adapter.connect_url(8000)
    assert url.startswith("wss://x/ws?encoding=pcm_s16le&sample_rate=8000")
    assert "language_codes=%5B%22pt%22%2C%22en%22%5D" in url
    assert "speaker_labels=true" in url
    assert adapter.audio_message(b"\x00", 8000) == b"\x00"
    assert adapter.closing_messages() == ['{"type":"Terminate"}']
    assert isinstance(adapter.new_parser(), AssemblyAiParser)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("R$ 100 mil", "cem mil reais"),
        ("R$ 2,5 milhões", "dois vírgula cinco milhões de reais"),
        ("R$ 1 milhão", "um milhão de reais"),
        ("R$ 1.250,50", "mil duzentos e cinquenta reais e cinquenta centavos"),
        ("R$ 40", "quarenta reais"),
        ("23º andar", "vigésimo terceiro andar"),
        ("a 3ª reunião", "a terceira reunião"),
        ("10º lugar", "décimo lugar"),
        ("21ª vez", "vigésima primeira vez"),
        ("150º item", "cento e cinquenta item"),
        ("30° de febre", "trinta de febre"),
        ("2° turno", "segundo turno"),
        ("15%", "quinze por cento"),
        ("3,14", "três vírgula quatorze"),
        ("0,05", "zero vírgula zero cinco"),
        ("1.000.000", "um milhão"),
        ("2.000.500", "dois milhões e quinhentos"),
        ("1100", "mil e cem"),
        ("1234", "mil duzentos e trinta e quatro"),
        ("100", "cem"),
        ("101", "cento e um"),
        ("0", "zero"),
        ("pra vc tb, hum, tá?", "para você também está"),
        ("bem-vindo", "bem vindo"),
    ],
)
def test_normalize_pt_writes_numbers_and_speech_forms_as_words(text: str, expected: str) -> None:
    assert normalize_pt(text) == expected


def test_normalize_pt_options_and_large_numbers() -> None:
    raw = SttNormalization(expand_numbers=False, expand_contractions=False, drop_hesitations=False)
    assert normalize_pt("Pra 23, hum", raw) == "pra 23 hum"
    assert number_to_words_pt(-7) == "menos sete"
    assert (
        number_to_words_pt(10**12)
        == "um zero zero zero zero zero zero zero zero zero zero zero zero"
    )
    assert number_to_words_pt(3_000_000_000) == "três bilhões"
    assert ordinal_to_words_pt(0) is None
    assert ordinal_to_words_pt(100) is None


def utterance(
    index: int, text: str, speaker: str | None, start: float, end: float
) -> FinalUtterance:
    return FinalUtterance(index, text, speaker, end + 500.0, start, end, 500.0)


REFERENCES = [
    ReferenceUtterance(start_ms=0, end_ms=2000, speaker="Ana", text="bom dia a todos"),
    ReferenceUtterance(start_ms=3000, end_ms=5000, speaker="Rui", text="qual o prazo"),
    ReferenceUtterance(start_ms=6000, end_ms=8000, speaker="Ana", text="seis semanas"),
]


def test_speaker_accuracy_matches_the_labels_of_the_provider_to_the_reference() -> None:
    perfect = [
        utterance(0, "bom dia a todos", "spk_1", 0, 2000),
        utterance(1, "qual o prazo", "spk_0", 3000, 5000),
        utterance(2, "seis semanas", "spk_1", 6000, 8000),
    ]
    assert speaker_accuracy(perfect, REFERENCES) == 1.0
    # The last utterance has the label of the other speaker: 2 words of 9 are wrong.
    confused = [*perfect[:2], utterance(2, "seis semanas", "spk_0", 6000, 8000)]
    assert speaker_accuracy(confused, REFERENCES) == pytest.approx(7 / 9)
    # With no label, or with no speaker in the reference, there is no score.
    unlabelled = [utterance(0, "bom dia", None, 0, 2000)]
    assert speaker_accuracy(unlabelled, REFERENCES) is None
    anonymous = [ReferenceUtterance(start_ms=0, end_ms=2000, text="bom dia")]
    assert speaker_accuracy(perfect, anonymous) is None


def test_speaker_matching_falls_back_to_the_position_and_to_a_greedy_match() -> None:
    no_times = FinalUtterance(1, "qual o prazo", "x", 5500.0, None, None, None)
    assert match_reference(no_times, 1, REFERENCES).speaker == "Rui"
    assert match_reference(no_times, 9, REFERENCES).speaker == "Ana"
    only_end = FinalUtterance(0, "prazo", "x", 5500.0, None, 4000.0, 500.0)
    assert match_reference(only_end, 0, REFERENCES).speaker == "Ana"

    # More labels than the exact search takes: each label goes to its own speaker.
    many = [
        ReferenceUtterance(start_ms=i * 1000, end_ms=i * 1000 + 900, speaker=f"s{i}", text="oi")
        for i in range(8)
    ]
    finals = [utterance(i, "oi", f"label{i}", i * 1000, i * 1000 + 900) for i in range(8)]
    assert speaker_accuracy(finals, many) == 1.0


def test_collect_finals_keeps_the_first_time_and_the_last_text() -> None:
    audio = AudioItem(id="a", wav=Path("a.wav"), utterances=REFERENCES)
    events = [
        ReceivedEvent(900.0, SttEvent(kind=PARTIAL, text="bom", utterance=0)),
        ReceivedEvent(2400.0, SttEvent(kind=FINAL, text="bom dia", utterance=0)),
        ReceivedEvent(2600.0, SttEvent(FINAL, "bom dia a todos", 0, 1000.0, 3000.0, "A")),
        ReceivedEvent(5600.0, SttEvent(kind=FINAL, text="qual o prazo")),
        ReceivedEvent(5700.0, SttEvent(kind=FINAL, text="  ", utterance=7)),
    ]
    trace = SessionTrace(provider="p", audio_ms=9000.0, preroll_ms=1000.0, events=events)
    finals = collect_finals(trace, audio)
    assert [final.text for final in finals] == ["bom dia a todos", "qual o prazo"]
    # The word times hold the silence before the audio, and it is taken out: 3000 - 1000.
    assert finals[0].audio_end_ms == 2000.0
    assert finals[0].latency_ms == pytest.approx(400.0)
    assert finals[0].speaker == "A"
    # With no word time, the end of the reference utterance at the same position is used.
    assert finals[1].audio_end_ms == 5000.0
    billing = SttBilling(basis="audio_seconds", usd_per_hour=1.0)
    metrics = session_metrics(trace, audio, billing, SttNormalization())
    assert metrics.first_partial_ms == 900.0
    assert metrics.first_partial_after_speech_ms == 900.0
    assert metrics.hypothesis == "bom dia a todos qual o prazo"
