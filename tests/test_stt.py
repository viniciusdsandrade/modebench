"""Tests for STT Portuguese normalization, WER/CER metrics, dialects (AssemblyAI, ElevenLabs), and fake STT server."""

import json
from pathlib import Path

import pytest

from modebench.stt.assemblyai import AssemblyAiAdapter, AssemblyAiParser
from modebench.stt.base import CLOSED, ERROR, FINAL, OPEN, PARTIAL, SttEvent
from modebench.stt.config import AudioItem, ReferenceUtterance, SttBilling
from modebench.stt.elevenlabs import ElevenLabsAdapter, ElevenLabsParser
from modebench.stt.fake import FakeSttAdapter
from modebench.stt.metrics import collect_finals, session_cost
from modebench.stt.normalize import normalize_pt
from modebench.stt.replay import ReceivedEvent, SessionTrace
from modebench.stt.wer import error_rates


def test_normalize_pt_lowercases_and_strips_punctuation() -> None:
    text = "Olá! Vamos fechar a meta de R$ 100 mil, né?"
    normalized = normalize_pt(text)
    assert "!" not in normalized
    assert "," not in normalized
    assert "?" not in normalized
    assert normalized == "ola vamos fechar a meta de r 100 mil ne"


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
    assert "audio_event" in parsed_audio
    assert parsed_audio["audio_event"]["audio_base_64"] == "AQIDBA=="

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
