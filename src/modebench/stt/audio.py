"""Reads WAV files and cuts them into the chunks that a live capture sends."""

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from modebench.errors import DatasetError

BYTES_PER_SAMPLE = 2


@dataclass(frozen=True, slots=True)
class PcmAudio:
    """Mono audio, 16 bit, little endian."""

    sample_rate: int
    samples: bytes

    @property
    def duration_ms(self) -> float:
        """Return the length of the audio in milliseconds."""
        return len(self.samples) / BYTES_PER_SAMPLE / self.sample_rate * 1000.0


def read_wav(path: Path) -> PcmAudio:
    """Read a 16 bit PCM WAV file. More than one channel is mixed down to one."""
    try:
        with wave.open(str(path), "rb") as handle:
            width = handle.getsampwidth()
            channels = handle.getnchannels()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except FileNotFoundError as exc:
        raise DatasetError(f"audio file not found: {path}") from exc
    except wave.Error as exc:
        raise DatasetError(f"{path} is not a WAV file that can be read: {exc}") from exc
    if width != BYTES_PER_SAMPLE:
        raise DatasetError(f"{path} must be 16 bit PCM, and it has {width * 8} bit samples")
    if channels > 1:
        matrix = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
        frames = matrix.mean(axis=1).astype("<i2").tobytes()
    return PcmAudio(sample_rate=rate, samples=frames)


def silence(sample_rate: int, duration_ms: int) -> bytes:
    """Return silence of the given length."""
    return bytes(sample_rate * duration_ms // 1000 * BYTES_PER_SAMPLE)


def chunk_bytes(samples: bytes, sample_rate: int, chunk_ms: int) -> list[bytes]:
    """Cut audio into chunks of `chunk_ms`. The last chunk can be shorter."""
    size = sample_rate * chunk_ms // 1000 * BYTES_PER_SAMPLE
    if size <= 0:
        raise ValueError("the chunk is shorter than one sample")
    return [samples[start : start + size] for start in range(0, len(samples), size)]
