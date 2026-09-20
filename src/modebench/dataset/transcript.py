"""Writes the user message of an Analyze request, as the application writes it.

The application sends the meeting in two stretches. The first stretch is
context that an earlier answer covered. The second stretch is what the answer
is about. The headings below are those of the application, character for
character, so the benchmark measures the request that production sends.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from modebench.config import TranscriptConfig
from modebench.dataset.schema import FillerFile
from modebench.dataset.variants import RenderLine
from modebench.hashing import stable_seed

# The heading of the application has a long dash. It is written as an escape
# so that this file has none.
EARLIER_HEADING = (
    "=== EARLIER IN THE MEETING (context only \u2014 already answered, do not answer again) ==="
)
NEW_HEADING = "=== NEW SINCE THE LAST ANSWER (answer only this) ==="
NOTHING_HERE = "(nothing)"


@dataclass(frozen=True, slots=True)
class Labels:
    """The channel names and the mark of a line that is not settled."""

    mic: str
    system: str
    partial_marker: str

    @classmethod
    def from_config(cls, config: TranscriptConfig) -> Self:
        """Return the labels of the bench file."""
        return cls(
            mic=config.mic_label,
            system=config.system_label,
            partial_marker=config.partial_marker,
        )


def render_lines(lines: Sequence[RenderLine], labels: Labels) -> str:
    """Return the lines as `CHANNEL: text`, one for each line."""
    rendered: list[str] = []
    for line in lines:
        channel = labels.mic if line.speaker == "mic" else labels.system
        mark = f" {labels.partial_marker}" if line.partial else ""
        rendered.append(f"{channel}: {line.text}{mark}")
    return "\n".join(rendered)


def render_window(prior: Sequence[RenderLine], fresh: Sequence[RenderLine], labels: Labels) -> str:
    """Return the two stretches below their headings. An empty stretch reads `(nothing)`."""
    earlier_text = render_lines(prior, labels) if prior else NOTHING_HERE
    fresh_text = render_lines(fresh, labels) if fresh else NOTHING_HERE
    return f"{EARLIER_HEADING}\n{earlier_text}\n\n{NEW_HEADING}\n{fresh_text}"


def nonce_line(nonce: str) -> RenderLine:
    """Return the first line of a transcript.

    The nonce makes the start of each transcript different, so a provider
    cannot serve a request from a prompt cache that an earlier request filled.
    """
    return RenderLine(speaker="mic", text=f"Registro da sessão {nonce}.")


def count_words(lines: Sequence[RenderLine]) -> int:
    """Return the number of words in the lines."""
    return sum(len(line.text.split()) for line in lines)


def build_filler(
    filler: FillerFile, minutes: int, words_per_minute: int, seed: int
) -> tuple[RenderLine, ...]:
    """Return the meeting talk of the first `minutes` minutes.

    The blocks of the filler file come in a seeded order, again and again,
    until the text has the words that a meeting of that length has. The order
    does not depend on `minutes`, so the filler of a short meeting is the
    start of the filler of a long one.
    """
    target = minutes * words_per_minute
    lines: list[RenderLine] = []
    words = 0
    cycle = 0
    while words < target:
        words_before = words
        order = list(range(len(filler.blocks)))
        random.Random(stable_seed(seed, "filler", cycle)).shuffle(order)
        for block_index in order:
            for line in filler.blocks[block_index].lines:
                if words >= target:
                    return tuple(lines)
                lines.append(RenderLine(speaker=line.speaker, text=line.text))
                words += len(line.text.split())
        if words == words_before:
            break
        cycle += 1
    return tuple(lines)
