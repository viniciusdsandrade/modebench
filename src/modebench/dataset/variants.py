"""Makes the variants of each case.

A click on Analyze rarely comes after a complete, clean question. The
variants put each case in the conditions of real use:

- truncation: the click comes when only a part of the question is said;
- ASR noise: the recogniser gets some words wrong, at a word error rate
  that the profile gives;
- noise: there is no question at all, and the correct answer is a refusal.

Each function is deterministic. A seed and a case always give the same text.
"""

import random
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload

from modebench.config import NoiseKind, Profile
from modebench.dataset.loader import LoadedDataset
from modebench.dataset.schema import Case, DatasetFile, Line
from modebench.hashing import stable_seed

TRAILING_PUNCTUATION = ".,;:!?"

# Words that a Portuguese recogniser confuses, because they sound the same.
_CONFUSIONS: dict[str, str] = {
    "mas": "mais",
    "mais": "mas",
    "sessão": "seção",
    "seção": "sessão",
    "cem": "sem",
    "sem": "cem",
    "tem": "têm",
    "há": "a",
    "e": "é",
    "é": "e",
    "está": "esta",
    "esta": "está",
    "traz": "trás",
    "nós": "nos",
    "para": "pra",
    "você": "cê",
    "estou": "tô",
    "não": "num",
    "vai": "vão",
    "foi": "fui",
    "ser": "cer",
    "prazo": "prato",
    "custo": "gusto",
    "cliente": "clientes",
    "reunião": "união",
}

_INSERTIONS: tuple[str, ...] = ("é", "né", "tipo", "assim", "então", "hum")

_STRAY_PAIRS: tuple[str, ...] = (
    "mesa amanhã",
    "verde relatório",
    "janela quinze",
    "café planilha",
    "porta ontem",
    "azul servidor",
)

_HESITATIONS: tuple[str, ...] = (
    "hum... é... ahn",
    "ahn... hum",
    "é... hã... hum...",
    "hmm... eh... ahn...",
)


@dataclass(frozen=True, slots=True)
class RenderLine:
    """One transcript line that is ready to be written. `partial` marks speech not settled."""

    speaker: str
    text: str
    partial: bool = False


@dataclass(frozen=True, slots=True)
class Variant:
    """One case in one condition."""

    variant_id: str
    case_id: str
    kind: str
    truncation_pct: int
    asr_wer: float
    noise_kind: str | None
    expect_refusal: bool
    private: bool
    earlier: tuple[RenderLine, ...]
    fresh: tuple[RenderLine, ...]
    gold_question: str
    key_points: tuple[str, ...]
    reference_answer: str

    @property
    def is_baseline(self) -> bool:
        """Return True for the complete, clean form of a case."""
        return self.kind == "truncation" and self.truncation_pct == 100


def strip_accents(text: str) -> str:
    """Return the text without its diacritics."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def truncate_words(text: str, pct: int) -> str:
    """Return the first `pct` percent of the words, and at least one word.

    Speech that stops in the middle has no closing punctuation, so the
    punctuation at the cut is removed.
    """
    words = text.split()
    if pct >= 100 or len(words) <= 1:
        return text
    keep = max(1, (len(words) * pct + 50) // 100)
    if keep >= len(words):
        return text
    return " ".join(words[:keep]).rstrip(TRAILING_PUNCTUATION)


def _corrupt(word: str, rng: random.Random) -> str:
    """Return a different word that a recogniser could write for `word`."""
    lowered = word.strip(TRAILING_PUNCTUATION).lower()
    if lowered in _CONFUSIONS:
        return _CONFUSIONS[lowered]
    candidates: list[str] = []
    plain = strip_accents(lowered)
    if plain != lowered:
        candidates.append(plain)
    if len(lowered) > 3:
        candidates.append(lowered[:-1])
    if len(lowered) > 4:
        candidates.append(lowered[1:])
    candidates.append(lowered[:-1] if lowered.endswith("s") and len(lowered) > 1 else lowered + "s")
    return rng.choice(candidates)


def _apply_asr_noise_lines(
    lines: Sequence[RenderLine], wer: float, rng: random.Random
) -> tuple[RenderLine, ...]:
    tokens: list[tuple[int, str]] = [
        (index, word) for index, line in enumerate(lines) for word in line.text.split()
    ]
    total = len(tokens)
    errors = min(total, int(wer * total + 0.5))
    positions = rng.sample(range(total), errors)
    substitutions = int(errors * 0.6 + 0.5)
    deletions = int(errors * 0.25 + 0.5)
    operations: dict[int, str] = {}
    for rank, position in enumerate(positions):
        if rank < substitutions:
            operations[position] = "substitute"
        elif rank < substitutions + deletions:
            operations[position] = "delete"
        else:
            operations[position] = "insert"
    rebuilt: list[list[str]] = [[] for _ in lines]
    for position, (index, word) in enumerate(tokens):
        operation = operations.get(position)
        if operation == "delete":
            continue
        if operation == "substitute":
            rebuilt[index].append(_corrupt(word, rng))
            continue
        rebuilt[index].append(word)
        if operation == "insert":
            rebuilt[index].append(rng.choice(_INSERTIONS))
    return tuple(
        RenderLine(speaker=line.speaker, text=" ".join(words), partial=line.partial)
        for line, words in zip(lines, rebuilt, strict=True)
        if words
    )


@overload
def apply_asr_noise(
    lines: str,
    wer: float,
    rng: random.Random | None = None,
    *,
    seed: int | None = None,
) -> str: ...


@overload
def apply_asr_noise(
    lines: Sequence[RenderLine],
    wer: float,
    rng: random.Random | None = None,
    *,
    seed: int | None = None,
) -> tuple[RenderLine, ...]: ...


def apply_asr_noise(
    lines: Sequence[RenderLine] | str,
    wer: float,
    rng: random.Random | None = None,
    *,
    seed: int | None = None,
) -> tuple[RenderLine, ...] | str:
    """Return the lines with word errors at the rate `wer`.

    The number of errors is exact: `wer` times the number of words, rounded.
    Six errors in ten are substitutions, a quarter are deletions and the
    others are insertions, which is the usual mix of a streaming recogniser.
    """
    actual_rng = rng if rng is not None else random.Random(seed if seed is not None else 0)
    if isinstance(lines, str):
        input_lines = [RenderLine(speaker="mic", text=lines)]
        corrupted = _apply_asr_noise_lines(input_lines, wer, actual_rng)
        return corrupted[0].text if corrupted else ""
    return _apply_asr_noise_lines(lines, wer, actual_rng)


def _render(lines: Sequence[Line]) -> tuple[RenderLine, ...]:
    return tuple(RenderLine(speaker=line.speaker, text=line.text) for line in lines)


def _variant(
    case: Case,
    *,
    private: bool,
    kind: str,
    fresh: tuple[RenderLine, ...],
    truncation_pct: int = 100,
    asr_wer: float = 0.0,
) -> Variant:
    suffix = f"trunc{truncation_pct}" if kind == "truncation" else f"wer{asr_wer:g}"
    if case.expect_refusal:
        kind, suffix = "noise", "authored"
    return Variant(
        variant_id=f"{case.id}|{suffix}",
        case_id=case.id,
        kind=kind,
        truncation_pct=truncation_pct,
        asr_wer=asr_wer,
        noise_kind="authored" if case.expect_refusal else None,
        expect_refusal=case.expect_refusal,
        private=private,
        earlier=_render(case.earlier),
        fresh=fresh,
        gold_question=case.question.text if case.question is not None else "",
        key_points=tuple(case.key_points),
        reference_answer=case.reference_answer or "",
    )


def truncation_variant(case: Case, pct: int, *, private: bool) -> Variant:
    """Return the case with its question cut at `pct` percent of the words."""
    fresh = list(_render(case.context))
    if case.question is not None:
        text = truncate_words(case.question.text, pct)
        fresh.append(RenderLine(speaker=case.question.speaker, text=text, partial=pct < 100))
    return _variant(
        case, private=private, kind="truncation", fresh=tuple(fresh), truncation_pct=pct
    )


def asr_variant(case: Case, wer: float, seed: int, *, private: bool) -> Variant:
    """Return the complete case with synthetic recogniser errors."""
    clean = truncation_variant(case, 100, private=private)
    rng = random.Random(stable_seed(seed, "asr", case.id, wer))
    fresh = apply_asr_noise(clean.fresh, wer, rng)
    return _variant(case, private=private, kind="asr_noise", fresh=fresh, asr_wer=wer)


def noise_variant(kind: NoiseKind, instance: int, seed: int) -> Variant:
    """Return one generated noise case. The correct answer to it is a refusal."""
    rng = random.Random(stable_seed(seed, "noise", kind, instance))
    fresh: tuple[RenderLine, ...]
    if kind == "empty":
        fresh = ()
    elif kind == "two_words":
        fresh = (RenderLine(speaker="system", text=rng.choice(_STRAY_PAIRS), partial=True),)
    else:
        fresh = (RenderLine(speaker="system", text=rng.choice(_HESITATIONS), partial=True),)
    case_id = f"noise-{kind.replace('_', '-')}-{instance}"
    return Variant(
        variant_id=f"{case_id}|noise",
        case_id=case_id,
        kind="noise",
        truncation_pct=100,
        asr_wer=0.0,
        noise_kind=kind,
        expect_refusal=True,
        private=False,
        earlier=(),
        fresh=fresh,
        gold_question="",
        key_points=(),
        reference_answer="",
    )


def select_cases(datasets: Sequence[LoadedDataset], limit: int | None) -> list[tuple[Case, bool]]:
    """Return the first `limit` cases in file order, each with its privacy."""
    cases = [(case, dataset.private) for dataset in datasets for case in dataset.data.cases]
    return cases if limit is None else cases[:limit]


def build_variants(datasets: Sequence[LoadedDataset], profile: Profile, seed: int) -> list[Variant]:
    """Return each variant that the profile asks for.

    A case with no gold question is an imported session. It is used as it
    stands, because its text is already what a recogniser wrote at a click.
    """
    variants: list[Variant] = []
    for case, private in select_cases(datasets, profile.cases):
        synthetic = case.question is not None and not case.expect_refusal
        if not synthetic:
            variants.append(truncation_variant(case, 100, private=private))
            continue
        for pct in profile.truncations:
            variants.append(truncation_variant(case, pct, private=private))
        for wer in profile.asr_wers:
            variants.append(asr_variant(case, wer, seed, private=private))
    for kind in profile.noise_kinds:
        for instance in range(1, profile.noise_instances + 1):
            variants.append(noise_variant(kind, instance, seed))
    return variants


def generate_variants(
    cases: Sequence[Case], profile: Profile, seed: int, *, private: bool = False
) -> list[Variant]:
    """Return each variant of `cases` that the profile asks for."""
    dataset = LoadedDataset(
        path=Path("synthetic.yaml"),
        private=private,
        data=DatasetFile(cases=list(cases)),
    )
    return build_variants([dataset], profile, seed)
