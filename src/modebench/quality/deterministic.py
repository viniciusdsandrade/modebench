"""Scorers that need no model: they read the answer and apply fixed rules.

The rules are those of the pre-prompt of the application: refuse only real
noise, never return nothing, and start with the question, not with a preamble.
"""

import re
from dataclasses import dataclass

from modebench.dataset.variants import strip_accents
from modebench.storage.records import ScoreRecord

SCORER = "deterministic"
REFUSAL_MAX_WORDS = 40

# Compared with the answer in lower case and without diacritics.
_REFUSAL_MARKERS: tuple[str, ...] = (
    "nao ha pergunta",
    "nao ha nenhuma pergunta",
    "nenhuma pergunta",
    "sem pergunta",
    "nao contem pergunta",
    "nao contem uma pergunta",
    "nao ha nada de novo",
    "nada de novo",
    "nada novo",
    "nao ha conteudo",
    "nao ha o que responder",
    "nada a responder",
    "nao foi possivel identificar",
    "nao consegui identificar",
    "nao identifiquei",
    "apenas ruido",
    "somente ruido",
    "so ha ruido",
    "palavras soltas",
    "no question",
    "nothing new",
    "only noise",
    "cannot identify a question",
    "there is nothing to answer",
)

# Openings of courtesy. Each one must be whole words: "ola" is a preamble and
# "Olavo" is a name, "entendi" is a preamble and "entendimento" is a noun.
_PREAMBLE_WORDS: tuple[str, ...] = (
    "claro",
    "com certeza",
    "certamente",
    "aqui esta",
    "aqui vai",
    "segue",
    "analisando",
    "entendi",
    "ola",
    "otima pergunta",
    "sure",
    "certainly",
    "of course",
    "here is",
    "here's",
    "based on",
    "according to the transcript",
    "let me",
)
# Openings whose last word has more than one ending ("com base na", "com base no").
_PREAMBLE_PREFIXES: tuple[str, ...] = (
    "com base n",
    "de acordo com a transcri",
)
_PREAMBLE_PATTERN = re.compile(
    "(?:" + "|".join(re.escape(start) for start in _PREAMBLE_WORDS) + r")(?!\w)"
)

_INCOMPLETE_REMARKS: tuple[str, ...] = (
    "transcricao esta incompleta",
    "transcricao incompleta",
    "transcricao parece incompleta",
    "fala foi cortada",
    "frase foi cortada",
    "texto esta incompleto",
    "transcript is incomplete",
    "transcript seems incomplete",
    "was cut off",
)


@dataclass(frozen=True, slots=True)
class DeterministicScores:
    """What the fixed rules say about one answer."""

    non_empty: bool
    is_refusal: bool
    refusal_only_on_noise: bool
    no_preamble: bool


def _plain(text: str) -> str:
    return " ".join(strip_accents(text).lower().split())


def is_refusal(answer: str) -> bool:
    """Return True if the answer is a short text that declines to answer."""
    plain = _plain(answer)
    if not plain or len(plain.split()) > REFUSAL_MAX_WORDS:
        return False
    return any(marker in plain for marker in _REFUSAL_MARKERS)


def has_preamble(answer: str) -> bool:
    """Return True if the answer opens with courtesy or says that the transcript is cut."""
    plain = _plain(answer)
    if _PREAMBLE_PATTERN.match(plain) is not None:
        return True
    if any(plain.startswith(start) for start in _PREAMBLE_PREFIXES):
        return True
    return any(remark in plain for remark in _INCOMPLETE_REMARKS)


def score_answer(answer: str, *, expect_refusal: bool) -> DeterministicScores:
    """Apply the fixed rules to one answer.

    `refusal_only_on_noise` is True when the answer refuses noise or answers
    speech. It is False for a false refusal and for a false acceptance.
    """
    non_empty = bool(answer.strip())
    refusal = is_refusal(answer)
    return DeterministicScores(
        non_empty=non_empty,
        is_refusal=refusal,
        refusal_only_on_noise=non_empty and refusal == expect_refusal,
        no_preamble=non_empty and not has_preamble(answer),
    )


def to_records(scores: DeterministicScores) -> list[ScoreRecord]:
    """Return the scores as database rows."""
    return [
        ScoreRecord(SCORER, "non_empty", float(scores.non_empty)),
        ScoreRecord(SCORER, "is_refusal", float(scores.is_refusal)),
        ScoreRecord(SCORER, "refusal_only_on_noise", float(scores.refusal_only_on_noise)),
        ScoreRecord(SCORER, "no_preamble", float(scores.no_preamble)),
    ]
