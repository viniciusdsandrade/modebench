"""Text normalisation for Brazilian Portuguese, before the error rates are computed.

A word error rate must count recognition errors, and not the differences of
writing between a reference and a recogniser. The reference says "23" and the
recogniser writes "vinte e três"; one writes "pra" and the other "para". The
normalisation makes those equal: lower case, numbers as words, no punctuation,
the contractions of speech expanded, and the sounds of hesitation removed.
Accents stay, because an accent changes the word in Portuguese.
"""

import re
import unicodedata

from modebench.stt.config import SttNormalization

_UNITS = (
    "zero um dois três quatro cinco seis sete oito nove dez onze doze treze quatorze "
    "quinze dezesseis dezessete dezoito dezenove"
).split()
_TENS = {
    2: "vinte",
    3: "trinta",
    4: "quarenta",
    5: "cinquenta",
    6: "sessenta",
    7: "setenta",
    8: "oitenta",
    9: "noventa",
}
_HUNDREDS = {
    1: "cento",
    2: "duzentos",
    3: "trezentos",
    4: "quatrocentos",
    5: "quinhentos",
    6: "seiscentos",
    7: "setecentos",
    8: "oitocentos",
    9: "novecentos",
}
_SCALES = ((1_000_000_000, "bilhão", "bilhões"), (1_000_000, "milhão", "milhões"))
_ORDINALS_M = (
    "primeiro segundo terceiro quarto quinto sexto sétimo oitavo nono décimo"
).split()
_CONTRACTIONS = {
    "pra": "para",
    "pras": "para as",
    "pro": "para o",
    "pros": "para os",
    "tá": "está",
    "tão": "estão",
    "tô": "estou",
    "tava": "estava",
    "né": "não é",
    "cê": "você",
    "cês": "vocês",
    "vc": "você",
    "vcs": "vocês",
    "tb": "também",
    "tbm": "também",
}
_HESITATIONS = frozenset(
    {"ah", "ahn", "eh", "ehm", "er", "err", "hm", "hmm", "hmmm", "hum", "humm", "uh", "uhh"}
    | {"uhm", "ãh", "hã", "éh", "ahm"}
)

_CURRENCY = re.compile(r"r\$\s*(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d{2}))?")
_ORDINAL = re.compile(r"(\d+)\s*([ºª°])")
_NUMBER = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?")
_PUNCTUATION = re.compile(r"[^\w\s]|_")


def _below_thousand(value: int) -> str:
    if value == 100:
        return "cem"
    parts: list[str] = []
    hundreds, rest = divmod(value, 100)
    if hundreds:
        parts.append(_HUNDREDS[hundreds])
    if rest:
        if rest < 20:
            parts.append(_UNITS[rest])
        else:
            tens, units = divmod(rest, 10)
            parts.append(_TENS[tens] if not units else f"{_TENS[tens]} e {_UNITS[units]}")
    return " e ".join(parts)


def number_to_words_pt(value: int) -> str:
    """Return a whole number as Portuguese words: 1234 is "mil duzentos e trinta e quatro"."""
    if value == 0:
        return "zero"
    if value < 0:
        return "menos " + number_to_words_pt(-value)
    if value >= 10**12:
        return " ".join(_UNITS[int(digit)] for digit in str(value))
    groups: list[tuple[int, str]] = []
    remaining = value
    for scale, singular, plural in _SCALES:
        count, remaining = divmod(remaining, scale)
        if count:
            groups.append((count, f"{_below_thousand(count)} {singular if count == 1 else plural}"))
    thousands, remaining = divmod(remaining, 1000)
    if thousands:
        groups.append((thousands, "mil" if thousands == 1 else f"{_below_thousand(thousands)} mil"))
    if remaining:
        groups.append((remaining, _below_thousand(remaining)))
    if len(groups) == 1:
        return groups[0][1]
    last_count, last_words = groups[-1]
    joiner = " e " if last_count < 100 or last_count % 100 == 0 else " "
    return " ".join(words for _, words in groups[:-1]) + joiner + last_words


def _digits(text: str) -> str:
    return " ".join(_UNITS[int(digit)] for digit in text)


def _number_words(match: re.Match[str]) -> str:
    whole, _, fraction = match.group(0).partition(",")
    words = number_to_words_pt(int(whole.replace(".", "")))
    if fraction:
        spoken = _digits(fraction) if len(fraction) > 2 or fraction.startswith("0") else None
        words += " vírgula " + (spoken or number_to_words_pt(int(fraction)))
    return f" {words} "


def _currency_words(match: re.Match[str]) -> str:
    words = number_to_words_pt(int(match.group(1).replace(".", ""))) + " reais"
    cents = match.group(2)
    if cents and int(cents):
        words += " e " + number_to_words_pt(int(cents)) + " centavos"
    return f" {words} "


def _ordinal_words(match: re.Match[str]) -> str:
    value = int(match.group(1))
    if 1 <= value <= len(_ORDINALS_M):
        word = _ORDINALS_M[value - 1]
        if match.group(2) == "ª":
            word = word[:-1] + "a"
        return f" {word} "
    return f" {number_to_words_pt(value)} "


def normalize_pt(text: str, options: SttNormalization | None = None) -> str:
    """Return the text in the form in which a reference and a hypothesis are compared."""
    chosen = options if options is not None else SttNormalization()
    value = unicodedata.normalize("NFC", text).lower()
    if chosen.expand_numbers:
        value = _CURRENCY.sub(_currency_words, value)
        value = value.replace("%", " por cento ")
        value = _ORDINAL.sub(_ordinal_words, value)
        value = _NUMBER.sub(_number_words, value)
    value = _PUNCTUATION.sub(" ", value.replace("-", " "))
    tokens = value.split()
    if chosen.expand_contractions:
        tokens = [part for token in tokens for part in _CONTRACTIONS.get(token, token).split()]
    if chosen.drop_hesitations:
        tokens = [token for token in tokens if token not in _HESITATIONS]
    return " ".join(tokens)
