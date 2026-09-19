"""Secrets never reach a log line, an error message or a stored record."""

import logging
import re
from collections.abc import Iterable

REDACTED = "[REDACTED]"

# A credential header in any text: the value is replaced to the end of the token.
_HEADER_PATTERN = re.compile(
    r"(?i)\b(authorization|xi-api-key|x-api-key|api-key|x-goog-api-key)\b"
    r"(['\"]?\s*[:=]\s*['\"]?)(?:bearer\s+)?[^\s,;'\"}]+"
)
# A key or a token in a query string.
_QUERY_PATTERN = re.compile(r"(?i)([?&](?:key|token|api_key|apikey|access_token)=)[^&\s]+")


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Return `text` without credential headers, query tokens and known secret values."""
    cleaned = _HEADER_PATTERN.sub(lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", text)
    cleaned = _QUERY_PATTERN.sub(lambda match: f"{match.group(1)}{REDACTED}", cleaned)
    for secret in secrets:
        if len(secret) >= 6:
            cleaned = cleaned.replace(secret, REDACTED)
    return cleaned


class RedactingFilter(logging.Filter):
    """Logging filter that redacts each record before a handler formats it."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = tuple(secret for secret in secrets if secret)

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        record.msg = redact(message, self._secrets)
        record.args = None
        return True


def install_redaction(secrets: Iterable[str]) -> RedactingFilter:
    """Put the filter on each handler of the root logger and return it."""
    redacting = RedactingFilter(secrets)
    for handler in logging.getLogger().handlers:
        handler.addFilter(redacting)
    return redacting
