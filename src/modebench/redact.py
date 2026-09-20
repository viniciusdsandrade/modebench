"""Secrets never reach a log line, an error message or a stored record."""

import logging
import re
from collections.abc import Iterable

REDACTED = "[REDACTED]"

# A credential in any text, as a header, as a JSON field or as an assignment.
# The value is replaced to the end of the token. An authentication scheme
# (Bearer, Basic, Token and so on) is one more word before the credential, so
# one optional word goes with it. To redact one word too many is safe. To
# redact one word too few leaves the credential in the text.
_CREDENTIAL_NAMES = (
    "authorization|proxy-authorization|xi-api-key|x-api-key|x-goog-api-key|"
    "api-key|api_key|apikey|access_token|refresh_token|client_secret|secret|password"
)
_HEADER_PATTERN = re.compile(
    rf"(?i)\b({_CREDENTIAL_NAMES})\b"
    r"(['\"]?\s*[:=]\s*['\"]?)(?:[a-z][a-z0-9_-]*\s+)?[^\s,;'\"}]+"
)
# A key or a token in a query string.
_QUERY_PATTERN = re.compile(r"(?i)([?&](?:key|token|api_key|apikey|access_token)=)[^&\s]+")
_FORMATTER = logging.Formatter()


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Return `text` without credential headers, query tokens and known secret values."""
    cleaned = _HEADER_PATTERN.sub(lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", text)
    cleaned = _QUERY_PATTERN.sub(lambda match: f"{match.group(1)}{REDACTED}", cleaned)
    for secret in secrets:
        if len(secret) >= 6:
            cleaned = cleaned.replace(secret, REDACTED)
    return cleaned


class RedactingFilter(logging.Filter):
    """Logging filter that redacts each record before a handler formats it.

    The message is not the only text of a record. A traceback can hold a URL
    or a header of the failed request, so the text of the exception and of the
    stack is redacted too.
    """

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = tuple(secret for secret in secrets if secret)

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        record.msg = redact(message, self._secrets)
        record.args = None
        if record.exc_info:
            # A formatter uses `exc_text` when it is set, and does not format `exc_info` again.
            text = record.exc_text or _FORMATTER.formatException(record.exc_info)
            record.exc_text = redact(text, self._secrets)
        if record.stack_info:
            record.stack_info = redact(record.stack_info, self._secrets)
        return True


def install_redaction(secrets: Iterable[str]) -> RedactingFilter:
    """Put the filter on each handler of the root logger and return it."""
    redacting = RedactingFilter(secrets)
    for handler in logging.getLogger().handlers:
        handler.addFilter(redacting)
    return redacting
