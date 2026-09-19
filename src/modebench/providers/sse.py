"""A parser for server-sent events that keeps the arrival time of each event.

The parser gets one line at a time, with the clock reading of that line. An
event keeps the time of its first `data` line, because that is the moment at
which the token was in the process. Comment lines are the keep-alive of some
gateways; they are counted and they are not events.
"""

from dataclasses import dataclass

DONE = "[DONE]"


@dataclass(frozen=True, slots=True)
class SseEvent:
    """The data of one event and the clock reading of its first data line."""

    data: str
    at_ns: int


class SseParser:
    """Turns the lines of an event stream into events."""

    def __init__(self) -> None:
        self._data: list[str] = []
        self._at_ns: int | None = None
        self.comments = 0

    def feed(self, line: str, now_ns: int) -> SseEvent | None:
        """Take one line without its line ending. Return an event when one is complete."""
        if line == "":
            return self._dispatch()
        if line.startswith(":"):
            self.comments += 1
            return None
        name, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if name == "data":
            if self._at_ns is None:
                self._at_ns = now_ns
            self._data.append(value)
        return None

    def flush(self) -> SseEvent | None:
        """Return the event that a stream left open when it ended, if there is one."""
        return self._dispatch()

    def _dispatch(self) -> SseEvent | None:
        if not self._data or self._at_ns is None:
            self._data = []
            self._at_ns = None
            return None
        event = SseEvent(data="\n".join(self._data), at_ns=self._at_ns)
        self._data = []
        self._at_ns = None
        return event
