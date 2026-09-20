"""The SSE parser: events, comments, and the arrival time of each event."""

from modebench.providers.sse import SseParser


def test_event_is_dispatched_at_the_blank_line_with_the_time_of_its_data_line() -> None:
    parser = SseParser()
    assert parser.feed('data: {"a": 1}', 100) is None
    event = parser.feed("", 250)
    assert event is not None
    assert event.data == '{"a": 1}'
    assert event.at_ns == 100


def test_comment_lines_are_counted_and_are_not_events() -> None:
    parser = SseParser()
    assert parser.feed(": OPENROUTER PROCESSING", 1) is None
    assert parser.feed("", 2) is None
    assert parser.comments == 1


def test_data_lines_of_one_event_are_joined_and_keep_the_first_time() -> None:
    parser = SseParser()
    parser.feed("data: first", 10)
    parser.feed("data: second", 20)
    event = parser.feed("", 30)
    assert event is not None
    assert event.data == "first\nsecond"
    assert event.at_ns == 10


def test_other_fields_are_ignored_and_only_one_space_is_removed() -> None:
    parser = SseParser()
    parser.feed("event: message", 1)
    parser.feed("id: 7", 2)
    parser.feed("data:  two spaces", 3)
    event = parser.feed("", 4)
    assert event is not None
    assert event.data == " two spaces"


def test_flush_returns_an_event_that_the_stream_left_open() -> None:
    parser = SseParser()
    parser.feed("data: [DONE]", 5)
    event = parser.flush()
    assert event is not None
    assert event.data == "[DONE]"
    assert parser.flush() is None


def test_a_blank_line_with_no_data_gives_no_event() -> None:
    parser = SseParser()
    assert parser.feed("", 1) is None
    assert parser.flush() is None
