"""The shape of a dataset file and of the filler file.

A case is what a person in a meeting said before the click on Analyze: some
lines of context, the question, and the key points that a good answer has.
"""

from typing import Literal, Self

from pydantic import Field, model_validator

from modebench.config import StrictModel

Speaker = Literal["mic", "system"]


class Line(StrictModel):
    """One line of a transcript: the channel that heard it and the words."""

    speaker: Speaker
    text: str = Field(min_length=1)


class Case(StrictModel):
    """One Analyze case.

    `earlier` and `reference_answer` are for imported sessions: the true
    transcript before the click, and the answer that production gave. A case
    with `expect_refusal` is noise, and the only correct answer is a refusal.
    """

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    context: list[Line] = Field(default_factory=list)
    question: Line | None = None
    key_points: list[str] = Field(default_factory=list)
    expect_refusal: bool = False
    earlier: list[Line] = Field(default_factory=list)
    reference_answer: str | None = None
    needs_annotation: bool = False
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_content(self) -> Self:
        if self.question is None and not self.context and not self.expect_refusal:
            raise ValueError(f"case {self.id} has no question and no context")
        return self


class DatasetFile(StrictModel):
    """A set of cases. `visibility` is `private` for data that must stay on this machine."""

    version: Literal[1] = 1
    language: str = "pt-BR"
    visibility: Literal["public", "private"] = "public"
    cases: list[Case] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_unique_ids(self) -> Self:
        ids = [case.id for case in self.cases]
        duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate case ids: {', '.join(duplicates)}")
        return self


class FillerBlock(StrictModel):
    """A short exchange about one topic. Filler blocks make the long transcripts."""

    topic: str
    lines: list[Line] = Field(min_length=1)


class FillerFile(StrictModel):
    """The meeting talk that comes before a case in an accumulated transcript."""

    version: Literal[1] = 1
    language: str = "pt-BR"
    blocks: list[FillerBlock] = Field(min_length=1)
