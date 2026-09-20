# ADR-003: Blind LLM Judge and Deterministic Quality Scorers

## Status

Accepted

## Context

Evaluating natural language answers requires both deterministic rules and semantic understanding. Real-time assistant answers must meet three key criteria:

1. Never output empty text.
2. Refuse to answer only when the audio input is genuine noise.
3. Start directly with the answer, without conversational preambles like "Sure, based on the transcript...".

Semantic evaluation (such as question inference and factual coverage) requires an LLM judge. However, LLM judges can be biased if they know the identity or brand of the evaluated model.

## Decision

1. Deterministic Scorers:
   We evaluate non-emptiness, refusal patterns, and preamble presence using deterministic text rules. These run locally without model calls.

2. Blind LLM Judge:
   We evaluate question inference, key point coverage, and utility with a dedicated LLM judge from a different model family. The judge prompt contains only the fresh transcript, expected question, and candidate answer. All identifiers for the provider, model, and latency are removed.

3. Schema Enforcement:
   The judge must return a structured JSON object. The parser validates the object against a Pydantic model (the types and the ranges are checked, and unknown keys are ignored). If the request or the validation fails, the runner asks again, to a maximum of `judge.max_attempts` attempts (3 by default), with a pause that grows after each failed request. If no attempt gives a valid verdict, it records a judge failure rather than making an unverified guess.

4. Untrusted Text:
   The transcript and the candidate answer are data that the benchmark does not control. The judge message divides its blocks with tags (`<new_stretch>`, `<expected>`, `<answer>`). Before the message is built, each such tag in the data is made inert (`<` becomes `&lt;`), so an answer cannot end its block and give instructions to the judge.

5. Quality Needs Verdicts:
   Some answers have a quality with no judge: a failure, an answer to noise, a false refusal. If no answer of a mode has a verdict, these values are not a sample of the mode, and the mode has no quality estimate. A mode is not eligible for a role if the judge graded less than 90 percent of its answers.

## Consequences

- Objective metrics (preambles, refusals, empty answers) are fast and cost zero tokens.
- Subjective evaluation is blind and resistant to model brand bias.
- Invalid judge outputs are caught deterministically.
