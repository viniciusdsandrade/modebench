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
   The judge must return a structured JSON object. The parser validates the object against a strict Pydantic model. If validation fails, the runner retries the evaluation once. If it fails again, it records a judge failure rather than making an unverified guess.

## Consequences

- Objective metrics (preambles, refusals, empty answers) are fast and cost zero tokens.
- Subjective evaluation is blind and resistant to model brand bias.
- Invalid judge outputs are caught deterministically.
