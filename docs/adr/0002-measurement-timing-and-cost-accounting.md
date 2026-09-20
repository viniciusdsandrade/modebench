# ADR-002: Measurement Timing and Cost Accounting

## Status

Accepted

## Context

Real-time applications need clear distinctions between different phases of response generation:

1. Time to First Token (TTFT): when the server sends the first network packet or event.
2. Time to First Answer Token (TTFAT): when the user sees the first visible word of the answer.
3. Total Latency: when the generation ends.

Models with internal reasoning (such as thinking models) emit reasoning tokens before outputting visible answer text. If we do not separate reasoning from final answer text, latency metrics give false measurements of user wait time.

In addition, benchmark runs must handle network errors and timeouts correctly without distorting statistical percentiles.

## Decision

1. Nanosecond Resolution:
   We record timestamps with time.perf_counter_ns for high precision.

2. Event Parsing:
   The Server-Sent Events (SSE) parser inspects each delta chunk. It marks TTFT on the first chunk received. It marks TTFAT only when a chunk contains non-empty visible text outside reasoning blocks.

3. Failure Treatment:
   If a request fails, times out, or returns an empty answer, we do not discard it. Instead, we record its latency as the maximum timeout duration (60 seconds by default). This ensures that failed requests penalize the p95 latency and appear in the error rate.

4. Cost Calculation:
   When the provider returns usage costs in the response metadata (e.g. OpenRouter cost field), we record it directly. Otherwise, we calculate cost from input, cached, and output token counts using the configured price table.

## Consequences

- Measurements accurately reflect the true user-perceived delay.
- Unreliable models cannot cheat p95 latency by failing early.
- Cost accounting is reproducible across providers.
