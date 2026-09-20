# ADR-002: Measurement Timing and Cost Accounting

## Status

Accepted

## Context

Real-time applications need clear distinctions between different phases of response generation:

1. Time to First Token (TTFT): when the first token of the model arrives. A reasoning token is a token.
2. Time to First Answer Token (TTFAT): when the user sees the first visible word of the answer.
3. Total Latency: when the generation ends.

Models with internal reasoning (such as thinking models) emit reasoning tokens before outputting visible answer text. If we do not separate reasoning from final answer text, latency metrics give false measurements of user wait time.

In addition, benchmark runs must handle network errors and timeouts correctly without distorting statistical percentiles.

## Decision

1. Nanosecond Resolution:
   We record timestamps with time.perf_counter_ns for high precision.

2. Event Parsing:
   The Server-Sent Events (SSE) parser inspects each delta chunk. It records the time of the first valid chunk as `first_chunk_ms`. It marks TTFT on the first chunk that contains a reasoning token or an answer token. It marks TTFAT only when a chunk contains non-empty visible text outside reasoning blocks.

3. Failure Treatment:
   If a request fails, times out, or returns an empty answer, we do not discard it. Instead, we record its latency as the maximum timeout duration (60 seconds by default). This ensures that failed requests penalize the p95 latency and appear in the error rate.

4. Cost Calculation:
   When the provider returns usage costs in the response metadata (e.g. OpenRouter cost field), we record it directly. Otherwise, we calculate cost from input, cached, and output token counts using the configured price table.

5. Cost Ceiling:
   A request with no usage block has no recorded cost. If the provider can bill it (a success, a timeout, a broken stream or an empty answer), the cost ceiling counts it as the estimate of one request. A request that the provider refused (an HTTP error or a transport error) counts as zero. The run record keeps only the measured cost.

6. Connection Reuse:
   The HTTP client keeps an idle connection for 300 seconds. The default of the library is 5 seconds, which is shorter than one round of modes. With the default, the first mode of each endpoint opens a connection for each request, and its latencies contain a TCP and TLS handshake that the other modes do not pay. The raw log records `new_connection` and `connect_ms` for each request.

## Consequences

- Measurements accurately reflect the true user-perceived delay.
- Unreliable models cannot cheat p95 latency by failing early.
- Cost accounting is reproducible across providers.
- The cost ceiling is effective when a provider does not report usage.
- The order of the modes in a round does not change the latency of a mode.
