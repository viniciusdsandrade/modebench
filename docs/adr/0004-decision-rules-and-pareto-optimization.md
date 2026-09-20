# ADR-004: Decision Rules and Pareto Optimization

## Status

Accepted

## Context

Different application roles have different performance requirements. For example, a quick summary role needs low latency (p95 <= 2.0 s), while an in-depth analysis role can accept higher latency (p95 <= 8.0 s) to get higher quality.

When multiple candidate models satisfy a role Service Level Objective (SLO), the selection algorithm must make an objective, reproducible choice that balances quality, latency, and cost.

## Decision

1. SLO Gating:
   A candidate mode is eligible for a role only if it satisfies all role constraints:
   - TTFAT p95 is less than or equal to the maximum threshold.
   - Question accuracy is greater than or equal to the minimum threshold.
   - Error rate is less than or equal to the maximum threshold (if specified).

2. Statistical Confidence Overlap:
   Point estimates are subject to measurement variance. We calculate 95% bootstrap confidence intervals for quality and latency. If the confidence interval of candidate A overlaps with candidate B, their metrics are considered statistically tied.

3. Tie-Breaking Hierarchy:
   - Primary criterion: Highest quality score.
   - First tie-breaker: If quality intervals overlap, choose the mode with lower TTFAT p95 latency.
   - Second tie-breaker: If TTFAT p95 intervals also overlap, choose the mode with lower mean cost.
   - Final tie-breaker: Lexicographical order of mode ID for deterministic repeatability.

4. Output Document:
   The selected winners and rejection reasons are written to recommended_modes.toml for direct consumption by applications like stt-bridge.

## Consequences

- Mode recommendations are objective, reproducible, and explainable.
- Small statistical fluctuations do not cause erratic mode switches.
- Applications can load recommended modes directly via TOML configuration.
