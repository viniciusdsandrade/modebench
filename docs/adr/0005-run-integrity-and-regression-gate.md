# ADR-005: Run Integrity and the Regression Gate

## Status

Accepted

## Context

Other people and other programs use three outputs of the benchmark as facts: the decision (`recommended_modes.toml`), the baseline, and the result of `compare`. A review found conditions in which these outputs came from data that was not sufficient:

- A run that an error or the operator stopped kept the status `running`. `--run latest` then selected it.
- A dry run and a run that stopped at the cost ceiling could become a decision or a comparison.
- With no judge, the quality of a mode was the mean of its failures and its noise cases only, and the decision rejected the mode for an accuracy of 0.00.
- A speech session that the provider ended in the middle of the audio was recorded as a success.
- The regression gate compared two point estimates. Two equal runs differ by more than that in a normal week.

## Decision

1. Run Status:
   A run ends with one of these statuses: `completed`, `stopped_at_cost_ceiling`, `failed`, `interrupted`. The executor writes the status in a `finally` block, so no run stays `running` after its process ends.

2. Usable Runs:
   `decide`, `baseline` and `compare` stop with an error for a run of another suite, for a run that is not `completed`, and for a dry run. `--allow-dry-run` is only for a test of the pipeline, and a dry run never writes the default `recommended_modes.toml`.

3. Quality Needs Verdicts:
   A mode with no verdict of the judge has no quality estimate. A mode with verdicts for less than 90 percent of its answers is not eligible for a role. ADR-003 gives the reason.

4. Speech Sessions:
   A session is a failure if it ends before each chunk of audio went out. The reason of the close stays in the error message. A defect in a parser is the error of one session, and it does not stop the suite. The speech suite has the same cost ceiling during a run as the Analyze suites.

5. Regression Gate:
   A regression must be outside the noise of the measurement. For latency, TTFAT p95 is above the permitted ratio and its interval is fully above the interval of the baseline. For quality, the interval is fully below the interval of the baseline. Overlapping intervals are a tie, as in ADR-004. `compare` gives a warning if the profile, the dataset, the prompts, the timeout or the judge are not the same in the two runs.

6. Privacy by Location:
   The host of an endpoint decides that a route goes through OpenRouter, and not only the `privacy` label of the config. A speech manifest below `data/private/` is private, as a dataset is. The session importer does not write in a directory of the repository that git follows, and only the owner can read its output.

## Consequences

- A decision, a baseline and a comparison come only from real and complete runs.
- A run with no judge gives latency and error numbers, and it says clearly that it has no quality data.
- The gate has fewer false alarms. A small regression in a small run can stay below the noise, so use the `full` profile for a release decision.
- An operator who wants to compare two runs that are not complete must read the report of each run.
