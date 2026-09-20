# modebench

Automated benchmark framework that selects the optimal LLM and Speech-to-Text mode (provider, model, and reasoning effort) for each role in real-time applications such as stt-bridge.

Modes are evaluated across three dimensions:
- Quality (question accuracy, utility, key point coverage, refusal correctness)
- Latency (Time to First Token, Time to First Answer Token, total response duration)
- Cost (USD per million tokens or per audio hour)

All text in this repository follows ASD-STE100 English and contains no em-dashes or en-dashes.

---

## Quickstart

### Prerequisites

- Python 3.12 or newer
- uv package manager
- Valid API keys in `.env` (optional for `--dry-run`)

### Setup

```bash
# Clone and enter directory
cd /Volumes/src/work/modebench

# Copy environment template
cp .env.example .env

# Run smoke test in dry-run mode (no network requests, zero cost)
python -m modebench run --profile smoke --dry-run --no-judge
```

---

## Core Metrics

### Latency

- TTFT (Time to First Token): Elapsed time in milliseconds to receive the first token of the model. A reasoning token is a token. The time of the first event of the stream is recorded apart (`first_chunk_ms`).
- TTFAT (Time to First Answer Token): Elapsed time in milliseconds to receive the first visible word of the actual answer. Reasoning tokens do not count as answer tokens.
- Total Latency: Total elapsed time until the stream ends.
- p50 / p95: 50th and 95th percentiles computed with 95% bootstrap confidence intervals.
- Failure Penalty: Every failed or timed-out request is assigned the full timeout duration (60.0 s default) so it directly penalizes p95 latency.
- Connections: The HTTP client keeps an idle connection for 300 s, so each mode uses an open connection after the warm-up. The raw log (`raw.jsonl`) says for each request if it opened a connection (`new_connection`) and how long the handshake took (`connect_ms`).

### Quality

- Deterministic Quality:
  - Non-empty answer check
  - Refusal correctness (refusal permitted only when input is noise)
  - Preamble detection (rejects conversational fillers like "Sure, here is...")
- Semantic Quality (Blind LLM Judge):
  - Inferred question accuracy (0 or 1)
  - Key points coverage count
  - Utility rating (1 to 5)
  - False refusal and false acceptance rates
  - A mode has a quality estimate only if the judge graded its answers. A run with `--no-judge`, or a run that stopped before the judge, gives latency and error numbers only.

### Speech to Text (STT)

- First Partial Latency: Milliseconds to first interim hypothesis after audio starts.
- Final Utterance Latency: Milliseconds to final settled transcript after speech finishes.
- Word Error Rate (WER) and Character Error Rate (CER): Computed with JiWER after Portuguese text normalization.
- Speaker Diarization Accuracy: Correct speaker attribution ratio.

---

## CLI Usage

### 1. Run Benchmark

```bash
# Smoke profile (fast dry run)
python -m modebench run --profile smoke --dry-run

# Print the plan and the cost estimate of a profile, and stop
python -m modebench run --profile quick --dry-run --estimate-only

# Quick profile with real providers. The ceiling must be above the estimate.
python -m modebench run --profile quick --max-cost 20.0

# Speech to text suite
python -m modebench run --suite stt --profile smoke --dry-run
```

A request with no usage block (a timeout, for example) counts as the estimate of one request for the cost ceiling, so the ceiling holds also when the provider does not report a cost.

### 2. Decide Optimal Modes

Evaluates candidate modes against role Service Level Objectives (SLOs) and writes `recommended_modes.toml`:

```bash
python -m modebench decide --run latest
```

`decide`, `baseline` and `compare` read only a real and complete run. They stop with an error for a dry run and for a run whose status is not `completed` (`failed`, `interrupted`, `stopped_at_cost_ceiling`). With `--allow-dry-run`, `decide` writes only in the run directory, unless you give `--out`.

### 3. Generate Report

Renders a Markdown report and Pareto front visualizations:

```bash
python -m modebench report --run latest
```

### 4. Compare Against Baseline

Make a baseline from a real run, and then compare later runs with it:

```bash
python -m modebench baseline --run latest --out baselines/baseline.json
python -m modebench compare --run latest --baseline baselines/baseline.json
```

A regression is a difference that is outside the noise of the measurement:
- TTFAT p95 is above the baseline by more than `regression.p95_increase_ratio` of `configs/bench.toml` (0.20 by default), and its interval is fully above the interval of the baseline.
- Or the quality interval is fully below the quality interval of the baseline.

`compare` prints a warning if the two runs do not have the same profile, dataset, prompts, timeout or judge.

Exit code:
- `0`: No regression detected
- `1`: Regression detected
- `2`: Command error
- `3`: Cost ceiling reached

---

## How to Add a New Mode

Edit `configs/modes.toml`:

```toml
[[modes]]
id = "my-new-model-low"
provider = "openrouter"
model = "deepseek/deepseek-r1"
family = "deepseek"
roles = ["analysis", "summary"]
private_data_ok = false

[modes.params]
reasoning = { effort = "low" }
provider = { order = ["DeepSeek"], data_collection = "deny" }

[modes.price]
usd_per_mtok_in = 0.55
usd_per_mtok_out = 2.19
usd_per_mtok_cached_in = 0.14
```

---

## How to Add a New Test Case

Edit `data/public/analyze/cases.yaml`:

```yaml
cases:
  - id: custom-case-01
    context:
      - speaker: mic
        text: "Qual e a data limite de entrega da primeira fase?"
    gold_question: "Qual e a data limite de entrega da primeira fase?"
    key_points:
      - "quinze de outubro"
      - "apenas fase inicial"
    reference_answer: "A data limite da primeira fase e dia 15 de outubro."
```

---

## Automation and Deployment

- Systemd: Service and timer unit files in `deploy/systemd/`
- Launchd: macOS property list in `deploy/launchd/com.modebench.weekly.plist`
- Continuous Integration: GitHub Actions workflow in `.github/workflows/ci.yml`
