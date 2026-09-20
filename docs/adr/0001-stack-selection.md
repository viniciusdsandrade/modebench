# ADR-001: Technology Stack Selection

## Status

Accepted

## Context

We need an automated benchmark tool for real-time speech and analysis applications. The tool must choose the best mode (provider, model, and reasoning level) for each application role. We evaluate modes by quality, latency, and cost using our own data.

The project requires reliable performance, strict type safety, fast dependency management, and reproducible execution.

## Decision

We select the following technology stack:

- Python 3.12: Modern runtime with improved performance, stable type annotations, and standard library support.
- uv: High-speed package manager and environment resolver written in Rust. It ensures fast, deterministic installations.
- Typer: Type-driven command line interface builder based on Click and standard Python type hints.
- Pydantic: Data validation and settings management using Python type annotations. It enforces strict schemas for configuration files and database records.
- HTTPX: Modern HTTP client with synchronous and asynchronous APIs, HTTP/2 support, and streaming response capabilities.
- WebSockets: Robust library for client and server WebSocket connections, required for streaming Speech-to-Text evaluations.
- JiWER: Standard evaluation library for Word Error Rate (WER) and Character Error Rate (CER) computations.
- NumPy: High-performance numerical computing library used for bootstrap confidence intervals, percentiles, and statistical aggregations.
- Matplotlib: Standard visualization library used to render Pareto front and latency distribution charts.
- pytest and respx: Modern testing framework with mock HTTP transports for streaming SSE and REST endpoints.
- Ruff: Fast Python linter and code formatter.
- Mypy (strict mode): Static type checker configured with strict type rules to catch type errors before execution.

## Consequences

- The development cycle is fast because uv and Ruff execute quickly.
- Type errors are prevented by Mypy strict mode.
- All HTTP interactions can be mocked cleanly in tests using respx and httpx.MockTransport.
- Memory and compute requirements remain low during statistical analysis due to batched NumPy operations.
