# Project Conventions: modebench

## General Rules

1. Language:
   - Documentation (README, ADRs, architectural notes) must be written in Simplified Technical English (ASD-STE100).
   - Commit messages must be written in Portuguese in the indicative mood (for example: `feat(stt): adiciona testes do dialeto ElevenLabs`).
   - Code comments and docstrings must be in English.

2. Punctuation:
   - Absolutely no em-dashes (U+2014) or en-dashes (U+2013) in any text or code. Use standard hyphens (-), colons (:), or parentheses instead.

3. Testing and Code Quality:
   - Python 3.12 with strict typing (mypy --strict).
   - Ruff for linting and formatting.
   - pytest for automated unit tests.
   - Core modules must maintain >= 90% test coverage.
   - Do not execute live builds or tests during automated verification turns; verify correctness by reading code.

4. Security and Data Protection:
   - API keys and tokens must only be read from `.env` or system environment. Never hardcode secrets.
   - Authorization headers and secret tokens must be redacted from all log outputs.
   - Private datasets must never be routed to `:free` models or OpenRouter routes without `data_collection = "deny"`.
   - `data/private/` and `runs/` directories are strictly ignored by git.
