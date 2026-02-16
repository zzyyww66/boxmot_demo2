# CLAUDE.md – BoxMOT Guidelines

## Build & Environment
- **Python Version:** 3.11
- **Package Manager:** `uv`
- **Install Dependencies:** `uv sync --all-extras --all-groups`
- **Execution Context:** Always run python entry points as modules from repo root.
    - **Good:** `uv run python -m boxmot.engine.cli ...`
    - **Bad:** `python boxmot/engine/cli.py ...` (Breaks imports)

## Common Commands
- **Run CLI:** `uv run python -m boxmot.engine.cli <command> [args]`
- **Show Help:** `uv run python -m boxmot.engine.cli --help`
- **Run All Tests:** `uv run pytest`
- **Run Specific Test:** `uv run pytest tests/path/to/test.py`
- **Smoke Tests (Examples):**
    - `uv run python -m boxmot.engine.cli track --source <src>`
    - `uv run python -m boxmot.engine.cli generate --source <src>`

## Coding Standards
- **Typing:** Use Python type hints and docstrings for all new/modified code.
- **Imports:** Keep sorted and minimal. Avoid `try/except` wraps on imports unless necessary.
- **Logging:** Use `LOGGER` for library code. `print` is allowed only in CLI entry points for UX.
- **CLI Development:**
    - Use reusable decorators (e.g., `core_options`, `plural_model_options`).
    - Use helpers (`parse_tuple`, `parse_hw_tuple`) instead of ad-hoc parsing.
    - Update help strings and README examples if CLI behavior changes.
- **Performance:** Do not commit large model weights/binaries. Use deterministic seeds for tests.

## Workflow & Git
- **Branches:** `codex/<short-topic>`
- **Commits:** Conventional Commits format (`feat:`, `fix:`, `refactor:`, `docs:`, `ci:`, `perf:`).
- **Scope:** One logical change per PR.
- **Testing Constraints:** If GPU/CUDA is missing in environment, document skipped tests rather than faking results.