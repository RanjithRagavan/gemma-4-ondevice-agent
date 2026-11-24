---
name: prism_swe_toolkit
description: >-
  Fast, deterministic, zero-dependency Python scripts for multi-hop AST/call-graph
  bug localization (NBD-PPR), exact line-numbered symbol inspection, and
  pre-submission git diff hygiene and syntax verification inside /workspace.
---

# PRISM SWE Toolkit

Use the scripts in this skill via `run_skill_script` to save tool calls and avoid context truncation inside `/workspace`.

## Available Scripts

1. **`scripts/prism_locate.py`** — Multi-hop PRISM Non-Backtracking Code-Graph Localizer.
   - Pass keywords, exception names, or symbol names from the issue description as command-line arguments (e.g., `"AttributeError split partition_by"`).
   - Builds a fast AST + Call/Inheritance/Containment graph of Python source files in `/workspace`, runs 3-hop Non-Backtracking Diffusion (`NBD-PPR`) with Causal Sink Amplification, and prints:
     - Top-10 ranked root-cause symbols with exact `filepath:start_line-end_line`, signatures, and call hops.
     - Top-5 most relevant `test_*.py` files for regression testing.

2. **`scripts/symbol_inspect.py`** — Compact, 1-indexed line-numbered symbol reader.
   - Pass `<filepath> <symbol_name>` (or `<symbol_name>` alone to search across `/workspace`).
   - Outputs exact line-numbered source code of the symbol plus its direct base classes, callers, and callees, bounded cleanly under 4,500 characters.

3. **`scripts/check_patch_hygiene.py`** — Pre-submission verification gate.
   - No arguments required.
   - Checks `git status --short`, flags untracked scratch files in `/workspace`, checks for accidental modifications to protected test/runner files (`pytest.ini`, `conftest.py`, `tests/`), validates Python AST/syntax compilation on every modified `.py` file, prints `git diff --stat`, and suggests targeted `pytest` commands.

## Resources

- **`resources/swe_playbook.md`**: Load via `load_skill_resource` for concise troubleshooting patterns covering FastAPI/Pydantic/Starlette, HTTPX/Requests, Rich, typing/Python 3.13 edge cases, and pytest invocation flags.
