# On-Device Python SWE Playbook

## 1. Fast Targeted Pytest Commands
- Always run targeted tests first rather than the entire repository suite:
  - `pytest tests/test_target.py -k "test_name_substring" -q --tb=short`
  - Add `--maxfail=3` to avoid blowing the 5,000-char stdout cap.
- If a test uses `anyio` or async fixtures, remember the grading harness passes `-p no:anyio` unless explicitly required; verify with standard `pytest <file> -q` first.

## 2. High-Frequency Root-Cause Patterns in Python Libraries
- **Inheritance & Mixin Overrides:** The bug reported on a subclass (`APIRouter`, `Client`, `Console`, `Session`) often lives in a base class (`BaseClient`, `Router`, `FieldInfo`, `PreparedRequest`) or a shared utility module 2–3 hops away.
- **Truthy vs. None Checks:** Watch for `if value:` bugs that wrongly treat `0`, `0.0`, `""`, `False`, or `[]` as missing instead of `if value is not None:`.
- **Mutable Default / Shared State / Caching:** Dict/list mutations across requests, stale `@lru_cache` / key canonicalization ignoring kwargs or unhashable types, or modifying caller-owned dicts/headers in place.
- **Header / URL / Encoding Case & Normalization:** Case-insensitive header keys, percent-encoding, multi-byte UTF-8 boundaries, or `bytes` vs. `str` handling.
- **Pydantic v1/v2 & Type Annotation Introspection:** `Optional`, `Union`, `Annotated`, `ForwardRef`, `get_origin`, `get_args`, and Python 3.10+ `X | Y` union syntax.

## 3. Harness Survival Rules
- Never leave `/workspace/repro.py` or temporary files in `/workspace` — put all scratch scripts in `/tmp/repro.py`.
- Never edit `/workspace/pytest.ini`, `/workspace/conftest.py`, or rely on editing test files.
- Always run `check_patch_hygiene.py` before calling `submit_patch()`.
