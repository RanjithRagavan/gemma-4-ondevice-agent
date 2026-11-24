# Entry 03 — Kaggle Prediction Competition Submission (`gemma-4-developer-agent`)

Complete, strictly validated Google ADK declarative submission packages (`submission.zip`) for **[Google - The Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview)** ($110,000 total prize pool: **$65K Main Leaderboard** + **$10K Small Model Bonus Prize**).

---

## Ready-to-Upload Submission Archives (`dist/`)

Because teams may select **two final submissions**—and submissions using `gemma-4-e4b-it-qat-w4a16-ct` are eligible for **both** the General Leaderboard AND the **$10,000 Small Model Bonus Prize**—this package builds and validates both variants automatically:

| Archive | Base Model (`ALLOWED_MODEL_NAMES`) | Target Prize Track | Key Settings |
| :--- | :--- | :--- | :--- |
| **`dist/submission_31b.zip`** | `gemma-4-31b-it-qat-w4a16-ct` | **Main Leaderboard ($37K 1st Place)** | `thinking_budget: 3072`, `include_thoughts: false`, `max_tool_calls: 45`, `max_time_minutes: 8` |
| **`dist/submission_e4b.zip`** | `gemma-4-e4b-it-qat-w4a16-ct` | **$10K Small Model Bonus Prize + Main** | `thinking_budget: 2048`, `include_thoughts: false`, `max_tool_calls: 40`, `max_time_minutes: 6` |

> **How to Submit on Kaggle:** Click **Submit Prediction** on [the competition page](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview) and upload `dist/submission_31b.zip` (rename to `submission.zip` if desired) and `dist/submission_e4b.zip` as your two tracked entries before the **Nov 25, 2026 Entry Deadline** (final deadline **Dec 2, 2026**).

---

## Architecture & Competitive Advantages Engineered for `swegemma`

1. **Zero-Bloat Multi-Turn Reasoning (`include_thoughts: false` + `thinking_budget: 3072`):**
   * Per `HARNESS_README.md` Section 2.4, setting `thinking_budget > 0` with `include_thoughts: false` enables full deep reasoning on every turn while stripping verbose `<|channel>thought...<channel|>` blocks from outgoing conversation history—preventing premature compaction (14,336-token threshold) across 30+ tool calls.
2. **Context-Isolated Read-Only Sub-Agents (`agent_tool`):**
   * `sub_agents/code_locator.yaml`: Runs graph navigation, file searches, and PRISM multi-hop AST ranking in an isolated sub-session and returns only a concise **Localization Dossier**.
   * `sub_agents/patch_reviewer.yaml`: Audits `git status`, `git diff`, syntax compilation, and edge-case regressions before final submission.
3. **Sandboxed PRISM ADK Skill Toolkit (`skills/prism_swe_toolkit/`):**
   * `scripts/prism_locate.py`: Runs our 3-hop **Non-Backtracking Causal Diffusion (`NBD-PPR`)** directly inside `/workspace` in `<1.5s` with zero external dependencies, ranking root-cause symbols + discovering relevant `test_*.py` files in 1 tool call.
   * `scripts/symbol_inspect.py`: Exact 1-indexed line-numbered symbol reader bounded cleanly under the 5,000-character stdout cap.
   * `scripts/check_patch_hygiene.py`: Automated pre-submission gate verifying non-empty production diffs, zero stray `/workspace` repro scripts, untouched `pytest.ini`/`conftest.py`, and valid Python AST compilation.
4. **Zero-Gotcha Contract Compliance (`validate_submission.py`):**
   * Zero `..` path traversal components in any `!include` directive.
   * Zero bare `{identifier}` curly braces in any prompt markdown (prevents Google ADK `session.state` `KeyError` crashes).
   * Strictly permitted file extensions (`.yaml`, `.md`, `.py`), single declared base model, and `agent.yaml` at the root of each `.zip` archive.

---

## Rebuild, Validate & Test Locally

```bash
cd 03-kaggle-agent-submission
python3 build_submissions.py
```

## Optional: Upgrade with a Fine-Tuned PEFT LoRA Adapter on Kaggle L4x4

Use [`training/train_gemma4_lora.py`](training/train_gemma4_lora.py) inside a Kaggle L4x4 notebook attached to the competition dataset (`tasks.jsonl` + `graphs/`) to train a `r=32` PEFT `.safetensors` adapter, place it under `template/adapters/main_lora/`, add `adapter: main_lora` to `template/agent.yaml`, and re-run `python3 build_submissions.py`.
