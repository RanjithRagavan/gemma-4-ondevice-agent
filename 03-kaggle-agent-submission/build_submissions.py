#!/usr/bin/env python3
"""
Builds and strictly validates ready-to-upload Kaggle submission.zip archives for
Google - The Gemma 4 Developer Agent Competition (100% schema-aligned with official sample_submission):

1. dist/submission_31b.zip -> gemma-4-31b-it-qat-w4a16-ct (Main Leaderboard — $37K 1st Place)
2. dist/submission_e4b.zip -> gemma-4-e4b-it-qat-w4a16-ct (Small Model Bonus Prize — $10K + Main)
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict

from validate_submission import validate_path

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "template"
DIST_DIR = BASE_DIR / "dist"

BUILDS: Dict[str, Dict] = {
    "submission_31b.zip": {
        "model": "gemma-4-31b-it-qat-w4a16-ct",
        "coder_thinking_budget": 4096,
        "analyzer_thinking_budget": 2048,
        "timeout_seconds": 90,
        "max_time_minutes": 4,
        "max_tool_calls": 25,
        "max_turns": 50,
    },
    "submission_e4b.zip": {
        "model": "gemma-4-e4b-it-qat-w4a16-ct",
        "coder_thinking_budget": 3072,
        "analyzer_thinking_budget": 1536,
        "timeout_seconds": 90,
        "max_time_minutes": 3,
        "max_tool_calls": 22,
        "max_turns": 45,
    },
}


def build_variant(zip_name: str, cfg: Dict) -> Path:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    out_zip = DIST_DIR / zip_name

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "stage"
        shutil.copytree(
            TEMPLATE_DIR,
            stage,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )

        # Substitute model alias across all YAML files
        for yaml_file in stage.rglob("*.yaml"):
            text = yaml_file.read_text(encoding="utf-8")
            text = text.replace("__MODEL_ALIAS__", cfg["model"])
            yaml_file.write_text(text, encoding="utf-8")

        # Customize thinking budgets & eval_config per variant
        coder_cfg = stage / "configs" / "coder_sampling.yaml"
        coder_cfg.write_text(
            coder_cfg.read_text(encoding="utf-8").replace(
                "thinking_budget: 4096", f"thinking_budget: {cfg['coder_thinking_budget']}"
            ),
            encoding="utf-8",
        )
        an_cfg = stage / "sub_agents" / "configs" / "analyzer_sampling.yaml"
        an_cfg.write_text(
            an_cfg.read_text(encoding="utf-8").replace(
                "thinking_budget: 2048", f"thinking_budget: {cfg['analyzer_thinking_budget']}"
            ),
            encoding="utf-8",
        )
        eval_cfg = stage / "eval_config.yaml"
        eval_cfg.write_text(
            (
                "evaluation:\n"
                f"  timeout_seconds: {cfg['timeout_seconds']}\n"
                f"  max_tool_calls: {cfg['max_tool_calls']}\n"
                f"  max_time_minutes: {cfg['max_time_minutes']}\n"
                f"  max_turns: {cfg['max_turns']}\n"
            ),
            encoding="utf-8",
        )

        # Package with agent.yaml strictly at the ROOT of the zip archive
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fpath in sorted(stage.rglob("*")):
                if fpath.is_file():
                    arcname = str(fpath.relative_to(stage))
                    zf.write(fpath, arcname=arcname)

    return out_zip


def main() -> None:
    built = []
    for zip_name, cfg in BUILDS.items():
        zp = build_variant(zip_name, cfg)
        size_kb = zp.stat().st_size / 1024.0
        print(f"Built {zp.name} ({size_kb:.1f} KB) for model={cfg['model']}")
        built.append(zp)

    print("\nRunning strict ADK/swegemma contract validation...")
    all_ok = all(validate_path(zp) for zp in built)
    if not all_ok:
        sys.exit(1)
    print("\nAll submission archives built and validated cleanly!")


if __name__ == "__main__":
    main()
