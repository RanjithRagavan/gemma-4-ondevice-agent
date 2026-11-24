#!/usr/bin/env python3
"""
Pre-submission Patch Hygiene, Syntax Compilation, and Protected-File Auditor for /workspace.
Run before calling submit_patch() to catch stray files, broken syntax, or accidental config edits.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import List

PROTECTED_BASENAMES = {
    "conftest.py",
    "pytest.ini",
    ".pytest.ini",
    "pyproject.toml",
    "tox.ini",
    "setup.cfg",
    "sitecustomize.py",
    "usercustomize.py",
    "_swegemma_stubs.py",
}


def is_protected_path(rel_path: str) -> bool:
    p = Path(rel_path)
    name = p.name.lower()
    if name in PROTECTED_BASENAMES or name.endswith(".pth"):
        return True
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    if any(part.lower() in ("test", "tests", "testing") for part in p.parts[:-1]) and name.endswith(".py"):
        return True
    return False


def run_git(args: List[str], cwd: Path) -> str:
    res = subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True, text=True)
    return res.stdout.strip()


def main() -> None:
    root = Path("/workspace") if Path("/workspace").exists() else Path.cwd()
    issues: List[str] = []

    status = run_git(["status", "--short"], root)
    changed_tracked = [line for line in run_git(["diff", "--name-only", "HEAD"], root).splitlines() if line.strip()]
    untracked = [line for line in run_git(["ls-files", "--others", "--exclude-standard"], root).splitlines() if line.strip()]

    print("=== PRISM Pre-Submission Patch Hygiene Report ===")
    print(f"Tracked modified files ({len(changed_tracked)}): {changed_tracked}")
    print(f"Untracked files ({len(untracked)}): {untracked}")

    all_touched = changed_tracked + untracked
    non_protected = [f for f in all_touched if not is_protected_path(f)]
    protected_touched = [f for f in all_touched if is_protected_path(f)]

    if not non_protected:
        issues.append("BLOCKER: No non-protected library source files have been modified! Your patch would be EMPTY after harness filtering.")

    if protected_touched:
        issues.append(
            f"WARNING: Protected test/config files modified ({protected_touched}). "
            "The grading harness strips these! Restore them via `git checkout HEAD -- <file>` if accidental."
        )

    for uf in untracked:
        stem = Path(uf).stem.lower()
        if any(k in stem for k in ("repro", "tmp", "temp", "debug", "scratch")):
            issues.append(f"BLOCKER: Scratch/repro file '{uf}' found inside /workspace! Delete it or move to /tmp/.")

    # AST syntax verification on every touched .py file
    for fpath in all_touched:
        full = root / fpath
        if full.is_file() and fpath.endswith(".py"):
            try:
                ast.parse(full.read_text(encoding="utf-8", errors="replace"), filename=fpath)
                print(f"  [AST OK] {fpath}")
            except SyntaxError as exc:
                issues.append(f"BLOCKER: SyntaxError in {fpath}:{exc.lineno}: {exc.msg}")

    diff_stat = run_git(["diff", "--stat", "HEAD"], root)
    if diff_stat:
        print("\n[git diff --stat HEAD]")
        print(diff_stat)

    if issues:
        print("\n[ISSUES FOUND — FIX BEFORE submit_patch()]")
        for item in issues:
            print(f"  - {item}")
        sys.exit(1)
    else:
        print("\nVERDICT: PASS — Patch is clean, syntax-valid, and ready for submit_patch().")


if __name__ == "__main__":
    main()
