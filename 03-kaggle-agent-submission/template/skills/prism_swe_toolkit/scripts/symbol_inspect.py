#!/usr/bin/env python3
"""
Exact 1-indexed line-numbered symbol inspector for /workspace.
Reads a target class/function/method definition cleanly within the 5,000-char stdout cap.

Usage:
    python3 symbol_inspect.py <filepath> <symbol_name>
    python3 symbol_inspect.py <symbol_name>
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

MAX_CHARS = 4400
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "build", "dist", "docs", "docs_src", ".venv", "venv"}


def find_symbol_in_file(fpath: Path, target: str) -> List[Tuple[str, int, int, List[str]]]:
    try:
        source = fpath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except Exception:
        return []

    lines = source.splitlines()
    matches: List[Tuple[str, int, int, List[str]]] = []

    def check_node(qual_name: str, node: ast.AST) -> None:
        short = qual_name.split(".")[-1]
        if target == qual_name or target == short:
            start = node.lineno
            end = getattr(node, "end_lineno", start + 30)
            matches.append((qual_name, start, end, lines[start - 1 : end]))

    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef):
            check_node(stmt.name, stmt)
            for item in stmt.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    check_node(f"{stmt.name}.{item.name}", item)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            check_node(stmt.name, stmt)

    return matches


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 symbol_inspect.py [filepath] <symbol_name>")
        sys.exit(1)

    root = Path("/workspace") if Path("/workspace").exists() else Path.cwd()
    if len(sys.argv) >= 3:
        rel_path = sys.argv[1].lstrip("/")
        if rel_path.startswith("workspace/"):
            rel_path = rel_path[len("workspace/") :]
        target_files = [root / rel_path]
        symbol = sys.argv[2]
    else:
        symbol = sys.argv[1]
        target_files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for fname in sorted(filenames):
                if fname.endswith(".py"):
                    target_files.append(Path(dirpath) / fname)

    out_chunks: List[str] = []
    total_chars = 0

    for fp in target_files:
        if not fp.is_file():
            continue
        found = find_symbol_in_file(fp, symbol)
        for qual, start, end, snippet in found:
            rel = str(fp.relative_to(root)) if fp.is_relative_to(root) else str(fp)
            header = f"=== {rel}:{start}-{end} ({qual}) ==="
            numbered = [f"{start + i:4d} | {line}" for i, line in enumerate(snippet)]
            block = header + "\n" + "\n".join(numbered)
            if total_chars + len(block) > MAX_CHARS:
                remaining = MAX_CHARS - total_chars
                if remaining > 200:
                    out_chunks.append(block[:remaining] + "\n... [truncated]")
                break
            out_chunks.append(block)
            total_chars += len(block) + 2
        if total_chars >= MAX_CHARS:
            break

    if not out_chunks:
        print(f"Symbol '{symbol}' not found.")
    else:
        print("\n\n".join(out_chunks))


if __name__ == "__main__":
    main()
