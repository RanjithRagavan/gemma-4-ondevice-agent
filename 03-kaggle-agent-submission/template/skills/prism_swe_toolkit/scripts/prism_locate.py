#!/usr/bin/env python3
"""
PRISM-Gemma Multi-Hop Code-Graph Bug Localizer (NBD-PPR) for /workspace.
Zero external dependencies (pure Python stdlib). Runs in <1.5s on typical repositories.

Usage:
    python3 prism_locate.py <query keywords, error message, or symbol names...>
"""

from __future__ import annotations

import ast
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    "node_modules",
    "docs",
    "docs_src",
    "doc",
    "examples",
    "benchmarks",
    "scripts",
    "site-packages",
}

MAX_FILES = 600
MAX_FILE_BYTES = 350_000


@dataclass
class SymbolNode:
    node_id: str
    file_path: str
    symbol_name: str
    node_type: str  # "class", "function", "method", "test"
    start_line: int
    end_line: int
    signature: str
    doc_first_line: str
    tokens: List[str]
    calls: List[str] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)


def tokenize_identifier(text: str) -> List[str]:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    raw = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", s2.lower())
    out: List[str] = []
    for tok in raw:
        parts = [p for p in tok.split("_") if len(p) > 1]
        out.extend(parts)
        if len(tok) > 2:
            out.append(tok)
    return out


def is_test_path(rel_path: str) -> bool:
    parts = Path(rel_path).parts
    name = Path(rel_path).name.lower()
    return (
        any(p.lower() in ("test", "tests", "testing") for p in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def collect_python_files(root: Path) -> Tuple[List[Path], List[Path]]:
    src_files: List[Path] = []
    test_files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".") and not d.endswith(".egg-info")
        )
        for fname in sorted(filenames):
            if not fname.endswith(".py"):
                continue
            fpath = Path(dirpath) / fname
            try:
                if fpath.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            rel = str(fpath.relative_to(root))
            if is_test_path(rel):
                test_files.append(fpath)
            else:
                src_files.append(fpath)
            if len(src_files) + len(test_files) >= MAX_FILES:
                return src_files, test_files
    return src_files, test_files


class WorkspacePrismGraph:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.nodes: Dict[str, SymbolNode] = {}
        self.symbol_index: Dict[str, List[str]] = defaultdict(list)
        self.adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.idf: Dict[str, float] = {}

    def _add_node(self, node: SymbolNode) -> None:
        self.nodes[node.node_id] = node
        self.symbol_index[node.symbol_name].append(node.node_id)
        short = node.symbol_name.split(".")[-1]
        if short != node.symbol_name:
            self.symbol_index[short].append(node.node_id)

    def _add_edge(self, src: str, dst: str, etype: str) -> None:
        if src != dst and src in self.nodes and dst in self.nodes:
            self.adj[src].append((dst, etype))

    def parse_file(self, fpath: Path, is_test: bool) -> None:
        rel = str(fpath.relative_to(self.root))
        try:
            source = fpath.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except Exception:
            return
        lines = source.splitlines()

        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef):
                self._extract_class(rel, stmt, lines, is_test)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._extract_fn(rel, None, None, stmt, lines, is_test)

    def _extract_class(self, rel: str, cls_node: ast.ClassDef, lines: List[str], is_test: bool) -> None:
        cls_id = f"{rel}::{cls_node.name}"
        bases: List[str] = []
        for b in cls_node.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute):
                bases.append(b.attr)
        start = cls_node.lineno
        end = getattr(cls_node, "end_lineno", start + 5)
        doc = (ast.get_docstring(cls_node) or "").splitlines()
        doc_first = doc[0].strip()[:120] if doc else ""
        sig = f"class {cls_node.name}({', '.join(bases)}):"
        header = "\n".join(lines[start - 1 : min(start + 10, end)])
        tokens = tokenize_identifier(f"{rel} {cls_node.name} {sig} {doc_first} {header}")

        self._add_node(
            SymbolNode(
                node_id=cls_id,
                file_path=rel,
                symbol_name=cls_node.name,
                node_type="test" if is_test else "class",
                start_line=start,
                end_line=end,
                signature=sig,
                doc_first_line=doc_first,
                tokens=tokens,
                bases=bases,
            )
        )
        for item in cls_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._extract_fn(rel, cls_id, cls_node.name, item, lines, is_test)

    def _extract_fn(
        self,
        rel: str,
        parent_id: Optional[str],
        cls_name: Optional[str],
        fn_node: ast.FunctionDef | ast.AsyncFunctionDef,
        lines: List[str],
        is_test: bool,
    ) -> None:
        qual = f"{cls_name}.{fn_node.name}" if cls_name else fn_node.name
        fn_id = f"{rel}::{qual}"
        start = fn_node.lineno
        end = getattr(fn_node, "end_lineno", start + 10)
        args = [a.arg for a in fn_node.args.args][:8]
        prefix = "async def" if isinstance(fn_node, ast.AsyncFunctionDef) else "def"
        sig = f"{prefix} {qual}({', '.join(args)}):"
        doc = (ast.get_docstring(fn_node) or "").splitlines()
        doc_first = doc[0].strip()[:120] if doc else ""
        body_snippet = "\n".join(lines[start - 1 : min(end, start + 60)])
        tokens = tokenize_identifier(f"{rel} {qual} {sig} {doc_first} {body_snippet}")

        calls: List[str] = []
        for sub in ast.walk(fn_node):
            if isinstance(sub, ast.Call):
                if isinstance(sub.func, ast.Name):
                    calls.append(sub.func.id)
                elif isinstance(sub.func, ast.Attribute):
                    calls.append(sub.func.attr)

        ntype = "test" if (is_test or fn_node.name.startswith("test_")) else ("method" if cls_name else "function")
        self._add_node(
            SymbolNode(
                node_id=fn_id,
                file_path=rel,
                symbol_name=qual,
                node_type=ntype,
                start_line=start,
                end_line=end,
                signature=sig,
                doc_first_line=doc_first,
                tokens=tokens,
                calls=calls,
            )
        )
        if parent_id:
            self._add_edge(parent_id, fn_id, "CONTAINS")

    def finalize_graph(self) -> None:
        # Resolve cross-references (cap overly generic names like 'get', 'append', 'items')
        for nid, node in list(self.nodes.items()):
            for base in node.bases:
                targets = self.symbol_index.get(base, [])
                if 0 < len(targets) <= 6:
                    for t in targets:
                        self._add_edge(nid, t, "INHERITS")
            seen_calls: Set[str] = set()
            for callee in node.calls:
                if callee in seen_calls or len(callee) <= 2:
                    continue
                seen_calls.add(callee)
                targets = self.symbol_index.get(callee, [])
                if 0 < len(targets) <= 5:
                    for t in targets:
                        if node.node_type == "test" and self.nodes[t].node_type != "test":
                            self._add_edge(nid, t, "TESTS")
                        elif node.node_type != "test" and self.nodes[t].node_type != "test":
                            self._add_edge(nid, t, "CALLS")

        # Compute IDF
        df: Dict[str, int] = defaultdict(int)
        n_docs = max(1, len(self.nodes))
        for node in self.nodes.values():
            for tok in set(node.tokens):
                df[tok] += 1
        self.idf = {t: math.log(1.0 + (n_docs - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def localize(self, query: str) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        q_tokens = tokenize_identifier(query)
        exact_idents = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", query))
        has_trace = bool(re.search(r"(Traceback|Error|Exception|File \")", query))

        edge_w = {
            "CALLS": 2.8 if has_trace else 2.6,
            "INHERITS": 2.4 if has_trace else 2.5,
            "CONTAINS": 1.9,
            "TESTS": 1.8,
        }

        src_ids = [nid for nid, n in self.nodes.items() if n.node_type != "test"]
        test_ids = [nid for nid, n in self.nodes.items() if n.node_type == "test"]
        src_set = set(src_ids)

        # 1. BM25 + exact symbol seed scores
        raw_scores: Dict[str, float] = {}
        for nid, node in self.nodes.items():
            tf_map: Dict[str, int] = defaultdict(int)
            for t in node.tokens:
                tf_map[t] += 1
            bm25 = 0.0
            dl = max(1, len(node.tokens))
            for qt in q_tokens:
                tf = tf_map.get(qt, 0)
                if tf > 0:
                    idf = self.idf.get(qt, 1.0)
                    bm25 += idf * (tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * dl / 75.0))
            short = node.symbol_name.split(".")[-1]
            if node.symbol_name in exact_idents or short in exact_idents:
                bm25 *= 2.5
            if Path(node.file_path).name in query or node.file_path in query:
                bm25 *= 2.0
            raw_scores[nid] = max(bm25, 1e-6)

        src_total = sum(raw_scores[u] for u in src_ids) or 1.0
        s0 = {u: raw_scores[u] / src_total for u in src_ids}

        # 2. Forward causal transitions for NBD-PPR
        transitions: Dict[str, List[Tuple[str, float]]] = {}
        out_causal: Dict[str, int] = {}
        for u in src_ids:
            nbrs = [(v, et) for v, et in self.adj.get(u, []) if v in src_set and et in edge_w]
            out_causal[u] = sum(1 for _, et in nbrs if et in ("CALLS", "INHERITS"))
            tw = sum(edge_w[et] for _, et in nbrs)
            transitions[u] = [(v, edge_w[et] / tw) for v, et in nbrs] if tw > 0 else []

        # 3. 3-Hop Non-Backtracking Wave Propagation
        hop_weights = [0.24, 0.25, 0.27, 0.24]
        accum = {u: hop_weights[0] * s0[u] for u in src_ids}
        frontier = dict(s0)
        prev_frontier = {u: 0.0 for u in src_ids}

        for hw in hop_weights[1:]:
            nxt: Dict[str, float] = defaultdict(float)
            for u, mass in frontier.items():
                if mass <= 1e-8:
                    continue
                for v, prob in transitions[u]:
                    bt = 0.25 if prev_frontier.get(v, 0.0) > mass else 1.0
                    nxt[v] += mass * prob * bt
            wsum = sum(nxt.values())
            if wsum > 1e-9:
                for u, val in nxt.items():
                    accum[u] += hw * (val / wsum)
            prev_frontier = frontier
            frontier = dict(nxt)

        # 4. Causal Sink Amplification
        ranked_src: List[Tuple[str, float]] = []
        for u in src_ids:
            node = self.nodes[u]
            sc = accum[u]
            if node.node_type in ("function", "method"):
                sc *= 1.40 if out_causal[u] == 0 else 1.15
            elif node.node_type == "class":
                sc *= 0.70
            ranked_src.append((u, sc))
        ranked_src.sort(key=lambda x: x[1], reverse=True)

        # 5. Rank relevant existing test files via TESTS edges + lexical overlap
        top_src_set = {u for u, _ in ranked_src[:15]}
        test_file_scores: Dict[str, float] = defaultdict(float)
        for tid in test_ids:
            tnode = self.nodes[tid]
            score = raw_scores[tid]
            for nbr, et in self.adj.get(tid, []):
                if et == "TESTS" and nbr in top_src_set:
                    score += 5.0
            test_file_scores[tnode.file_path] = max(test_file_scores[tnode.file_path], score)

        ranked_tests = sorted(test_file_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_src[:10], ranked_tests[:5]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 prism_locate.py <query terms or error message>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    root = Path("/workspace") if Path("/workspace").exists() else Path.cwd()
    src_files, test_files = collect_python_files(root)

    graph = WorkspacePrismGraph(root)
    for fp in src_files:
        graph.parse_file(fp, is_test=False)
    for fp in test_files[:200]:
        graph.parse_file(fp, is_test=True)
    graph.finalize_graph()

    top_symbols, top_tests = graph.localize(query)

    print(f"=== PRISM-Gemma Localization ({len(src_files)} src files, {len(graph.nodes)} symbols) ===")
    print("\n[Top Ranked Root-Cause Symbols (NBD-PPR 3-Hop)]")
    for idx, (nid, score) in enumerate(top_symbols, 1):
        n = graph.nodes[nid]
        callees = [graph.nodes[v].symbol_name for v, et in graph.adj.get(nid, []) if et in ("CALLS", "INHERITS")][:4]
        callee_str = f" -> {', '.join(callees)}" if callees else ""
        doc_str = f" | {n.doc_first_line}" if n.doc_first_line else ""
        print(f"{idx:2d}. {n.file_path}:{n.start_line}-{n.end_line}  [{n.node_type}] {n.signature}{callee_str}{doc_str}")

    if top_tests:
        print("\n[Most Relevant Existing Test Files]")
        for tpath, _ in top_tests:
            print(f"  - {tpath}")


if __name__ == "__main__":
    main()
