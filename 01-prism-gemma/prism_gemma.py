"""
PRISM-Gemma: Personalized Repository Indexing via Structural Multi-Hop Code-Graphs
for On-Device Gemma 4 Software Engineering Agents.

Core library implementing:
1. Heterogeneous AST & Symbol Code-Graph Construction (Definitions, Calls, Inheritance, Imports, Tests)
2. Relation-Weighted Personalized PageRank (RW-PPR) with Issue-Type Adaptive Conductance
3. Budget-Constrained Subgraph Extraction & Dominator-Topological Prefix Linearization (Topo-KV)
4. Graph-Proximity Reward Shaping (GPRS) for GRPO / LoRA Post-Training of Gemma 4
"""

from __future__ import annotations

import ast
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class CodeNode:
    node_id: str          # e.g. "pkg/module.py::ClassName.method_name"
    file_path: str
    symbol_name: str
    node_type: str        # "module", "class", "function", "method", "test"
    start_line: int
    end_line: int
    signature: str
    docstring: str
    body_text: str
    token_estimate: int
    calls: List[str] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    raises: List[str] = field(default_factory=list)


@dataclass
class CodeEdge:
    src: str
    dst: str
    edge_type: str        # "CONTAINS", "CALLS", "INHERITS", "IMPORTS", "TESTS", "RAISES_FLOW"
    weight: float = 1.0


class HeterogeneousCodeGraph:
    """
    Constructs and queries a multi-relational Heterogeneous Code Property Graph (H-CPG)
    directly from repository source files with zero external binary dependencies.
    """

    DEFAULT_EDGE_WEIGHTS: Dict[str, Dict[str, float]] = {
        "stack_trace": {
            "CALLS": 2.8,
            "INHERITS": 2.4,
            "CONTAINS": 1.9,
            "IMPORTS": 1.4,
            "TESTS": 2.0,
            "CALLED_BY": 0.35,
            "INHERITED_BY": 0.6,
            "CONTAINED_BY": 0.25,
            "IMPORTED_BY": 0.2,
            "TESTED_BY": 0.3,
        },
        "semantic_bug": {
            "CALLS": 2.6,
            "INHERITS": 2.5,
            "CONTAINS": 1.8,
            "IMPORTS": 1.3,
            "TESTS": 1.8,
            "CALLED_BY": 0.4,
            "INHERITED_BY": 0.7,
            "CONTAINED_BY": 0.3,
            "IMPORTED_BY": 0.2,
            "TESTED_BY": 0.3,
        },
    }

    def __init__(self) -> None:
        self.nodes: Dict[str, CodeNode] = {}
        self.edges: List[CodeEdge] = []
        self.adj: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        self.symbol_index: Dict[str, List[str]] = defaultdict(list)
        self.idf_cache: Dict[str, float] = {}

    def add_node(self, node: CodeNode) -> None:
        self.nodes[node.node_id] = node
        self.symbol_index[node.symbol_name].append(node.node_id)
        short_name = node.symbol_name.split(".")[-1]
        if short_name != node.symbol_name:
            self.symbol_index[short_name].append(node.node_id)

    def add_edge(self, src: str, dst: str, edge_type: str, weight: float = 1.0) -> None:
        if src in self.nodes and dst in self.nodes and src != dst:
            self.edges.append(CodeEdge(src, dst, edge_type, weight))
            self.adj[src].append((dst, edge_type, weight))

    @staticmethod
    def tokenize_identifier(text: str) -> List[str]:
        # Split camelCase and snake_case into normalized sub-tokens
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
        s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", s2.lower())
        expanded = []
        for t in tokens:
            parts = [p for p in t.split("_") if len(p) > 1]
            expanded.extend(parts)
            if len(t) > 2:
                expanded.append(t)
        return expanded

    def build_from_directory(self, root_dir: str) -> None:
        root_path = Path(root_dir)
        py_files = sorted(root_path.rglob("*.py"))
        for file_path in py_files:
            rel_path = str(file_path.relative_to(root_path))
            try:
                source = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            self._parse_python_file(rel_path, source)
        self._resolve_cross_references()
        self._compute_idf()

    def build_from_virtual_files(self, files: Dict[str, str]) -> None:
        for rel_path, source in sorted(files.items()):
            self._parse_python_file(rel_path, source)
        self._resolve_cross_references()
        self._compute_idf()

    def _parse_python_file(self, rel_path: str, source: str) -> None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return

        lines = source.splitlines()
        mod_id = f"{rel_path}::<module>"
        is_test_file = "test" in Path(rel_path).name.lower()

        mod_imports: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mod_imports.append(node.module)
                for alias in node.names:
                    mod_imports.append(alias.name)

        mod_doc = ast.get_docstring(tree) or ""
        header_lines = "\n".join(lines[: min(25, len(lines))])
        self.add_node(
            CodeNode(
                node_id=mod_id,
                file_path=rel_path,
                symbol_name=Path(rel_path).stem,
                node_type="test" if is_test_file else "module",
                start_line=1,
                end_line=len(lines),
                signature=f"# Module {rel_path}",
                docstring=mod_doc,
                body_text=header_lines,
                token_estimate=max(16, len(header_lines.split()) * 4 // 3),
                imports=mod_imports,
            )
        )

        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef):
                self._extract_class(rel_path, mod_id, stmt, lines, is_test_file)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._extract_function(rel_path, mod_id, None, stmt, lines, is_test_file)

    def _extract_class(
        self,
        rel_path: str,
        mod_id: str,
        cls_node: ast.ClassDef,
        lines: List[str],
        is_test_file: bool,
    ) -> None:
        cls_id = f"{rel_path}::{cls_node.name}"
        bases = []
        for b in cls_node.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute):
                bases.append(b.attr)

        start = cls_node.lineno
        end = getattr(cls_node, "end_lineno", start + 10)
        cls_doc = ast.get_docstring(cls_node) or ""
        sig = f"class {cls_node.name}({', '.join(bases)}):"
        class_header = "\n".join(lines[start - 1 : min(start + 12, end)])

        self.add_node(
            CodeNode(
                node_id=cls_id,
                file_path=rel_path,
                symbol_name=cls_node.name,
                node_type="test" if is_test_file or cls_node.name.startswith("Test") else "class",
                start_line=start,
                end_line=end,
                signature=sig,
                docstring=cls_doc,
                body_text=class_header,
                token_estimate=max(20, len(class_header.split()) * 4 // 3),
                bases=bases,
            )
        )
        self.add_edge(mod_id, cls_id, "CONTAINS", 1.0)
        self.add_edge(cls_id, mod_id, "CONTAINED_BY", 1.0)

        for item in cls_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._extract_function(rel_path, cls_id, cls_node.name, item, lines, is_test_file)

    def _extract_function(
        self,
        rel_path: str,
        parent_id: str,
        cls_name: Optional[str],
        fn_node: ast.FunctionDef | ast.AsyncFunctionDef,
        lines: List[str],
        is_test_file: bool,
    ) -> None:
        qual_name = f"{cls_name}.{fn_node.name}" if cls_name else fn_node.name
        fn_id = f"{rel_path}::{qual_name}"
        start = fn_node.lineno
        end = getattr(fn_node, "end_lineno", start + 15)
        body_text = "\n".join(lines[start - 1 : end])
        fn_doc = ast.get_docstring(fn_node) or ""

        args = [a.arg for a in fn_node.args.args]
        sig = f"def {qual_name}({', '.join(args)}):"

        calls: List[str] = []
        raises: List[str] = []
        for sub in ast.walk(fn_node):
            if isinstance(sub, ast.Call):
                if isinstance(sub.func, ast.Name):
                    calls.append(sub.func.id)
                elif isinstance(sub.func, ast.Attribute):
                    calls.append(sub.func.attr)
            elif isinstance(sub, ast.Raise) and sub.exc is not None:
                if isinstance(sub.exc, ast.Call) and isinstance(sub.exc.func, ast.Name):
                    raises.append(sub.exc.func.id)
                elif isinstance(sub.exc, ast.Name):
                    raises.append(sub.exc.id)

        is_test = is_test_file or fn_node.name.startswith("test_")
        node_type = "test" if is_test else ("method" if cls_name else "function")

        self.add_node(
            CodeNode(
                node_id=fn_id,
                file_path=rel_path,
                symbol_name=qual_name,
                node_type=node_type,
                start_line=start,
                end_line=end,
                signature=sig,
                docstring=fn_doc,
                body_text=body_text,
                token_estimate=max(24, int(len(body_text.split()) * 1.35)),
                calls=calls,
                raises=raises,
            )
        )
        self.add_edge(parent_id, fn_id, "CONTAINS", 1.0)
        self.add_edge(fn_id, parent_id, "CONTAINED_BY", 1.0)

    def _resolve_cross_references(self) -> None:
        for node_id, node in list(self.nodes.items()):
            # Resolve inheritance
            for base_name in node.bases:
                for target_id in self.symbol_index.get(base_name, []):
                    self.add_edge(node_id, target_id, "INHERITS", 1.0)
                    self.add_edge(target_id, node_id, "INHERITED_BY", 1.0)

            # Resolve calls & test relationships
            for callee in node.calls:
                for target_id in self.symbol_index.get(callee, []):
                    if node.node_type == "test" and self.nodes[target_id].node_type != "test":
                        self.add_edge(node_id, target_id, "TESTS", 1.0)
                        self.add_edge(target_id, node_id, "TESTED_BY", 1.0)
                    else:
                        self.add_edge(node_id, target_id, "CALLS", 1.0)
                        self.add_edge(target_id, node_id, "CALLED_BY", 1.0)

            # Resolve module imports
            for imp in node.imports:
                imp_tail = imp.split(".")[-1]
                for target_id in self.symbol_index.get(imp_tail, []):
                    self.add_edge(node_id, target_id, "IMPORTS", 1.0)
                    self.add_edge(target_id, node_id, "IMPORTED_BY", 1.0)

    def _compute_idf(self) -> None:
        df: Dict[str, int] = defaultdict(int)
        total_docs = max(1, len(self.nodes))
        for node in self.nodes.values():
            tokens = set(
                self.tokenize_identifier(
                    f"{node.symbol_name} {node.signature} {node.docstring} {node.body_text}"
                )
            )
            for t in tokens:
                df[t] += 1
        self.idf_cache = {
            t: math.log(1.0 + (total_docs - freq + 0.5) / (freq + 0.5))
            for t, freq in df.items()
        }

    def compute_seed_distribution(self, query: str) -> Tuple[Dict[str, float], str]:
        """
        Computes hybrid lexical + structural symbol seed distribution s_0(v) over V
        and detects issue regime ('stack_trace' vs 'semantic_bug').
        """
        has_trace = bool(re.search(r"(Traceback|File \".*\.py\"|Error:|Exception)", query))
        regime = "stack_trace" if has_trace else "semantic_bug"

        q_tokens = self.tokenize_identifier(query)
        exact_symbols = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", query))

        scores: Dict[str, float] = {}
        for node_id, node in self.nodes.items():
            doc_text = f"{node.symbol_name} {node.signature} {node.docstring} {node.body_text}"
            doc_tokens = self.tokenize_identifier(doc_text)
            if not doc_tokens:
                scores[node_id] = 1e-6
                continue

            tf_map: Dict[str, int] = defaultdict(int)
            for t in doc_tokens:
                tf_map[t] += 1

            bm25 = 0.0
            k1, b = 1.5, 0.75
            avg_dl = 65.0
            dl = len(doc_tokens)
            for qt in q_tokens:
                if qt in tf_map:
                    idf = self.idf_cache.get(qt, 1.0)
                    tf = tf_map[qt]
                    bm25 += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avg_dl))

            # Boost exact identifier or symbol hits
            short_sym = node.symbol_name.split(".")[-1]
            if node.symbol_name in exact_symbols or short_sym in exact_symbols:
                bm25 *= 2.6
            if Path(node.file_path).name in exact_symbols or node.file_path in query:
                bm25 *= 2.2

            # Slight down-weighting of pure test nodes during localization unless explicitly queried
            if node.node_type == "test" and "test" not in query.lower():
                bm25 *= 0.45

            scores[node_id] = max(bm25, 1e-6)

        total = sum(scores.values())
        return {k: v / total for k, v in scores.items()}, regime

    def run_rw_ppr(
        self,
        query: str,
        damping: float = 0.78,
        max_iter: int = 4,
        tol: float = 1e-6,
    ) -> Dict[str, float]:
        """
        Non-Backtracking Relation-Weighted Personalized PageRank (NBD-PPR):
        Propagates seed mass s_0 across K directed causal hops (CALLS, INHERITS, CONTAINS)
        while suppressing 2-cycle backtracking and module-hub short-circuits, and amplifying
        downstream causal leaf implementations.
        """
        s0, regime = self.compute_seed_distribution(query)
        edge_weights = self.DEFAULT_EDGE_WEIGHTS[regime]

        # Exclude module pseudo-nodes from absorbing symbol walk mass
        symbol_ids = [nid for nid, n in self.nodes.items() if n.node_type != "module"]
        sym_set = set(symbol_ids)

        # Normalize seed distribution over symbol nodes
        sym_total = sum(s0[u] for u in symbol_ids) or 1.0
        s_sym = {u: s0[u] / sym_total for u in symbol_ids}

        # Build forward causal transition matrix (excluding module hub nodes)
        forward_types = {"CALLS", "INHERITS", "CONTAINS", "TESTS"}
        transitions: Dict[str, List[Tuple[str, float]]] = {}
        out_causal_calls: Dict[str, int] = {}

        for u in symbol_ids:
            neighbors = [
                (v, etype, base_w)
                for v, etype, base_w in self.adj.get(u, [])
                if v in sym_set and etype in forward_types
            ]
            out_causal_calls[u] = sum(1 for _, et, _ in neighbors if et in ("CALLS", "INHERITS"))
            if not neighbors:
                transitions[u] = []
                continue
            weighted = []
            total_w = 0.0
            for v, etype, base_w in neighbors:
                w = base_w * edge_weights.get(etype, 1.0)
                weighted.append((v, w))
                total_w += w
            transitions[u] = [(v, w / total_w) for v, w in weighted] if total_w > 0 else []

        # Multi-hop non-backtracking wave propagation (k = 1 .. K)
        hop_weights = [0.22, 0.24, 0.28, 0.26]  # Hop 0, Hop 1, Hop 2, Hop 3+
        accumulated = {u: hop_weights[0] * s_sym[u] for u in symbol_ids}

        frontier = dict(s_sym)
        prev_frontier = {u: 0.0 for u in symbol_ids}

        for hop_idx in range(1, len(hop_weights)):
            next_frontier = {u: 0.0 for u in symbol_ids}
            hw = hop_weights[hop_idx]
            for u, mass in frontier.items():
                if mass <= 1e-9:
                    continue
                for v, prob in transitions[u]:
                    # Non-backtracking suppression: attenuate transitions returning to strong prior frontier
                    backtrack_penalty = 0.25 if prev_frontier.get(v, 0.0) > frontier.get(u, 0.0) else 1.0
                    next_frontier[v] += mass * prob * backtrack_penalty

            # Normalize hop wave so deep hops are not exponentially starved
            wave_sum = sum(next_frontier.values())
            if wave_sum > 1e-9:
                for u in symbol_ids:
                    norm_val = next_frontier[u] / wave_sum
                    accumulated[u] += hw * norm_val
            prev_frontier = frontier
            frontier = next_frontier

        # Causal leaf/implementation boost: functions/methods at the end of a call chain
        # (out_causal_calls == 0) or base methods that implement core logic
        final_scores: Dict[str, float] = {}
        for nid, node in self.nodes.items():
            if node.node_type == "module":
                final_scores[nid] = 1e-6
                continue
            score = accumulated.get(nid, 1e-6)
            if node.node_type in ("function", "method"):
                if out_causal_calls.get(nid, 0) == 0:
                    score *= 1.45  # Causal terminal implementation node
                else:
                    score *= 1.15
            elif node.node_type == "class":
                score *= 0.65  # Prefer concrete methods/functions over class headers
            final_scores[nid] = score

        total_final = sum(final_scores.values()) or 1.0
        return {k: v / total_final for k, v in final_scores.items()}

    def extract_topological_context(
        self,
        query: str,
        token_budget: int = 1200,
        skeleton_ratio: float = 0.35,
    ) -> Tuple[List[CodeNode], str]:
        """
        Selects a high-mass subgraph under a strict token budget B and linearizes nodes
        in upstream-first topological order (Topo-KV) so invariant base classes/interfaces
        occupy the static KV-cache prefix while mutable edit targets sit at the suffix.
        """
        ppr_scores = self.run_rw_ppr(query)
        ranked_ids = sorted(ppr_scores.keys(), key=lambda k: ppr_scores[k], reverse=True)

        selected_ids: List[str] = []
        used_tokens = 0
        primary_budget = int(token_budget * (1.0 - skeleton_ratio))

        for nid in ranked_ids:
            node = self.nodes[nid]
            if node.node_type == "module" and len(selected_ids) > 0:
                continue
            if used_tokens + node.token_estimate <= primary_budget or not selected_ids:
                selected_ids.append(nid)
                used_tokens += node.token_estimate
            if used_tokens >= primary_budget:
                break

        # Add 1-hop structural bridge neighbors (signatures/skeletons) for remaining budget
        bridge_candidates: Set[str] = set()
        for nid in selected_ids:
            for nbr, etype, _ in self.adj.get(nid, []):
                if nbr not in selected_ids and etype in ("CALLS", "INHERITS", "CALLED_BY"):
                    bridge_candidates.add(nbr)

        bridge_ranked = sorted(bridge_candidates, key=lambda k: ppr_scores.get(k, 0.0), reverse=True)
        skeleton_ids: List[str] = []
        for bid in bridge_ranked:
            sig_tokens = max(12, len(self.nodes[bid].signature.split()) * 2)
            if used_tokens + sig_tokens <= token_budget:
                skeleton_ids.append(bid)
                used_tokens += sig_tokens

        # Topological prefix ordering: upstream dependencies (bases, callees, skeletons) first,
        # downstream high-probability edit targets last for KV-cache prefix stability.
        all_chosen = skeleton_ids + selected_ids
        ordered_ids = self._topological_prefix_sort(all_chosen, ppr_scores, set(skeleton_ids))

        prompt_blocks: List[str] = []
        for nid in ordered_ids:
            node = self.nodes[nid]
            if nid in skeleton_ids:
                prompt_blocks.append(
                    f"# [UPSTREAM CONTRACT] {node.file_path}:{node.start_line}\n{node.signature}\n    \"\"\"{node.docstring}\"\"\""
                )
            else:
                prompt_blocks.append(
                    f"# [ACTIVE SUBGRAPH] {node.file_path}:{node.start_line}-{node.end_line}\n{node.body_text}"
                )

        return [self.nodes[nid] for nid in ordered_ids], "\n\n".join(prompt_blocks)

    def _topological_prefix_sort(
        self,
        node_ids: List[str],
        ppr_scores: Dict[str, float],
        skeleton_set: Set[str],
    ) -> List[str]:
        """
        Orders upstream definitions (inherited bases, imported utilities, skeleton contracts)
        at the beginning of the prompt (static across agent turns) and downstream bug targets
        at the end of the prompt.
        """
        sub_set = set(node_ids)
        in_degree: Dict[str, int] = {nid: 0 for nid in sub_set}
        dep_graph: Dict[str, List[str]] = defaultdict(list)

        for nid in sub_set:
            for nbr, etype, _ in self.adj.get(nid, []):
                if nbr in sub_set and etype in ("CALLS", "INHERITS", "IMPORTS"):
                    # nbr is upstream of nid -> nbr should appear BEFORE nid in prefix
                    dep_graph[nbr].append(nid)
                    in_degree[nid] += 1

        ready = sorted(
            [nid for nid, deg in in_degree.items() if deg == 0],
            key=lambda x: (0 if x in skeleton_set else 1, ppr_scores.get(x, 0.0)),
        )
        ordered: List[str] = []
        while ready:
            curr = ready.pop(0)
            ordered.append(curr)
            for nxt in dep_graph[curr]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    ready.append(nxt)
                    ready.sort(key=lambda x: (0 if x in skeleton_set else 1, ppr_scores.get(x, 0.0)))

        # Append any remaining nodes (cycles) sorted by ascending PPR so top edit target is last
        remaining = [nid for nid in sub_set if nid not in ordered]
        remaining.sort(key=lambda x: (0 if x in skeleton_set else 1, ppr_scores.get(x, 0.0)))
        ordered.extend(remaining)
        return ordered

    def shortest_path_distance(self, src: str, targets: Set[str]) -> int:
        if src in targets:
            return 0
        visited = {src}
        queue = [(src, 0)]
        while queue:
            curr, dist = queue.pop(0)
            for nbr, _, _ in self.adj.get(curr, []):
                if nbr in targets:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        return 6

    def compute_grpo_trajectory_reward(
        self,
        inspected_nodes: List[str],
        ground_truth_nodes: Set[str],
        patch_code: str,
        tests_passed: bool,
    ) -> Dict[str, float]:
        """
        Graph-Proximity Reward Shaping (GPRS) for GRPO post-training of Gemma 4:
          R_total = R_test + lambda_1 * R_hop_progress + lambda_2 * R_ast_valid - lambda_3 * R_cycle_penalty
        Provides dense structural gradient signal even when small models initially fail unit tests.
        """
        r_test = 1.0 if tests_passed else 0.0

        # AST syntax validity check
        try:
            ast.parse(patch_code)
            r_ast = 0.25
        except SyntaxError:
            r_ast = -0.5

        # Graph-hop progress reward across trajectory steps
        if not inspected_nodes:
            r_hop = -0.3
            r_cycle = 0.0
        else:
            dists = [self.shortest_path_distance(n, ground_truth_nodes) for n in inspected_nodes]
            min_dist = min(dists)
            initial_dist = dists[0]
            # Reward exponential proximity to ground-truth nodes + positive hop reduction
            r_hop = 0.5 * math.exp(-0.7 * min_dist) + 0.15 * max(0, initial_dist - min_dist)
            unique_ratio = len(set(inspected_nodes)) / len(inspected_nodes)
            r_cycle = 0.35 * (1.0 - unique_ratio)

        total = r_test + r_hop + r_ast - r_cycle
        return {
            "r_total": round(total, 4),
            "r_test": round(r_test, 4),
            "r_hop": round(r_hop, 4),
            "r_ast": round(r_ast, 4),
            "r_cycle_penalty": round(r_cycle, 4),
        }
