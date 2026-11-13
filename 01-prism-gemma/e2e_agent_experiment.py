"""
End-to-End On-Device Agent Experiment for PRISM-Gemma.

Runs a REAL Gemma 4 model (via a local llama.cpp OpenAI-compatible server) on all
SWE-HOP-12 scenarios and measures:

1. E2E repair success ("e2e" mode):
   For each scenario x context policy (BM25/grep context vs. PRISM NBD-PPR + Topo-KV
   context), the model must (a) name the root-cause `file.py::symbol` and
   (b) emit the complete corrected file. The patched repository is materialized to
   disk and a behavioral test is executed in a subprocess.

   Metrics per policy: root-cause identification rate, patch AST validity,
   behavioral test pass rate (resolve rate), TTFT (prompt_ms), decode tokens/s.

2. Real KV-cache prefix reuse ("kv" mode):
   A 5-turn iterative repair session is replayed against the live server.
   llama.cpp reports `timings.cache_n` (tokens served from the KV cache) per
   request, giving MEASURED (not simulated) prefix hit rates under three context
   orderings: relevance-descending, alphabetical, and Topo-KV.

3. GPRS on real model rollouts ("gprs" mode):
   Samples G real completions per scenario at temperature 0.9, scores each rollout
   with the GPRS reward vs. the sparse binary reward, and reports informative
   gradient-group rate and Kendall concordance with oracle rollout quality.

Usage:
    python3 e2e_agent_experiment.py e2e  --server http://127.0.0.1:8399 [--scenarios SWE-HOP-01,SWE-HOP-02] [--policies bm25,prism]
    python3 e2e_agent_experiment.py kv   --server http://127.0.0.1:8399 [--scenarios ...]
    python3 e2e_agent_experiment.py gprs --server http://127.0.0.1:8399 [--scenarios ...] [--group-size 6]
    python3 e2e_agent_experiment.py summarize
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evaluate_prism_benchmark import build_benchmark_repositories, rank_bm25_only
from prism_gemma import HeterogeneousCodeGraph

RESULTS_DIR = HERE / "e2e_results"

# ---------------------------------------------------------------------------
# Behavioral tests: each is executed in a fresh subprocess against the
# materialized (patched) repository. They test the *reported behavior* through
# the public entry point, so a correct fix in ANY file counts.
# ---------------------------------------------------------------------------

BEHAVIORAL_TESTS: Dict[str, str] = {
    "SWE-HOP-01": '''
from core.coercion import normalize_numeric_primitive
assert normalize_numeric_primitive(None) is None, "None must be preserved, not coerced to 0.0"
assert normalize_numeric_primitive("3.5") == 3.5
assert normalize_numeric_primitive(2) == 2.0
''',
    "SWE-HOP-02": '''
from orm.expressions import WindowFrameNode
node = WindowFrameNode({"partition_by": ["region", "created_at"]})
clause = node.render_partition_clause()
assert "region" in clause and "created_at" in clause, clause
''',
    "SWE-HOP-03": '''
from cache.backend import InMemoryStore
store = InMemoryStore(ttl=300)
calls = []
def fn(self, sku, include_tax=False):
    calls.append(1)
    return {"sku": sku, "tax": include_tax}
r1 = store.get_or_compute(fn, ("s", "SKU1"), {"include_tax": True})
r2 = store.get_or_compute(fn, ("s", "SKU1"), {"include_tax": False})
assert r1["tax"] is True and r2["tax"] is False, (r1, r2)
assert len(calls) == 2, "different kwarg values must not share a cache key"
''',
    "SWE-HOP-04": '''
from workers.retry_policy import ExponentialBackoffPolicy
attempts = []
class P(ExponentialBackoffPolicy):
    def _send(self, payload):
        attempts.append(1)
        raise TimeoutError("Gateway 504 timeout")
p = P(max_attempts=3)
try:
    p.run_attempt_loop({})
except TimeoutError:
    pass
assert len(attempts) == 3, f"expected 3 attempts, got {len(attempts)}"
''',
    "SWE-HOP-05": '''
from modeling.masking import expand_causal_padding_mask
m = expand_causal_padding_mask([[True, False]], seq_len=2)
assert m[0][0] == 0.0, "valid token must NOT be masked"
assert m[0][1] < -1e8, "padding token must be masked"
''',
    "SWE-HOP-06": '''
from config.loader import RuntimeConfigLoader
cfg = RuntimeConfigLoader().resolve_effective_config({"temperature": 0.0})
assert cfg["temperature"] == 0.0, cfg
''',
    "SWE-HOP-07": '''
from export.traversal import collect_reachable_nodes
class N:
    def __init__(self, name):
        self.name = name
        self.children = []
a, b = N("a"), N("b")
a.children = [b]
b.children = [a]
res = collect_reachable_nodes(a)
names = sorted(n.name for n in res)
assert names == ["a", "b"], names
''',
    "SWE-HOP-08": '''
from auth.validators import is_jwt_expired
assert is_jwt_expired(1000, 975) is True, "inside 30s leeway window -> refresh proactively"
assert is_jwt_expired(1000, 500) is False
assert is_jwt_expired(1000, 1000) is True
''',
    "SWE-HOP-09": '''
from streaming.decoder import IncrementalUTF8Decoder
d = IncrementalUTF8Decoder()
data = "x" * 63 + "\\U0001F600"
b = data.encode("utf-8")
part1, part2 = b[:64], b[64:]
r = d.decode_frame(part1) + d.decode_frame(part2)
assert r == data, repr(r)
''',
    "SWE-HOP-10": '''
from patcher.applier import UnifiedDiffApplier
text = "l1\\nl2\\nl3\\nl4\\nl5"
hunks = [(2, 1, ["a", "b", "c"]), (4, 1, ["z"])]
res = UnifiedDiffApplier().apply_hunks(text, hunks)
assert res == "l1\\na\\nb\\nc\\nl3\\nz\\nl5", res
''',
    "SWE-HOP-11": '''
from pagination.cursor import KeysetPaginator
p = KeysetPaginator(limit=2)
recs = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
page1 = p.fetch_page(recs, None)
page2 = p.fetch_page(recs, page1[-1]["id"])
assert [r["id"] for r in page2] == [3, 4], page2
''',
    "SWE-HOP-12": '''
from peft.scaling import compute_adapter_delta
d16 = compute_adapter_delta(2.0, 3.0, 4.0, rank=8, alpha=16)
d64 = compute_adapter_delta(2.0, 3.0, 4.0, rank=8, alpha=64)
assert abs(d64 / d16 - 4.0) < 1e-6, (d16, d64)
assert abs(d16 - (16 / 8) * 2.0 * 3.0 * 4.0) < 1e-6, d16
''',
}

# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------

def build_bm25_context(graph: HeterogeneousCodeGraph, query: str, top_k: int = 5) -> str:
    """Baseline: top-k BM25 symbols, relevance-descending, full bodies (grep-style agent context)."""
    ranked = [nid for nid in rank_bm25_only(graph, query) if not nid.endswith("::<module>")][:top_k]
    blocks = []
    for nid in ranked:
        node = graph.nodes[nid]
        blocks.append(f"# FILE {node.file_path} (lines {node.start_line}-{node.end_line})\n{node.body_text}")
    return "\n\n".join(blocks)


def build_prism_context(graph: HeterogeneousCodeGraph, query: str, budget: int = 1200) -> str:
    """PRISM: NBD-PPR subgraph selection + Topo-KV prefix linearization."""
    _, ctx = graph.extract_topological_context(query, token_budget=budget)
    return ctx


# ---------------------------------------------------------------------------
# LLM server client
# ---------------------------------------------------------------------------

def chat_completion(server: str, messages: List[Dict], max_tokens: int = 1200,
                    temperature: float = 0.0, seed: int = 0) -> Dict:
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "cache_prompt": True,
    }
    req = urllib.request.Request(
        server.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Response parsing & scoring
# ---------------------------------------------------------------------------

def parse_root_cause(text: str) -> Optional[str]:
    m = re.search(r"ROOT\s*CAUSE\s*:\s*`?([A-Za-z0-9_./-]+\.py)::([A-Za-z0-9_.]+)`?", text)
    if m:
        return f"{m.group(1)}::{m.group(2)}"
    return None


def parse_patched_file(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (file_path, code) from a fenced block, or from a bare '# FILE:' header
    running to the end of the response (compact models often skip fences)."""
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if m:
        code = m.group(1)
        fm = re.search(r"#\s*FILE\s*:\s*([A-Za-z0-9_./-]+\.py)", code)
        path = fm.group(1) if fm else None
        if path is None:
            pm = re.search(r"([A-Za-z0-9_./-]+\.py)", text[: m.start()])
            path = pm.group(1) if pm else None
        return path, code
    # Fallback: bare "# FILE: path" header followed by code until end of text
    m = re.search(r"#\s*FILE\s*:\s*([A-Za-z0-9_./-]+\.py)\n(.*)$", text, re.DOTALL)
    if m:
        return m.group(1), m.group(2).strip()
    return None, None


def claim_matches(claim: Optional[str], ground_truth: Set[str]) -> bool:
    if not claim:
        return False
    if claim in ground_truth:
        return True
    c_file, _, c_sym = claim.partition("::")
    c_short = c_sym.split(".")[-1]
    for gt in ground_truth:
        g_file, _, g_sym = gt.partition("::")
        if c_file == g_file and (c_sym == g_sym or c_short == g_sym.split(".")[-1]):
            return True
    return False


def run_behavioral_test(scenario: Dict, patched_path: Optional[str],
                        patched_code: Optional[str], timeout: int = 30) -> Tuple[bool, str]:
    """Materialize repo, apply patch, run behavioral test in a subprocess."""
    if not patched_path or not patched_code:
        return False, "no parsable patch"
    if patched_path not in scenario["files"]:
        return False, f"patched path {patched_path} not in repository"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for rel, src in scenario["files"].items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(src)
        (root / patched_path).write_text(patched_code)
        test_file = root / "_behavioral_test.py"
        test_file.write_text(BEHAVIORAL_TESTS[scenario["id"]])
        proc = subprocess.run(
            [sys.executable, str(test_file)],
            cwd=td, capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode == 0:
            return True, "pass"
        err = (proc.stderr.strip().splitlines() or ["unknown"])[-1]
        return False, err[:200]


def ast_valid(code: Optional[str]) -> bool:
    if not code:
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert autonomous software-engineering agent running fully offline on a "
    "consumer device. You repair real bugs in Python repositories. Be precise and concise."
)

TASK_TEMPLATE = """{query}

Below are the most relevant repository code sections retrieved for this issue:

{context}

---
Task:
1. Identify the single root-cause function or method responsible for the bug.
   Output exactly one line in the form:
   ROOT CAUSE: <file_path>::<symbol_name>
2. Then output the COMPLETE corrected contents of the file that contains the root cause,
   as one python code block whose first line is `# FILE: <file_path>`.
   Change only what is needed to fix the bug; keep all other definitions intact."""


def e2e_prompt(query: str, context: str) -> List[Dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TASK_TEMPLATE.format(query=query, context=context)},
    ]


# ---------------------------------------------------------------------------
# Mode: e2e
# ---------------------------------------------------------------------------

def save_json(name: str, data) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def append_jsonl(name: str, record: Dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_jsonl(name: str) -> List[Dict]:
    p = RESULTS_DIR / name
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def run_e2e(server: str, scenario_ids: Optional[Set[str]], policies: List[str],
            max_tokens: int = 1400) -> None:
    scenarios = [s for s in build_benchmark_repositories()
                 if not scenario_ids or s["id"] in scenario_ids]
    done = {(r["scenario"], r["policy"]) for r in load_jsonl("e2e_runs.jsonl")}
    for sc in scenarios:
        graph = HeterogeneousCodeGraph()
        graph.build_from_virtual_files(sc["files"])
        contexts = {}
        if "bm25" in policies:
            contexts["bm25"] = build_bm25_context(graph, sc["query"])
        if "prism" in policies:
            contexts["prism"] = build_prism_context(graph, sc["query"])
        for policy, ctx in contexts.items():
            if (sc["id"], policy) in done:
                print(f"[skip] {sc['id']} {policy} already done")
                continue
            t0 = time.time()
            resp = chat_completion(server, e2e_prompt(sc["query"], ctx), max_tokens=max_tokens)
            wall = time.time() - t0
            text = resp["choices"][0]["message"]["content"]
            claim = parse_root_cause(text)
            p_path, p_code = parse_patched_file(text)
            ok, err = run_behavioral_test(sc, p_path, p_code)
            record = {
                "scenario": sc["id"],
                "policy": policy,
                "root_cause_claim": claim,
                "root_cause_correct": claim_matches(claim, sc["ground_truth"]),
                "patch_file": p_path,
                "patch_ast_valid": ast_valid(p_code),
                "test_pass": ok,
                "test_error": err,
                "wall_s": round(wall, 2),
                "prompt_tokens": resp.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": resp.get("usage", {}).get("completion_tokens"),
                "ttft_ms": round(resp.get("timings", {}).get("prompt_ms", 0), 1),
                "decode_tps": round(resp.get("timings", {}).get("predicted_per_second", 0), 2),
                "raw_response": text,
            }
            append_jsonl("e2e_runs.jsonl", record)
            print(f"[{sc['id']} | {policy}] root_ok={record['root_cause_correct']} "
                  f"ast={record['patch_ast_valid']} test={ok} ({err}) "
                  f"ttft={record['ttft_ms']}ms wall={record['wall_s']}s")


# ---------------------------------------------------------------------------
# Mode: kv — measured multi-turn prefix reuse on the live server
# ---------------------------------------------------------------------------

TURN_INSTRUCTIONS = [
    "Inspect the code and describe the suspected defect in one sentence.",
    "Inspect intermediate caller arguments and state what you check next.",
    "Check None coercion and boundary conditions; state your finding.",
    "Draft a candidate fix for the target function in one code line comment.",
    "Verify the post-patch return contract in one sentence.",
]


def run_kv(server: str, scenario_ids: Optional[Set[str]]) -> None:
    scenarios = [s for s in build_benchmark_repositories()
                 if not scenario_ids or s["id"] in scenario_ids]
    done = {(r["scenario"], r["policy"]) for r in load_jsonl("kv_runs.jsonl")}
    for sc in scenarios:
        graph = HeterogeneousCodeGraph()
        graph.build_from_virtual_files(sc["files"])
        root_target = sorted(sc["ground_truth"])[0]
        ppr = graph.run_rw_ppr(sc["query"])
        # Realistic on-device context: top-8 subgraph nodes, full bodies (~1.2k tokens).
        # NOTE: llama.cpp only retains cross-request prompt-cache entries above a minimum
        # sequence length, so tiny contexts would show zero reuse for *every* policy.
        chosen = [nid for nid in sorted(ppr, key=ppr.get, reverse=True)
                  if graph.nodes[nid].node_type != "module"][:8]

        for policy in ("relevance_desc", "alphabetical", "prism_topo_kv"):
            if (sc["id"], policy) in done:
                print(f"[skip] kv {sc['id']} {policy}")
                continue
            if policy == "relevance_desc":
                ordered = sorted(chosen, key=lambda k: ppr[k], reverse=True)
            elif policy == "alphabetical":
                ordered = sorted(chosen)
            else:
                ordered = graph._topological_prefix_sort(list(chosen), ppr, set())
                if root_target in ordered:
                    ordered.remove(root_target)
                    ordered.append(root_target)

            turns = []
            for turn_idx, instr in enumerate(TURN_INSTRUCTIONS):
                blocks = []
                for nid in ordered:
                    body = graph.nodes[nid].body_text
                    if nid == root_target and turn_idx >= 2:
                        body = body + f"\n    # candidate_patch_revision_{turn_idx}"
                    blocks.append(f"### {nid}\n{body}")
                user_msg = (sc["query"] + "\n\n" + "\n\n".join(blocks) +
                            f"\n\nTurn {turn_idx + 1}/5: {instr}")
                resp = chat_completion(
                    server,
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user_msg}],
                    max_tokens=48, temperature=0.0,
                )
                tm = resp.get("timings", {})
                turns.append({
                    "turn": turn_idx + 1,
                    "prompt_n": tm.get("prompt_n", 0),
                    "cache_n": tm.get("cache_n", 0),
                    "prompt_ms": round(tm.get("prompt_ms", 0), 1),
                })
            record = {"scenario": sc["id"], "policy": policy, "turns": turns}
            append_jsonl("kv_runs.jsonl", record)
            hits = [t["cache_n"] / max(1, t["cache_n"] + t["prompt_n"]) for t in turns[1:]]
            print(f"[kv {sc['id']} | {policy}] mean hit rate turns2-5: "
                  f"{100 * sum(hits) / max(1, len(hits)):.2f}%")


# ---------------------------------------------------------------------------
# Mode: gprs — GPRS rewards over REAL sampled rollouts
# ---------------------------------------------------------------------------

def mentioned_node_sequence(text: str, graph: HeterogeneousCodeGraph) -> List[str]:
    """Ordered, de-duplicated graph nodes mentioned in the rollout text (the trajectory)."""
    seq: List[str] = []
    for m in re.finditer(r"([A-Za-z0-9_./-]+\.py)(?:::([A-Za-z0-9_.]+))?", text):
        fpath, sym = m.group(1), m.group(2)
        for nid in graph.nodes:
            n_file, _, n_sym = nid.partition("::")
            if n_file != fpath:
                continue
            if sym is None or n_sym == sym or n_sym.split(".")[-1] == sym.split(".")[-1]:
                if nid not in seq and graph.nodes[nid].node_type != "module":
                    seq.append(nid)
                break
    return seq


def oracle_rank_of(root_ok: bool, valid: bool, passed: bool,
                   seq: List[str], graph: HeterogeneousCodeGraph,
                   gt: Set[str]) -> int:
    if passed:
        return 6
    if root_ok and valid:
        return 5
    if root_ok:
        return 4
    if seq and min(graph.shortest_path_distance(n, gt) for n in seq) <= 1:
        return 3
    if seq:
        return 2
    return 1


def run_gprs(server: str, scenario_ids: Optional[Set[str]], group_size: int,
             backend: str = "mlx") -> None:
    scenarios = [s for s in build_benchmark_repositories()
                 if not scenario_ids or s["id"] in scenario_ids]
    mlx_model = mlx_tok = None
    if backend == "mlx":
        # Real sampled rollouts on an on-device quantized Gemma 4 (MLX 4-bit).
        # (llama.cpp b12180 mis-samples this QAT build at temp>0 on long prompts,
        #  so rollouts are collected via MLX instead; both are Q4-class local quants.)
        import mlx.core as mx
        from mlx_lm import load as mlx_load, generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler
        mlx_model, mlx_tok = mlx_load("mlx-community/gemma-4-e4b-it-4bit")

    for sc in scenarios:
        existing = [r for r in load_jsonl("gprs_rollouts.jsonl") if r["scenario"] == sc["id"]]
        if len(existing) >= group_size:
            print(f"[skip] gprs {sc['id']} ({len(existing)} rollouts)")
            continue
        graph = HeterogeneousCodeGraph()
        graph.build_from_virtual_files(sc["files"])
        ctx = build_prism_context(graph, sc["query"])
        msgs = e2e_prompt(sc["query"], ctx)
        for gi in range(len(existing), group_size):
            if backend == "mlx":
                m = [dict(msgs[0]), msgs[1]]
                m[0]["content"] += " Keep your reasoning under 120 words, then give the answer."
                prompt = mlx_tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                mx.random.seed(1000 + gi)
                text = mlx_generate(
                    mlx_model, mlx_tok, prompt=prompt, max_tokens=2500,
                    sampler=make_sampler(temp=1.0, top_p=0.95), verbose=False,
                )
            else:
                resp = chat_completion(server, msgs, max_tokens=3000,
                                       temperature=1.0, seed=1000 + gi)
                text = resp["choices"][0]["message"]["content"]
            claim = parse_root_cause(text)
            p_path, p_code = parse_patched_file(text)
            ok, err = run_behavioral_test(sc, p_path, p_code)
            root_ok = claim_matches(claim, sc["ground_truth"])
            valid = ast_valid(p_code)
            seq = mentioned_node_sequence(text, graph)
            orank = oracle_rank_of(root_ok, valid, ok, seq, graph, sc["ground_truth"])
            rewards = graph.compute_grpo_trajectory_reward(
                seq or [sorted(graph.nodes)[0]], sc["ground_truth"],
                p_code or "def broken(:", ok,
            )
            record = {
                "scenario": sc["id"], "rollout": gi,
                "root_cause_correct": root_ok, "patch_ast_valid": valid,
                "test_pass": ok, "oracle_rank": orank,
                "trajectory_nodes": seq,
                "r_binary": 1.0 if ok else 0.0,
                "r_gprs": rewards["r_total"],
            }
            append_jsonl("gprs_rollouts.jsonl", record)
            print(f"[gprs {sc['id']} #{gi}] root={root_ok} ast={valid} pass={ok} "
                  f"oracle={orank} r_gprs={rewards['r_total']}")


# ---------------------------------------------------------------------------
# Mode: summarize
# ---------------------------------------------------------------------------

def kendall_tau_b(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = (xs[j] > xs[i]) - (xs[j] < xs[i])
            dy = (ys[j] > ys[i]) - (ys[j] < ys[i])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = ((concordant + discordant + ties_x) * (concordant + discordant + ties_y)) ** 0.5
    return (concordant - discordant) / denom if denom else 0.0


def summarize() -> None:
    out: Dict = {}

    e2e = load_jsonl("e2e_runs.jsonl")
    if e2e:
        policies = sorted({r["policy"] for r in e2e})
        e2e_sum = {}
        for pol in policies:
            rows = [r for r in e2e if r["policy"] == pol]
            n = len(rows)
            e2e_sum[pol] = {
                "n": n,
                "root_cause_identified_pct": round(100 * sum(r["root_cause_correct"] for r in rows) / n, 2),
                "patch_ast_valid_pct": round(100 * sum(r["patch_ast_valid"] for r in rows) / n, 2),
                "resolve_rate_pct": round(100 * sum(r["test_pass"] for r in rows) / n, 2),
                "mean_ttft_ms": round(sum(r["ttft_ms"] for r in rows) / n, 1),
                "mean_decode_tps": round(sum(r["decode_tps"] for r in rows) / n, 2),
                "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in rows) / n, 1),
            }
        out["e2e"] = e2e_sum

    kv = load_jsonl("kv_runs.jsonl")
    if kv:
        kv_sum = {}
        for pol in sorted({r["policy"] for r in kv}):
            rows = [r for r in kv if r["policy"] == pol]
            hits, prefills = [], []
            for r in rows:
                for t in r["turns"][1:]:
                    hits.append(t["cache_n"] / max(1, t["cache_n"] + t["prompt_n"]))
                    prefills.append(t["prompt_ms"])
            kv_sum[pol] = {
                "measured_prefix_hit_rate_pct": round(100 * sum(hits) / len(hits), 2),
                "mean_turn_prompt_ms": round(sum(prefills) / len(prefills), 1),
            }
        out["kv_measured"] = kv_sum

    roll = load_jsonl("gprs_rollouts.jsonl")
    if roll:
        scen = sorted({r["scenario"] for r in roll})
        informative_binary = informative_gprs = 0
        per_scenario_concordance = []
        all_g, all_o = [], []
        for sid in scen:
            rows = [r for r in roll if r["scenario"] == sid]
            rb = [r["r_binary"] for r in rows]
            rg = [r["r_gprs"] for r in rows]
            if max(rb) - min(rb) > 1e-9:
                informative_binary += 1
            if max(rg) - min(rg) > 1e-9:
                informative_gprs += 1
            per_scenario_concordance.append(kendall_tau_b(rg, [r["oracle_rank"] for r in rows]))
            all_g.extend(rg)
            all_o.extend(r["oracle_rank"] for r in rows)
        out["gprs_real_rollouts"] = {
            "scenarios": len(scen),
            "rollouts": len(roll),
            "informative_groups_binary_pct": round(100 * informative_binary / len(scen), 1),
            "informative_groups_gprs_pct": round(100 * informative_gprs / len(scen), 1),
            "mean_per_scenario_kendall_tau": round(sum(per_scenario_concordance) / len(per_scenario_concordance), 4),
            "global_kendall_tau_vs_oracle": round(kendall_tau_b(all_g, all_o), 4),
        }

    save_json("e2e_summary.json", out)
    print(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["e2e", "kv", "gprs", "summarize"])
    ap.add_argument("--server", default="http://127.0.0.1:8399")
    ap.add_argument("--scenarios", default=None)
    ap.add_argument("--policies", default="bm25,prism")
    ap.add_argument("--group-size", type=int, default=6)
    ap.add_argument("--backend", default="mlx", choices=["mlx", "server"])
    ap.add_argument("--max-tokens", type=int, default=1400)
    args = ap.parse_args()

    ids = set(args.scenarios.split(",")) if args.scenarios else None
    if args.mode == "e2e":
        run_e2e(args.server, ids, args.policies.split(","), args.max_tokens)
    elif args.mode == "kv":
        run_kv(args.server, ids)
    elif args.mode == "gprs":
        run_gprs(args.server, ids, args.group_size, args.backend)
    else:
        summarize()


if __name__ == "__main__":
    main()
