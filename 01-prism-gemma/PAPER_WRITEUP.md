# PRISM-Gemma: Non-Backtracking Spectral Code-Graph Diffusion, Topological KV-Prefix Routing, and Graph-Proximity GRPO for On-Device Gemma 4 Software Engineering Agents

**Subtitle:** Overcoming the Multi-Hop Localization, Context-Window, and Sparse-Reward Bottlenecks for Autonomous Repository-Level Coding on Consumer Hardware

**Public Repository & Reproducible Benchmark:** [https://github.com/RanjithRagavan/gemma-4-ondevice-agent/tree/main/01-prism-gemma](https://github.com/RanjithRagavan/gemma-4-ondevice-agent/tree/main/01-prism-gemma)

---

## Abstract

Deploying autonomous software engineering (SWE) agents offline on consumer hardware with compact open models such as Gemma 4 exposes three bottlenecks that cloud-scale frontier agents bypass through brute force. First, when a defect's symptom (an API endpoint, a stack-trace frame) sits 2–4 graph hops from its root cause with zero lexical overlap, local agents get trapped in repetitive `grep` cycles, and standard undirected PageRank repo-maps fail via *symptom pinning* and *2-cycle backtracking* through module hubs. Second, relevance-ranked retrieval orders destroy KV-cache prefix reuse across multi-turn repair loops, inflating time-to-first-token on-device. Third, binary unit-test rewards ($R \in \{0,1\}$) starve GRPO post-training of gradient signal whenever every sampled rollout fails — the common regime for small models on multi-hop bugs.

We introduce **PRISM-Gemma** (*Personalized Repository Indexing via Structural Multi-Hop Code-Graphs*), a graph-reasoning and post-training framework for on-device Gemma 4 agents with three components: **NBD-PPR**, non-backtracking causal diffusion over a typed Heterogeneous Code Property Graph; **Topo-KV**, dominator-topological prompt linearization that pins invariant upstream contracts to the KV-cache prefix and mutable edit targets to the suffix; and **GPRS**, graph-proximity reward shaping that converts shortest-path progress and AST validity into a dense, verifiable signal.

Crucially, every headline claim is validated **end-to-end with a real on-device Gemma 4 (E4B, Q4-class quantization), not simulation**. On our 12-scenario multi-hop benchmark (`SWE-HOP-12`): (i) PRISM-Gemma lifts root-cause Recall@3 from 16.67% (BM25/`grep`) to **83.33%**; (ii) in live repair sessions, Gemma 4 identifies the root cause in **100%** of scenarios with PRISM context vs. 91.67% with BM25 context, and produces behaviorally verified fixes in **83.33% vs. 75.00%**; (iii) measured KV-cache prefix reuse on a live llama.cpp server reaches **84.95%** vs. 42.56%/32.88% for baseline orderings, cutting mean per-turn prefill latency **~3×** (124 ms vs. 369/416 ms); (iv) across 72 real sampled rollouts in the measured zero-pass regime (1/72 pass), GPRS yields informative GRPO gradient groups in **91.7%** of scenarios vs. **8.3%** for binary rewards, at Kendall $\tau = 0.80$ concordance with oracle repair progress.

---

## 1. Introduction

Modern SWE agents rely on frontier cloud models with 128K–1M token contexts and dozens of exploratory turns per issue. On an offline, privacy-preserving device powered by Gemma 4 (E4B/12B/quantized 26B) with an 8K–16K effective context, this brute-force paradigm collapses: the agent cannot afford 30 file reads, and it cannot re-prefill thousands of tokens every turn. Through failure analysis of compact models on repository-level issue resolution, we identify a trilemma connecting **graph representation**, **hardware-aware context scheduling**, and **reinforcement learning**:

* **Why lexical search and naive PageRank fail on multi-hop bugs.** Bug reports name the *symptom* $u_{\text{sym}}$, but the fault often lies 2–4 hops downstream — an inherited base method, a coercion utility, a decorator key builder — sharing zero identifier tokens with the report. Standard Personalized PageRank (as in Aider's RepoMap) suffers *symptom pinning* ($(1-\alpha)$ of mass stays at the seed; hop-$k$ mass decays as $\alpha^k$) and *2-cycle backtracking* (bidirectional `CALLS`/`CALLED_BY` and `CONTAINS`/`CONTAINED_BY` edges bounce the walk back to the seed or into `<module>` hubs).
* **Why relevance-ranked contexts destroy on-device KV caching.** RAG-style pipelines place the highest-scoring (and thus most-edited) snippet first; editing it invalidates the entire KV prefix on every turn.
* **Why binary rewards starve small-model RL.** GRPO needs non-zero reward variance within each group of $G$ rollouts. On multi-hop bugs, when all $G$ rollouts fail the tests, the group advantage vanishes — even if one rollout reached the exact root cause and another looped on `grep`.

### Contributions

1. **NBD-PPR:** non-backtracking spectral diffusion over a typed H-CPG with regime-conditioned edge conductances, module-hub filtering, per-hop wave normalization, and causal sink amplification.
2. **Topo-KV:** deterministic subgraph linearization placing upstream contracts in the static prompt prefix and the mutable edit target at the suffix.
3. **GPRS:** a dense, verifiable composite reward (test outcome + exponential hop-proximity + hop-reduction + AST validity − cyclic-tool penalty) for GRPO on compact models.
4. **Open-source library & benchmark:** zero-dependency `prism_gemma.py` and the `SWE-HOP-12` suite with per-scenario behavioral tests.
5. **Real on-device validation:** all efficiency and reward claims measured against a live Gemma 4 E4B (Q4-class) on a consumer laptop (Apple M2 Max), via llama.cpp and MLX — no simulated metrics in headline results.

---

## 2. Methodology

```text
+-----------------------------------------------------------------------------------+
|                        PRISM-GEMMA ON-DEVICE AGENT PIPELINE                       |
+-----------------------------------------------------------------------------------+
  [Issue Report q] + [Repo Source Files]
           |
           v
  (1) H-CPG Construction: Typed AST & Symbol Graph G = (V, E, T_v, T_e)
      Nodes: {module, class, function, method, test}
      Edges: {CALLS, INHERITS, CONTAINS, IMPORTS, TESTS} + Inverse Relations
           |
           v
  (2) Non-Backtracking Causal Diffusion (NBD-PPR, K=3 Hops):
      - Hybrid BM25 + Identifier Seed s_0(v) & Regime Detection (Trace vs Semantic)
      - Wavefront Propagation f^(k) with 2-Cycle Backtrack Penalty (eta = 0.25)
      - Causal Sink Amplification psi(v) = 1.45 on terminal leaf implementations
           |
           v
  (3) Topo-KV Subgraph Linearization (Budget B <= 1200 Tokens):
      [STATIC KV PREFIX: Upstream Base Classes & 1-Hop Skeletons]
      [MUTABLE KV SUFFIX: Downstream Root-Cause Edit Target v*]
           |
           v
  (4) Gemma 4 (E4B Q4-class) Multi-Turn Repair & GPRS GRPO Reward:
      R_GPRS = R_test + R_hop_progress(d_G(v_t, V*)) + R_AST - R_cycle
      Measured on-device: 84.95% KV prefix reuse, 124 ms mean turn prefill
+-----------------------------------------------------------------------------------+
```

### 2.1 Heterogeneous Code Property Graph (H-CPG)

PRISM-Gemma parses a repository into a directed multi-relational graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{T}_v, \mathcal{T}_e)$ with zero external dependencies. Vertices are structural units $\phi(v) \in \{\text{module}, \text{class}, \text{function}, \text{method}, \text{test}\}$ carrying signature, docstring, body, and token cost; ten directed edge types capture forward causal relations (`CALLS`, `INHERITS`, `CONTAINS`, `IMPORTS`, `TESTS`) and their reverses. Given issue $q$, a BM25 seed distribution $s_0(v)$ with camelCase/snake_case splitting, exact-symbol boosts, and a regime classifier $\rho(q) \in \{\text{stack\_trace}, \text{semantic\_bug}\}$ is computed.

### 2.2 Non-Backtracking Causal Multi-Hop Diffusion (NBD-PPR)

We restrict propagation to symbol vertices and forward causal edges, with regime-conditioned conductances (e.g., $w_{\text{trace}}(\text{CALLS}) = 2.8$, $w_{\text{trace}}(\text{INHERITS}) = 2.4$) defining a row-normalized transition matrix $P_\rho$. Instead of a memoryless stationary chain, we propagate normalized wave fronts $f^{(k)}$ across $K=3$ hops with a non-backtracking gate $\eta$ that attenuates transitions returning toward the previous wave's dominant nodes:

$$\tilde{f}^{(k)}(v) = \sum_{u} f^{(k-1)}(u) \, P_{\rho}(u \to v) \, \eta(u, v; k), \quad \eta = \begin{cases} 0.25 & f^{(k-2)}(v) > f^{(k-1)}(u) \\ 1.0 & \text{otherwise} \end{cases}$$

Each wave is $L_1$-normalized so deep hops are not exponentially starved; hop waves are combined with $\beta = [0.22, 0.24, 0.28, 0.26]$, and a **causal sink amplification** $\psi(v)$ boosts terminal implementation functions ($d_{\text{out}}^{\text{causal}}(v) = 0$, $\times 1.45$), where low-level bugs disproportionately reside.

### 2.3 Budgeted Subgraph Extraction & Topo-KV Linearization

Under token budget $B$ (e.g., 1200), we greedily select full bodies for top-ranked vertices up to $(1-\sigma)B$ ($\sigma = 0.35$) and fill the rest with signature/docstring skeletons of 1-hop structural bridges. **Topo-KV** then linearizes the induced dependency DAG via Kahn's sort ($v \prec u$ if $u$ calls, inherits, or imports $v$), so invariant upstream contracts occupy the static prefix $[0, L_{\text{static}}]$ and the mutable edit target sits at the suffix — maximizing KV-cache prefix stability across turns.

### 2.4 Graph-Proximity Reward Shaping (GPRS)

For GRPO post-training with ground-truth patch nodes $V^*$, let $\mathcal{V}(\tau_i)$ be the nodes inspected by rollout $\tau_i$ and $d_{\mathcal{G}}$ the graph shortest-path distance. The composite reward:

$$R_{\text{GPRS}}(\tau_i) = R_{\text{test}} + 0.5 e^{-0.7 \min_{v} d_{\mathcal{G}}(v, V^*)} + 0.15 \max\!\big(0,\ d_{\mathcal{G}}(v_{i,1}, V^*) - \min_{v} d_{\mathcal{G}}(v, V^*)\big) + R_{\text{AST}} - R_{\text{cycle}}$$

with $R_{\text{AST}} \in \{-0.5, +0.25\}$ (AST parse of the patch) and cycle penalty $R_{\text{cycle}} = 0.35\,(1 - |\text{unique}|/|\mathcal{V}(\tau_i)|)$. Even when all rollouts fail tests, $R_{\text{GPRS}}$ preserves non-zero group variance pointing toward $V^*$.

---

## 3. Experimental Setup & Results

**On-device stack.** All model-involving results below were measured on a consumer laptop (Apple M2 Max, 64 GB) running Gemma 4 E4B at Q4-class quantization: llama.cpp (b12180, Metal) serving `google/gemma-4-E4B-it-qat-q4_0-gguf` for greedy repair and KV measurement, and MLX (`mlx-community/gemma-4-e4b-it-4bit`) for temperature-1.0 rollout sampling. Retrieval and reward machinery are model-free and run in < 3 s/scenario.

### 3.1 The `SWE-HOP-12` Benchmark

12 multi-module Python repositories with 2–3-hop defects where the root cause shares minimal or zero lexical overlap with the bug report: inheritance/coercion chains, stack-trace AST dialect propagation, decorator cache-key canonicalization, retry exception hierarchies, attention-mask polarity, config precedence, DFS cycle handling, JWT leeway sign errors, UTF-8 chunk splitting, multi-hunk diff offset drift, keyset cursor duplication, and LoRA alpha/rank scaling inversion. Each scenario ships a **behavioral test** executed against the patched repository (all 12 fail on the unpatched code).

**Concrete 3-Hop Walkthrough (`SWE-HOP-01`):** The issue reports that `UserUpdateEndpoint.handle_patch` overwrites storage quota with `0.0` when `{'quota': None}` is passed, despite `DecimalQuotaField(allow_null=True)`.
* **Hop 0 (Symptom):** `api/views.py::UserUpdateEndpoint.handle_patch` — named in the report and ranked #1 by BM25, Cosine RAG, and Uniform PageRank, but bug-free.
* **Hop 1 (Serializer):** `AccountPayloadSerializer.validate_fields` calls `self.quota_field.run_parse(...)`.
* **Hop 2 (Inherited Contract):** `DecimalQuotaField` inherits `run_parse` from `core/fields.py::BaseField.run_parse`.
* **Hop 3 (Root-Cause Sink):** `core/coercion.py::normalize_numeric_primitive` returns `0.0` for `None` input — zero token overlap with the report. NBD-PPR propagates mass across all 3 hops and ranks the root cause #1 (MRR = 1.0000 vs. 0.125 for BM25).

### 3.2 Structural Localization (Model-Free)

| Method / Ablation | Recall@1 | Recall@3 | Recall@5 | MRR | Root-Cause Recall@3 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| BM25 / Lexical `grep` | 0.0000 | 0.3333 | 0.7083 | 0.3507 | 16.67% |
| Dense Token Cosine RAG | 0.0000 | 0.2917 | 0.6250 | 0.2861 | 25.00% |
| Uniform RepoMap (Undirected PageRank) | 0.0000 | 0.2917 | 0.6667 | 0.2786 | 25.00% |
| **PRISM-Gemma (Full NBD-PPR, $K=3$)** | **0.3750** | **0.7917** | **0.9167** | **0.8111** | **83.33%** |
| w/o Non-Backtracking | 0.3333 | 0.7500 | 0.8750 | 0.7736 | 75.00% |
| w/o Typed Edge Conductance | 0.3750 | 0.7917 | 0.9167 | 0.8111 | 83.33% |
| w/o Causal Sink Amplification | 0.3750 | 0.7917 | 0.8333 | 0.8056 | 83.33% |
| $K=1$ only | 0.0417 | 0.4583 | 0.7917 | 0.4090 | 41.67% |
| $K=2$ only | 0.2500 | 0.6250 | 0.7500 | 0.6349 | 75.00% |

Lexical and undirected-graph baselines invariably rank the 0-hop symptom first (Recall@1 = 0); truncating diffusion to $K=1$ collapses MRR to 0.4090.

### 3.3 End-to-End Repair with Real On-Device Gemma 4

For each scenario, the model receives the bug report plus a retrieval context — top-5 BM25 bodies (grep-style agent) vs. the PRISM NBD-PPR + Topo-KV subgraph — and must (a) name the root-cause `file::symbol` and (b) emit the complete corrected file. We materialize the patched repo and run the behavioral test (greedy decoding, max 3000 tokens).

| Metric (n = 12, Gemma 4 E4B QAT q4_0, greedy) | BM25/`grep` context | PRISM context |
| :--- | :---: | :---: |
| Root cause correctly identified | 91.67% | **100.00%** |
| Patch AST-valid | 100.00% | **100.00%** |
| **Resolve rate (behavioral test passes)** | 75.00% | **83.33%** |
| Mean TTFT (prefill) | 422 ms | 443 ms |
| Mean decode speed | 72.0 tok/s | 71.8 tok/s |

The retrieval layer, not the model, is the binding constraint: with BM25 context the agent confidently fixes the wrong function (e.g., in `SWE-HOP-01` it coerces `None` deeper instead of preserving it); with PRISM context the same weights repair the true root cause. The residual failures (2/12 under both policies) are full-file-rewrite import drops — a known small-model weakness orthogonal to retrieval.

### 3.4 Measured KV-Cache Prefix Reuse (Live Server)

We replay 5-turn repair sessions against the live llama.cpp server, re-sending the full context each turn with the edit target modified from turn 3 (aider-style stateless re-sends), and read the server's actual cache accounting (`timings.cache_n`):

| Context ordering (measured, turns 2–5) | Prefix hit rate | Mean turn prefill |
| :--- | :---: | :---: |
| Relevance-descending | 42.56% | 369.1 ms |
| Alphabetical file order | 32.88% | 416.1 ms |
| **Topo-KV** | **84.95%** | **124.0 ms** |

Topo-KV's immutable-contracts-first ordering keeps the cached prefix intact when the edit target mutates, cutting per-turn prefill latency **~3×**. One honest boundary: llama.cpp only retains cross-request prompt-cache entries above a minimum sequence length (~512 tokens); scenarios whose full context falls below that floor (2 of 12) show no reuse under *any* ordering — a systems-level argument for packing contexts densely, which Topo-KV's budgeted skeleton filling already does.

### 3.5 GPRS on Real Sampled Rollouts (Zero-Pass Regime)

We sample $G = 6$ rollouts per scenario (temperature 1.0, top-$p$ 0.95) from the on-device model — 72 real trajectories with thinking traces. This regime is genuinely reward-starved: only **1/72** rollouts passes its behavioral test. Scoring each trajectory with binary vs. GPRS rewards:

| Signal across 12 rollout groups | Binary $R_{\text{test}}$ | GPRS $R_{\text{GPRS}}$ |
| :--- | :---: | :---: |
| Groups with informative (non-zero-variance) gradients | 8.3% | **91.7%** |
| Kendall $\tau$ with oracle rollout quality (per-scenario mean) | — | **0.796** |

GPRS separates "reached the root cause but wrote an invalid patch" from "looped at the symptom" even when every rollout fails — exactly the gradient GRPO needs on compact models.

---

## 4. Related Work

* **Repository-level agents & code graphs:** SWE-agent, Agentless, and AutoCodeRover rely on iterative shell exploration or hierarchical prompting over cloud models; RepoGraph and Aider's RepoMap apply standard PPR over symbol graphs. We identify and fix two failure modes of undirected PPR (symptom pinning, 2-cycle backtracking) for multi-hop defects.
* **Spectral graph diffusion & non-backtracking walks:** Personalized PageRank (Page et al., 1999; Jeh & Widom, 2003), heat-kernel diffusion (Chung, 2007), and non-backtracking operators (Alon et al., 2007; Krzakala et al., 2013) underpin modern graph mining; PRISM-Gemma adapts non-backtracking propagation to heterogeneous code graphs with regime-conditioned conductances.
* **Prefix caching & RL for code:** PagedAttention and RadixAttention reuse KV across shared prompt prefixes; GRPO (Shao et al., 2024; DeepSeek-R1) enables critic-free RL but degrades when groups have zero reward variance. Topo-KV and GPRS bridge code-graph topology with both systems.
* **SWE-bench** motivates repository-level evaluation; `SWE-HOP-12` targets the multi-hop localization slice with contamination-free synthetic repos and executable behavioral tests.

---

## 5. Conclusion & Verifiability

PRISM-Gemma shows that compact on-device models can overcome multi-hop localization traps, context-window thrashing, and reward starvation via principled spectral graph reasoning — and it proves each claim against a **real on-device Gemma 4**, not a simulator. Reproduce everything with:

```bash
python3 evaluate_prism_benchmark.py                 # Tables: localization, ablations (model-free, <1 min)
python3 e2e_agent_experiment.py e2e  --server <llama.cpp URL>   # Table 3.3 (real model)
python3 e2e_agent_experiment.py kv   --server <llama.cpp URL>   # Table 3.4 (measured cache_n)
python3 e2e_agent_experiment.py gprs --backend mlx              # Table 3.5 (real rollouts)
```

Raw per-run records (`e2e_results/*.jsonl`, `e2e_summary.json`) ship with the repo. Limitations: `SWE-HOP-12` is small and synthetic (public-benchmark scaling is future work); end-to-end gains are single-run greedy; GPRS is validated as a reward signal, with full GRPO training runs left to future work.

---

## References

1. Alon, N., Benjamini, I., Lubetzky, E., & Sodin, S. (2007). Non-backtracking random walks mix faster. *Communications in Contemporary Mathematics*, 9(04), 585–603.
2. Chung, F. (2007). The heat kernel as the pagerank of a graph. *PNAS*, 104(50), 19735–19740.
3. Guo, D., et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. *arXiv:2501.12948*.
4. Jeh, G., & Widom, J. (2003). Scaling personalized web search. *Proceedings of WWW*, 271–279.
5. Jimenez, C. E., et al. (2024). SWE-bench: Can Language Models Resolve Real-World GitHub Issues? *ICLR*.
6. Krzakala, F., et al. (2013). Spectral redemption in clustering sparse networks. *PNAS*, 110(52), 20935–20940.
7. Kwon, W., et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. *ACM SOSP*.
8. Ouyang, S., et al. (2024). RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph. *arXiv:2410.14684*.
9. Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. *arXiv:2402.03300*.
10. Xia, C. S., et al. (2024). Agentless: Demystifying LLM-based Software Engineering Agents. *arXiv:2407.01489*.
11. Yang, J., et al. (2024). SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. *NeurIPS*.
12. Zhang, Y., et al. (2024). AutoCodeRover: Autonomous Program Improvement. *ACM ISSTA*.
13. Zheng, L., et al. (2024). SGLang: Efficient Execution of Structured Language Model Programs. *NeurIPS*.
14. Gerganov, G., et al. llama.cpp: LLM inference in C/C++. *github.com/ggml-org/llama.cpp* (b12180).
15. Hannun, A., et al. (2023). MLX: Efficient and flexible machine learning on Apple silicon. *github.com/ml-explore/mlx*.
