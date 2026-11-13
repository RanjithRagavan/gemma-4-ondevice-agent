# Entry 01 — PRISM-Gemma: Non-Backtracking Spectral Code-Graph Diffusion, Topological KV-Prefix Routing, and Graph-Proximity GRPO for On-Device Gemma 4 SWE Agents

* **Kaggle Writeup (`< 3,000` words):** [`PAPER_WRITEUP.md`](PAPER_WRITEUP.md)
* **Core Graph & Reward Library:** [`prism_gemma.py`](prism_gemma.py)
* **Multi-Hop SWE Benchmark & Ablation Suite (`SWE-HOP-12`):** [`evaluate_prism_benchmark.py`](evaluate_prism_benchmark.py)
* **On-Device End-to-End Experiment Harness:** [`e2e_agent_experiment.py`](e2e_agent_experiment.py)
* **Reproducible Empirical Output:** [`benchmark_results.json`](benchmark_results.json), [`e2e_results/`](e2e_results/) (raw per-run records + summary)

## Run the Model-Free Benchmark (localization + ablations, < 1 min)

```bash
cd 01-prism-gemma
python3 evaluate_prism_benchmark.py
```

## Run the On-Device Validation (real Gemma 4 E4B, Q4-class)

Requires a local llama.cpp server (`-m gemma-4-E4B_q4_0-it.gguf -c 8192`) and/or MLX (`pip install mlx-lm`, Apple silicon):

```bash
# Table 3.3 — end-to-end repair: PRISM vs BM25 context (greedy, llama.cpp)
python3 e2e_agent_experiment.py e2e --server http://127.0.0.1:8399

# Table 3.4 — measured multi-turn KV-cache prefix reuse (live cache_n accounting)
python3 e2e_agent_experiment.py kv --server http://127.0.0.1:8399

# Table 3.5 — GPRS rewards over 72 real sampled rollouts (MLX, temp 1.0)
python3 e2e_agent_experiment.py gprs --backend mlx

python3 e2e_agent_experiment.py summarize   # -> e2e_results/e2e_summary.json
```

## Headline Results (all measured on-device, Apple M2 Max)

| Result | Baseline | PRISM-Gemma |
| :--- | :---: | :---: |
| Root-Cause Recall@3 (structural, model-free) | 16.67% (BM25) / 25.00% (RepoMap PPR) | **83.33%** |
| E2E resolve rate, real Gemma 4 E4B greedy (behavioral tests) | 75.00% (BM25 ctx) | **83.33%** |
| E2E root-cause identification, real model | 91.67% | **100.00%** |
| Measured KV prefix reuse, 5-turn sessions (llama.cpp `cache_n`) | 42.56% (relevance) / 32.88% (alphabetical) | **84.95%** |
| Mean per-turn prefill latency | 369 / 416 ms | **124 ms (~3× faster)** |
| Informative GRPO groups in measured zero-pass regime (72 real rollouts) | 8.3% (binary reward) | **91.7%** |
| Kendall τ (GPRS vs. oracle rollout quality) | — | **0.796** |
