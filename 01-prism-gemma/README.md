# Entry 01 — PRISM-Gemma: Non-Backtracking Spectral Code-Graph Diffusion, Topological KV-Prefix Routing, and Graph-Proximity GRPO for On-Device Gemma 4 SWE Agents

* **Kaggle Writeup (`< 3,000` words):** [`PAPER_WRITEUP.md`](PAPER_WRITEUP.md)
* **Core Graph & Reward Library:** [`prism_gemma.py`](prism_gemma.py)
* **Multi-Hop SWE Benchmark & Ablation Suite (`SWE-HOP-12`):** [`evaluate_prism_benchmark.py`](evaluate_prism_benchmark.py)
* **Reproducible Empirical Output:** [`benchmark_results.json`](benchmark_results.json)

## Run Benchmark

```bash
cd 01-prism-gemma
python3 evaluate_prism_benchmark.py
```

## Summary of Empirical Results (`SWE-HOP-12`)

| Method / Variant | Recall@1 | Recall@3 | Recall@5 | MRR | Root-Cause Recall@3 | Mean Root-Cause Rank ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| BM25 / Lexical `grep` | 0.0000 | 0.3333 | 0.7083 | 0.3507 | 16.67% | 4.75 |
| Dense Token Cosine RAG | 0.0000 | 0.2917 | 0.6250 | 0.2861 | 25.00% | 5.17 |
| Uniform RepoMap (Undirected PageRank) | 0.0000 | 0.2917 | 0.6667 | 0.2786 | 25.00% | 5.17 |
| **PRISM-Gemma (Full NBD-PPR, $K=3$)** | **0.3750** | **0.7917** | **0.9167** | **0.8111** | **83.33%** | **2.75** |
