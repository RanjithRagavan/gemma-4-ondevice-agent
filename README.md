# Google DeepMind — The Gemma 4 Developer Agent Competition & Paper Track

Research papers, reproducible benchmarks, open-source code-graph libraries, and ready-to-upload **Google ADK `submission.zip`** packages for **The Gemma 4 Developer Agent Competition** and **Paper Track**.

---

## Repository Entries

| # | Folder | Title / Package | Target Prize Track | Key Empirical Result / Deliverables |
| :-: | :--- | :--- | :--- | :--- |
| **01** | [`01-prism-gemma/`](01-prism-gemma/) | **PRISM-Gemma:** Non-Backtracking Spectral Code-Graph Diffusion, Topological KV-Prefix Routing, and Graph-Proximity GRPO for On-Device Gemma 4 Software Engineering Agents ([Writeup](01-prism-gemma/PAPER_WRITEUP.md)) | **Overall Best Paper ($15K)** · **Best New Resource ($10K)** (`SWE-HOP-12`) | **83.33%** E2E resolve rate (vs. 75.00% BM25) and **100%** root-cause identification with a real on-device Gemma 4 E4B, **84.95%** measured KV-cache reuse (~3× faster turns), **91.7%** informative GRPO groups in measured zero-pass regime (vs. 8.3% binary) |
| **02** | [`02-polykernel-gemma/`](02-polykernel-gemma/) | **PolyKernel-Gemma:** Polyhedral Tensor-Dataflow Code-Graphs and Roofline-Guided Speculative Fusion for Autonomous GPU/TPU Kernel Synthesis with On-Device Gemma 4 ([Writeup](02-polykernel-gemma/PAPER_WRITEUP.md)) | **Best New Application ($10K)** · **Best New Resource ($10K)** (`KERNEL-GRAPH-10`) | **23.42×** Mean (**193.01×** Peak) HBM Traffic Reduction, **4.03×** Mean (**12.00×** Peak) Roofline Speedup, **100%** SRAM-Safe Synthesis |
| **03** | [`03-kaggle-agent-submission/`](03-kaggle-agent-submission/) | **PRISM-Gemma Kaggle ADK Agent Submissions** ([Guide](03-kaggle-agent-submission/README.md)) | **Prediction Competition** — Main Leaderboard ($65K) & Small Model Bonus Prize ($10K) | Ready-to-upload, strictly validated Google ADK archives: **`dist/submission_31b.zip`** (`gemma-4-31b-it-qat-w4a16-ct`) and **`dist/submission_e4b.zip`** (`gemma-4-e4b-it-qat-w4a16-ct`), featuring context-isolated `AgentTool` sub-agents, sandboxed PRISM skill scripts, and a Kaggle L4x4 PEFT LoRA recipe |

---

## License

MIT License — see [`LICENSE`](LICENSE).
