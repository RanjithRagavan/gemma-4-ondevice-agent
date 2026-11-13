# Google DeepMind — The Gemma 4 Developer Agent Paper Track Submissions

Research papers, reproducible benchmarks, and open-source libraries advancing the state-of-the-art for **on-device agentic software engineering** with **Gemma 4** on consumer hardware.

---

## Research Entries

| # | Folder | Paper Title | Target Prize Track | Key Empirical Result |
| :-: | :--- | :--- | :--- | :--- |
| **01** | [`01-prism-gemma/`](01-prism-gemma/) | **PRISM-Gemma:** Non-Backtracking Spectral Code-Graph Diffusion, Topological KV-Prefix Routing, and Graph-Proximity GRPO for On-Device Gemma 4 Software Engineering Agents ([Writeup](01-prism-gemma/PAPER_WRITEUP.md)) | **Overall Best Paper ($15K)** · **Best New Resource ($10K)** (`SWE-HOP-12`) | **83.33%** E2E resolve rate (vs. 75.00% BM25) and **100%** root-cause identification with a real on-device Gemma 4 E4B, **84.95%** measured KV-cache reuse (~3× faster turns), **91.7%** informative GRPO groups in measured zero-pass regime (vs. 8.3% binary) |
| **02** | [`02-polykernel-gemma/`](02-polykernel-gemma/) | **PolyKernel-Gemma:** Polyhedral Tensor-Dataflow Code-Graphs and Roofline-Guided Speculative Fusion for Autonomous GPU/TPU Kernel Synthesis with On-Device Gemma 4 ([Writeup](02-polykernel-gemma/PAPER_WRITEUP.md)) | **Best New Application ($10K)** · **Best New Resource ($10K)** (`KERNEL-GRAPH-10`) | **23.42×** Mean (**193.01×** Peak) HBM Traffic Reduction, **4.03×** Mean (**12.00×** Peak) Roofline Speedup, **100%** SRAM-Safe Synthesis |

---

## License

MIT License — see [`LICENSE`](LICENSE).
