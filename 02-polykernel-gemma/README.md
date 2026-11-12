# Entry 02 — PolyKernel-Gemma: Polyhedral Tensor-Dataflow Code-Graphs and Roofline-Guided Speculative Fusion for Autonomous GPU/TPU Kernel Synthesis with On-Device Gemma 4

* **Kaggle Writeup (`< 3,000` words):** [`PAPER_WRITEUP.md`](PAPER_WRITEUP.md)
* **Core Polyhedral Dataflow & Roofline Library:** [`polykernel_gemma.py`](polykernel_gemma.py)
* **10-Workload Kernel Benchmark (`KERNEL-GRAPH-10`):** [`evaluate_polykernel_benchmark.py`](evaluate_polykernel_benchmark.py)
* **Reproducible Empirical Output:** [`kernel_benchmark_results.json`](kernel_benchmark_results.json)

## Run Benchmark

```bash
python3 02-polykernel-gemma/evaluate_polykernel_benchmark.py
```

## Summary of Empirical Results (`KERNEL-GRAPH-10`)

| Metric | Direct AST Prompting Baseline | PolyKernel-Gemma (PT-DFG + `Roofline-Cut`) |
| :--- | :---: | :---: |
| **Mean HBM Memory Traffic Reduction** | 1.00× (Unfused Eager) | **23.42× (Peak 193.01× on FlashAttention)** |
| **Mean Arithmetic Intensity (FLOPs/Byte)** | 161.61 | **974.03** |
| **Mean Roofline Speedup** | 1.00× | **4.03× (Peak 12.00× on Softcap LogSumExp)** |
| **Hardware-Valid & SRAM-Safe Kernel Rate** | 0.0% | **100.0% (Mean 86.0% SRAM Utilization)** |
| **Max Numerical Error vs. FP64 Reference** | `NaN` (FP16 Overflow) | **0.0 (`< 1e-16`)** |
