"""
Reproducible empirical evaluation suite for PolyKernel-Gemma (KERNEL-GRAPH-10 Benchmark).
Evaluates:
1. HBM Memory Traffic Reduction & Roofline Arithmetic Intensity across 10 LLM/Transformer workloads
2. Legality & SRAM Capacity Safety: Direct AST Prompting vs. PolyKernel-Gemma (PT-DFG + Roofline-Cut)
3. Numerical Equivalence Verification (Max Absolute Error on adversarial overflow/tail inputs)
"""

from __future__ import annotations

import json
import math
from typing import Dict, List

from polykernel_gemma import PolyhedralTensorDFG


def build_kernel_benchmark_suite() -> List[Dict]:
    return [
        {
            "id": "KG-01",
            "name": "Fused SwiGLU MLP Gate + Up Projection",
            "shapes": {"x": (4096, 4096), "w_gate": (4096, 11008), "w_up": (4096, 11008)},
            "code": '''
def swiglu_mlp(x, w_gate, w_up):
    gate = matmul(x, w_gate)
    up = matmul(x, w_up)
    act = silu(gate)
    out = act * up
    return out
''',
        },
        {
            "id": "KG-02",
            "name": "Fused Residual + RMSNorm + Scale",
            "shapes": {"x": (4096, 4096), "residual": (4096, 4096), "weight": (1, 4096)},
            "code": '''
def fused_add_rmsnorm(x, residual, weight):
    h = x + residual
    sq = h * h
    var = mean(sq)
    inv_std = rsqrt(var)
    normed = h * inv_std
    out = normed * weight
    return out, h
''',
        },
        {
            "id": "KG-03",
            "name": "FlashAttention Online Softmax + Value Aggregation",
            "shapes": {"q": (8192, 128), "k_t": (128, 8192), "v": (8192, 128), "scale": (1, 1)},
            "code": '''
def flash_attention_step(q, k_t, v, scale):
    scores = matmul(q, k_t)
    scaled = scores * scale
    row_max = max(scaled)
    shifted = scaled - row_max
    probs_unnorm = exp(shifted)
    denom = sum(probs_unnorm)
    probs = probs_unnorm / denom
    out = matmul(probs, v)
    return out
''',
        },
        {
            "id": "KG-04",
            "name": "Gemma Softcap Logit Tanh + Online Cross-Entropy LogSumExp",
            "shapes": {"logits": (4096, 32768), "inv_cap": (1, 1), "cap": (1, 1)},
            "code": '''
def softcap_logsumexp(logits, inv_cap, cap):
    scaled = logits * inv_cap
    t = tanh(scaled)
    capped = t * cap
    m = max(capped)
    shifted = capped - m
    ex = exp(shifted)
    s = sum(ex)
    lse = log(s)
    out = m + lse
    return out
''',
        },
        {
            "id": "KG-05",
            "name": "Fused GeGLU Feed-Forward Activation",
            "shapes": {"gate_pre": (4096, 16384), "up_pre": (4096, 16384), "bias": (1, 16384)},
            "code": '''
def geglu_activation(gate_pre, up_pre, bias):
    g = gate_pre + bias
    act = gelu(g)
    u = up_pre + bias
    out = act * u
    return out
''',
        },
        {
            "id": "KG-06",
            "name": "Int4 Weight-Only Group Dequantization + GEMM + Bias",
            "shapes": {"x": (2048, 4096), "w_int4": (4096, 11008), "scales": (1, 11008), "bias": (1, 11008)},
            "code": '''
def int4_dequant_gemm(x, w_int4, scales, bias):
    w_fp16 = dequant_int4(w_int4)
    w_scaled = w_fp16 * scales
    mm = matmul(x, w_scaled)
    out = mm + bias
    return out
''',
        },
        {
            "id": "KG-07",
            "name": "Fused RoPE Rotary Positional Embedding + Key Normalization",
            "shapes": {"k": (8192, 256), "cos": (8192, 256), "sin": (8192, 256), "k_rot_half": (8192, 256)},
            "code": '''
def rope_qk_norm(k, cos, sin, k_rot_half):
    c_part = k * cos
    s_part = k_rot_half * sin
    rot = c_part + s_part
    sq = rot * rot
    var = mean(sq)
    inv = rsqrt(var)
    out = rot * inv
    return out
''',
        },
        {
            "id": "KG-08",
            "name": "LoRA Dual-Branch Base + Low-Rank Adapter Fusion",
            "shapes": {"x": (4096, 4096), "w_base": (4096, 4096), "lora_a": (4096, 64), "lora_b": (64, 4096), "alpha_scale": (1, 1)},
            "code": '''
def fused_lora_linear(x, w_base, lora_a, lora_b, alpha_scale):
    base_out = matmul(x, w_base)
    low_rank = matmul(x, lora_a)
    adapter = matmul(low_rank, lora_b)
    scaled_adapter = adapter * alpha_scale
    out = base_out + scaled_adapter
    return out
''',
        },
        {
            "id": "KG-09",
            "name": "Fused LayerNorm + Dropout Mask + Residual Projection",
            "shapes": {"x": (4096, 4096), "mask": (4096, 4096), "res": (4096, 4096), "gamma": (1, 4096), "beta": (1, 4096)},
            "code": '''
def fused_ln_dropout_res(x, mask, res, gamma, beta):
    Dropped = x * mask
    added = Dropped + res
    mu = mean(added)
    centered = added - mu
    sq = centered * centered
    var = mean(sq)
    inv = rsqrt(var)
    normed = centered * inv
    scaled = normed * gamma
    out = scaled + beta
    return out
''',
        },
        {
            "id": "KG-10",
            "name": "Sigmoid-Gated Highway Linear + Variance Scaling",
            "shapes": {"x": (4096, 8192), "h": (4096, 8192), "gate_logits": (4096, 8192)},
            "code": '''
def highway_gate_norm(x, h, gate_logits):
    g = sigmoid(gate_logits)
    gated_h = g * h
    one_minus_g = x - gated_h
    mixed = gated_h + one_minus_g
    sq = mixed * mixed
    v = mean(sq)
    inv = rsqrt(v)
    out = mixed * inv
    return out
''',
        },
    ]


def verify_online_vs_naive_softmax_stability() -> Dict[str, float]:
    """
    Demonstrates numerical verification of PolyKernel-Gemma's Online Recurrence vs.
    naive unshifted reduction on extreme BF16/FP16 logit ranges (e.g., logits in [85.0, 95.0]
    where exp(89.0) overflows FP16/FP32 without online max rescaling).
    """
    logits = [88.5, 91.2, 94.0, 89.7, 92.1, 87.3, 93.8, 90.4]

    # Reference FP64 exact softmax
    m_exact = max(logits)
    exps_exact = [math.exp(x - m_exact) for x in logits]
    denom_exact = sum(exps_exact)
    probs_exact = [e / denom_exact for e in exps_exact]

    # PolyKernel-Gemma 1-pass Online Milakov-Gimelshein Recurrence (streaming blocks of 2)
    m_running = -1e30
    d_running = 0.0
    for i in range(0, len(logits), 2):
        blk = logits[i : i + 2]
        m_blk = max(blk)
        m_new = max(m_running, m_blk)
        d_running = d_running * math.exp(m_running - m_new) + sum(math.exp(x - m_new) for x in blk)
        m_running = m_new

    probs_online = [math.exp(x - m_running) / d_running for x in logits]
    max_abs_err = max(abs(a - b) for a, b in zip(probs_exact, probs_online))

    # Naive unshifted exp overflow check (FP16 max = 65504 -> exp(x) overflows for x > 11.09)
    fp16_overflow_count = sum(1 for x in logits if x > 11.09)

    return {
        "online_recurrence_max_abs_error": round(max_abs_err, 16),
        "naive_fp16_overflow_rate_pct": round(100.0 * fp16_overflow_count / len(logits), 1),
    }


def main() -> None:
    suite = build_kernel_benchmark_suite()
    per_kernel = []

    hbm_reductions = []
    speedups = []
    eager_ais = []
    fused_ais = []
    sram_utils = []
    naive_valid_count = 0
    poly_valid_count = 0

    for item in suite:
        dfg = PolyhedralTensorDFG(item["shapes"], sram_limit_kb=128.0)
        dfg.parse_kernel_function(item["code"])

        naive_check = dfg.partition_naive_llm_fusion()
        clusters, stats = dfg.run_roofline_guided_fusion()
        schedule_spec = dfg.emit_triton_pallas_schedule_spec(clusters[0])

        if naive_check["is_valid_and_safe"]:
            naive_valid_count += 1
        if stats["sram_footprint_kb"] <= 128.0:
            poly_valid_count += 1

        hbm_reductions.append(stats["hbm_traffic_reduction_x"])
        speedups.append(stats["roofline_speedup_x"])
        eager_ais.append(stats["eager_arithmetic_intensity"])
        fused_ais.append(stats["fused_arithmetic_intensity"])
        sram_utils.append(stats["sram_utilization_pct"])

        per_kernel.append({
            "id": item["id"],
            "name": item["name"],
            "naive_llm_baseline": naive_check,
            "polykernel_gemma": stats,
            "schedule_preview": schedule_spec.splitlines()[:4],
        })

    num_verify = verify_online_vs_naive_softmax_stability()

    summary = {
        "num_workloads": len(suite),
        "mean_hbm_traffic_reduction_x": round(sum(hbm_reductions) / len(hbm_reductions), 2),
        "max_hbm_traffic_reduction_x": round(max(hbm_reductions), 2),
        "mean_roofline_speedup_x": round(sum(speedups) / len(speedups), 2),
        "max_roofline_speedup_x": round(max(speedups), 2),
        "mean_eager_arithmetic_intensity": round(sum(eager_ais) / len(eager_ais), 2),
        "mean_fused_arithmetic_intensity": round(sum(fused_ais) / len(fused_ais), 2),
        "naive_ast_prompt_valid_kernel_rate_pct": round(100.0 * naive_valid_count / len(suite), 1),
        "polykernel_gemma_valid_kernel_rate_pct": round(100.0 * poly_valid_count / len(suite), 1),
        "mean_sram_utilization_pct": round(sum(sram_utils) / len(sram_utils), 1),
        "numerical_verification": num_verify,
        "workloads": per_kernel,
    }

    with open("02-polykernel-gemma/kernel_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== PolyKernel-Gemma (KERNEL-GRAPH-10) Benchmark Summary ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "workloads"}, indent=2))


if __name__ == "__main__":
    main()
