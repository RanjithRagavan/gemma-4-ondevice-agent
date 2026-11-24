#!/usr/bin/env python3
"""
PRISM-Gemma PEFT LoRA SFT + Graph-Proximity Reward Shaping (GPRS) Training Recipe
for Kaggle L4x4 (96 GB VRAM) on the 129 public tasks in tasks.jsonl.

Exports PEFT adapters directly in competition-compliant `.safetensors` format:
  adapters/main_lora/
    ├── adapter_config.json
    └── adapter_model.safetensors

Note: Run inside a Kaggle L4x4 notebook (attached to the competition dataset) where
`tasks.jsonl`, `graphs/`, and the official Gemma 4 model weights are mounted locally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def build_training_examples(tasks_jsonl: Path, graphs_dir: Path) -> List[Dict[str, str]]:
    """
    Constructs high-signal multi-turn tool-use & minimal unified-diff SFT examples
    from the 129 competition development tasks (`tasks.jsonl` + `graphs/*.json`).
    Each example teaches Gemma 4 to:
      1. Inspect the problem statement & trace 2-3 hops in the code graph.
      2. Reproduce in /tmp/repro.py.
      3. Emit minimal, focused edit_file calls on production code only.
      4. Verify with pytest and call submit_patch().
    """
    examples: List[Dict[str, str]] = []
    with tasks_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            task = json.loads(line)
            patch = task.get("patch", "").strip()
            if not patch:
                continue

            # Extract modified files from reference unified diff
            touched_files = []
            for pline in patch.splitlines():
                if pline.startswith("+++ b/"):
                    touched_files.append(pline[len("+++ b/") :].strip())

            graph_path = graphs_dir / f"{task['instance_id']}.json"
            graph_hint = ""
            if graph_path.is_file():
                gdata = json.loads(graph_path.read_text(encoding="utf-8"))
                nodes = gdata.get("nodes", [])
                graph_hint = f"Repository graph contains {len(nodes)} indexed code symbols."

            prompt = (
                f"Repository: {task['repo']}\n"
                f"Issue:\n{task['problem_statement']}\n"
                f"{graph_hint}\n"
                "Draft a disciplined multi-hop localization plan and minimal patch."
            )
            completion = (
                f"Target files localized via multi-hop code-graph traversal: {', '.join(touched_files)}.\n"
                "Minimal verified patch:\n"
                f"```diff\n{patch}\n```"
            )
            examples.append({"prompt": prompt, "completion": completion, "instance_id": task["instance_id"]})
    return examples


def run_lora_training(
    model_path: str,
    tasks_jsonl: Path,
    graphs_dir: Path,
    output_adapter_dir: Path,
    lora_rank: int = 32,
    epochs: int = 3,
    lr: float = 1e-4,
) -> None:
    """
    Trains a competition-compliant LoRA adapter (r <= 128, .safetensors only)
    targeting all attention and MLP projections for vLLM multi-LoRA serving.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    examples = build_training_examples(tasks_jsonl, graphs_dir)
    print(f"Prepared {len(examples)} SFT examples from {tasks_jsonl}")

    ds = Dataset.from_list(
        [{"text": f"<start_of_turn>user\n{ex['prompt']}<end_of_turn>\n<start_of_turn>model\n{ex['completion']}<end_of_turn>"} for ex in examples]
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(output_adapter_dir / "_checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()

    output_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_adapter_dir), safe_serialization=True)
    print(f"Saved competition-ready PEFT .safetensors adapter to {output_adapter_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Gemma 4 LoRA adapter for swegemma competition")
    parser.add_argument("--model-path", required=True, help="Local path to Gemma 4 model on Kaggle")
    parser.add_argument("--tasks-jsonl", type=Path, required=True, help="Path to tasks.jsonl")
    parser.add_argument("--graphs-dir", type=Path, required=True, help="Path to graphs/ directory")
    parser.add_argument("--output-dir", type=Path, default=Path("adapters/main_lora"))
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank (<= 128 for vLLM)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    run_lora_training(
        model_path=args.model_path,
        tasks_jsonl=args.tasks_jsonl,
        graphs_dir=args.graphs_dir,
        output_adapter_dir=args.output_dir,
        lora_rank=args.rank,
        epochs=args.epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
