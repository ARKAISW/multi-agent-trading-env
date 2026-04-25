"""
PettingZoo-compatible GRPO training pipeline for Qwen 2.5.

Uses MultiAgentTradingEnv-derived scenarios where the Risk Manager and
Portfolio Manager send governance messages that become part of the Trader
prompt. The Trader is then trained with Unsloth + TRL GRPOTrainer.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.reward import (
    alignment_reward_func,
    format_reward_func,
    profit_reward_func,
)
from training.grpo_verifiers_multiagent import (
    governance_reward_func_multiagent,
    risk_reward_func_multiagent,
)
from training.prompt_utils import (
    SYSTEM_PROMPT,
    build_prompt_multiagent,
    generate_pz_scenarios,
)


DEFAULT_MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
DEFAULT_OUTPUT_DIR = "models/local_policy_grpo_multiagent"


def require_cuda():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("GRPO training requires CUDA.")
    return torch


def load_model(model_name: str, max_seq_length: int):
    from unsloth import FastLanguageModel, PatchFastRL

    PatchFastRL("GRPO", "unsloth")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def make_trainer(model, tokenizer, dataset, args, torch_module):
    from trl.trainer.grpo_config import GRPOConfig
    from trl.trainer.grpo_trainer import GRPOTrainer

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=1,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        bf16=torch_module.cuda.is_bf16_supported(),
        fp16=not torch_module.cuda.is_bf16_supported(),
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        report_to="none",
    )

    reward_funcs = [
        format_reward_func,
        alignment_reward_func,
        risk_reward_func_multiagent,
        profit_reward_func,
        governance_reward_func_multiagent,
    ]

    trainer_kwargs = {
        "model": model,
        "reward_funcs": reward_funcs,
        "args": training_args,
        "train_dataset": dataset,
    }

    trainer_signature = inspect.signature(GRPOTrainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    return GRPOTrainer(**trainer_kwargs)


def save_model(model, tokenizer, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if hasattr(model, "save_pretrained_merged"):
        model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    else:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-agent GRPO training for Trader (Qwen 2.5)")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--num-scenarios", type=int, default=500)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--max-completion-length", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(
        f"Generating {args.num_scenarios} scenarios from MultiAgentTradingEnv "
        f"(difficulty={args.difficulty})..."
    )
    scenarios = generate_pz_scenarios(n=args.num_scenarios, difficulty=args.difficulty)
    print(f"  Generated {len(scenarios)} scenarios.")

    prompts = [{"prompt": build_prompt_multiagent(sc)} for sc in scenarios]
    dataset = Dataset.from_list(prompts)

    torch_module = require_cuda()
    model, tokenizer = load_model(args.model_name, args.max_seq_length)

    trainer = make_trainer(model, tokenizer, dataset, args, torch_module)
    print(f"Starting multi-agent GRPO training on {len(dataset)} prompts...")
    trainer.train()

    history = trainer.state.log_history
    rewards = [x["reward"] for x in history if "reward" in x]
    losses = [x["loss"] for x in history if "loss" in x]

    try:
        from utils.plotting import plot_training_results

        plot_training_results(rewards, losses)
    except Exception as exc:
        print(f"  Warning: could not generate plots: {exc}")

    print(f"Saving GRPO policy to {args.output_dir}...")
    save_model(model, tokenizer, args.output_dir)

    metrics_path = Path(args.output_dir) / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump({"rewards": rewards, "losses": losses}, handle, indent=2)

    print("Multi-agent GRPO training complete.")
    print(f"  Model saved to:   {args.output_dir}")
    print(f"  Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
