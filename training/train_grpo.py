"""
GRPO training entrypoint for the local trading policy.

This script is intended for GPU-backed Hugging Face or local Linux runs where
Unsloth is available. It uses the same prompt schema as the runtime policy and
the verifier functions in env.reward.
"""

from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import inspect
import json
import random
import sys
from pathlib import Path

from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.reward import (
    alignment_reward_func,
    format_reward_func,
    profit_reward_func,
    risk_reward_func,
)
from utils.plotting import plot_training_results


DEFAULT_MODEL_NAME = "unsloth/SmolLM-135M-Instruct-bnb-4bit"
DEFAULT_OUTPUT_DIR = "models/local_policy_grpo"
DEFAULT_TRAJECTORY_PATH = "checkpoints/sft_trajectories.jsonl"

SYSTEM_PROMPT = """You are a Quant Trader operating inside a multi-agent market simulation.
Read the JSON scenario carefully and produce exactly one action.

Respond exactly in this format:
<thought>
Short reasoning about trend, fundamentals, and risk.
</thought>
<action>
{"direction": 0, "size": 0.0}
</action>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the trading policy with GRPO.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trajectory-path", default=DEFAULT_TRAJECTORY_PATH)
    parser.add_argument("--regime", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--min-grade", type=float, default=0.65)
    parser.add_argument("--max-records", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def build_prompt(state: list[float], signals: dict[str, float]) -> str:
    scenario = {
        "state": state,
        "signals": {
            "ta": float(signals["ta"]),
            "fa": float(signals["fa"]),
            "position_limit": float(signals["position_limit"]),
        },
    }
    return f"{SYSTEM_PROMPT}\nScenario:\n{json.dumps(scenario, separators=(',', ':'))}\n"


def synthetic_scenarios(regime: str) -> list[dict]:
    if regime == "easy":
        samples = [
            {"state": [1.00, 1.02, 1.04], "signals": {"ta": 0.9, "fa": 0.4, "position_limit": 1.0}},
            {"state": [0.95, 0.98, 1.01], "signals": {"ta": 0.7, "fa": 0.2, "position_limit": 0.8}},
            {"state": [1.10, 1.08, 1.05], "signals": {"ta": -0.8, "fa": -0.3, "position_limit": 0.7}},
        ]
    elif regime == "medium":
        samples = [
            {"state": [1.00, 0.99, 1.01], "signals": {"ta": 0.1, "fa": 0.0, "position_limit": 0.5}},
            {"state": [1.04, 1.01, 1.03], "signals": {"ta": -0.2, "fa": -0.1, "position_limit": 0.5}},
            {"state": [0.92, 0.96, 0.94], "signals": {"ta": -0.3, "fa": 0.2, "position_limit": 0.4}},
        ]
    else:
        samples = [
            {"state": [1.00, 1.15, 0.95], "signals": {"ta": -0.5, "fa": 0.8, "position_limit": 0.2}},
            {"state": [0.80, 0.70, 0.75], "signals": {"ta": -0.9, "fa": -0.8, "position_limit": 1.0}},
            {"state": [1.20, 1.05, 1.18], "signals": {"ta": 0.4, "fa": -0.6, "position_limit": 0.25}},
        ]
    return samples


def load_trajectory_scenarios(path: str, min_grade: float, max_records: int) -> list[dict]:
    if not os.path.exists(path):
        return []

    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("final_grade", 0.0) < min_grade:
                continue

            signal_blob = row.get("signals", {})
            records.append(
                {
                    "state": [float(x) for x in row.get("state", [])],
                    "signals": {
                        "ta": float(signal_blob.get("ta_score", 0.0)),
                        "fa": float(signal_blob.get("fa_sentiment", 0.0)),
                        "position_limit": float(signal_blob.get("position_limit", 1.0)),
                    },
                }
            )

    random.shuffle(records)
    return records[:max_records]


def build_dataset(args: argparse.Namespace) -> Dataset:
    random.seed(args.seed)

    scenarios = load_trajectory_scenarios(
        path=args.trajectory_path,
        min_grade=args.min_grade,
        max_records=args.max_records,
    )
    if not scenarios:
        scenarios = synthetic_scenarios(args.regime) * 64

    prompts = [{"prompt": build_prompt(item["state"], item["signals"])} for item in scenarios]
    return Dataset.from_list(prompts)


def require_cuda():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "GRPO training requires CUDA. Unsloth does not support CPU-only execution."
        )
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
        loftq_config=None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def make_trainer(model, tokenizer, dataset: Dataset, args: argparse.Namespace, torch_module):
    from trl import GRPOConfig, GRPOTrainer

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

    trainer_kwargs = {
        "model": model,
        "reward_funcs": [
            format_reward_func,
            alignment_reward_func,
            risk_reward_func,
            profit_reward_func,
        ],
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


def main() -> None:
    args = parse_args()
    torch_module = require_cuda()
    dataset = build_dataset(args)
    model, tokenizer = load_model(args.model_name, args.max_seq_length)

    trainer = make_trainer(model, tokenizer, dataset, args, torch_module)
    print(f"Starting GRPO training on {len(dataset)} prompts...")
    train_result = trainer.train()
    
    # Generate Plots
    metrics = train_result.metrics
    # TRL GRPOTrainer logs 'loss' and 'reward' in logs. We extract them from the history.
    history = trainer.state.log_history
    rewards = [x['reward'] for x in history if 'reward' in x]
    losses = [x['loss'] for x in history if 'loss' in x]
    plot_training_results(rewards, losses)

    print(f"Saving GRPO policy to {args.output_dir}...")
    save_model(model, tokenizer, args.output_dir)
    print("GRPO training complete.")


if __name__ == "__main__":
    main()
