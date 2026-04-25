"""
PettingZoo-Compatible GRPO Training Pipeline for Qwen 2.5.

Uses MultiAgentTradingEnv to generate scenarios where RM and PM
send governance messages that become part of the Trader's prompt.
The Trader is trained as a Qwen 2.5-1.5B model via Unsloth + TRL GRPOTrainer.

RM and PM use rule-based policies during Trader training (alternating opt.).
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
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Dataset

from env.multi_agent_env import (
    MultiAgentTradingEnv,
    RISK_MANAGER,
    PORTFOLIO_MGR,
    TRADER,
)
from env.reward import (
    format_reward_func,
    alignment_reward_func,
    profit_reward_func,
)
from training.train_multi_agent import (
    RuleRiskManagerPolicy,
    RulePortfolioManagerPolicy,
)


# ─── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
DEFAULT_OUTPUT_DIR = "models/local_policy_grpo_multiagent"

SYSTEM_PROMPT = """You are a trading agent in a multi-agent governance system.
The Risk Manager has set governance constraints, and the Portfolio Manager has allocated capital.
Your job: propose a trade that maximizes profit while respecting these constraints.

Respond exactly in this format:
<thought>
Your reasoning about the market state, risk constraints, and trade decision.
</thought>
<action>
{"direction": 0, "size": 0.0, "sl": 0, "tp": 0}
</action>
"""


# ─── Scenario Generation from PettingZoo Env ──────────────────────────────────

def generate_pz_scenarios(
    n: int = 500,
    difficulty: str = "easy",
    max_env_steps: int = 100,
) -> List[Dict]:
    """Run the PZ env with rule policies to generate realistic scenarios.

    Each scenario captures:
      - The Trader's full observation (29 dims)
      - The RM constraints decoded from the message
      - The PM allocation decoded from the message
    """
    env = MultiAgentTradingEnv(difficulty=difficulty, max_steps=max_env_steps)
    rm_policy = RuleRiskManagerPolicy()
    pm_policy = RulePortfolioManagerPolicy()

    scenarios: List[Dict] = []
    attempts = 0
    max_attempts = n * 3

    while len(scenarios) < n and attempts < max_attempts:
        env.reset()
        attempts += 1

        step_count = 0
        while env.agents and step_count < max_env_steps:
            agent = env.agent_selection

            if agent == RISK_MANAGER:
                obs = env.observe(agent)
                action = rm_policy.act(obs)
                env.step(action)

            elif agent == PORTFOLIO_MGR:
                obs = env.observe(agent)
                action = pm_policy.act(obs)
                env.step(action)

            elif agent == TRADER:
                obs = env.observe(agent)
                # Extract RM and PM messages from the observation
                # obs layout: base(24) + rm_msg(3) + pm_msg(2) = 29
                base_obs = obs[:24].tolist()
                rm_msg = obs[24:27].tolist()  # [size_limit, allow_new, force_reduce]
                pm_msg = obs[27:29].tolist()  # [cap_alloc, override_strength]

                rm_size_limit = float(rm_msg[0])
                rm_allow_new = bool(rm_msg[1] > 0.5)
                rm_force_reduce = bool(rm_msg[2] > 0.5)
                pm_cap_alloc = float(pm_msg[0])
                pm_override = float(pm_msg[1])

                scenarios.append({
                    "state": [round(float(x), 4) for x in base_obs[:5]],
                    "full_obs": [round(float(x), 4) for x in base_obs],
                    "rm_size_limit": round(rm_size_limit, 3),
                    "rm_allow_new": rm_allow_new,
                    "rm_force_reduce": rm_force_reduce,
                    "pm_cap_alloc": round(pm_cap_alloc, 3),
                    "pm_override": round(pm_override, 3),
                    "signals": {
                        "ta": round(float(obs[5] * 2 - 1), 3),  # RSI mapped to [-1,1]
                        "fa": round(float(obs[8]), 3),  # MACD as FA proxy
                        "position_limit": round(rm_size_limit, 3),
                        "rm_size_limit": round(rm_size_limit, 3),
                    },
                })

                if len(scenarios) >= n:
                    break

                # Take a random trader action so the env advances
                trader_action = {
                    "direction": random.choice([0, 1, 2]),
                    "size": np.array([random.uniform(0.05, 0.3)], dtype=np.float32),
                    "sl": np.array([0.0], dtype=np.float32),
                    "tp": np.array([0.0], dtype=np.float32),
                }
                env.step(trader_action)

            step_count += 1

    random.shuffle(scenarios)
    return scenarios[:n]


def build_prompt_multiagent(scenario: Dict) -> str:
    """Build the prompt for the Trader, including RM and PM constraints."""
    rm_limit = scenario["rm_size_limit"]
    rm_allow_str = "allowed" if scenario.get("rm_allow_new", True) else "BLOCKED"
    rm_force_str = "yes" if scenario.get("rm_force_reduce", False) else "no"
    pm_cap = scenario["pm_cap_alloc"]
    pm_override_str = "none" if scenario.get("pm_override", 0.0) < 0.5 else "ACTIVE"

    state = scenario.get("state", [1.0, 1.0, 1.0, 1.0, 1.0])
    signals = scenario.get("signals", {})

    body = json.dumps({
        "state": state,
        "signals": signals,
        "governance": {
            "rm_size_limit": rm_limit,
            "rm_allow_new": rm_allow_str,
            "rm_force_reduce": rm_force_str,
            "pm_cap_alloc": pm_cap,
            "pm_override": pm_override_str,
        },
    }, separators=(",", ":"))

    prompt = (
        f"{SYSTEM_PROMPT}\n"
        f"The Risk Manager has set the following constraints: "
        f"size_limit={rm_limit:.2f}, new_positions={rm_allow_str}, force_reduce={rm_force_str}.\n"
        f"The Portfolio Manager allocated: capital_cap={pm_cap:.2f}, override={pm_override_str}.\n\n"
        f"Scenario:\n{body}\n"
    )
    return prompt


# ─── Updated GRPO Verifiers ───────────────────────────────────────────────────

def _extract_json_action(completion: str):
    import re
    match = re.search(r"<action>\s*({.*?})\s*</action>", completion, re.DOTALL)
    if not match:
        return None
    return json.loads(match.group(1))


def _extract_signal_value(prompt: str, key: str):
    import re
    json_match = re.search(rf'"{key}"\s*:\s*(-?[\d\.]+)', prompt)
    if json_match:
        return float(json_match.group(1))
    plain_match = re.search(rf"{key}\s*[:=]\s*(-?[\d\.]+)", prompt)
    if plain_match:
        return float(plain_match.group(1))
    return None


def risk_reward_func_multiagent(prompts, completions, **kwargs) -> list[float]:
    """Updated risk verifier: reads RM's dynamic size_limit from the prompt."""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        try:
            # Read RM's size_limit from the governance block
            limit = _extract_signal_value(prompt, "rm_size_limit")
            if limit is None:
                limit = _extract_signal_value(prompt, "position_limit")
            if limit is None:
                limit = 1.0

            data = _extract_json_action(completion)
            if data is not None:
                size = float(data.get("size", 0.0))
                score = 0.7 if size <= limit else 0.0

                try:
                    thought = completion.split("<thought>")[1].split("</thought>")[0].lower()
                    if any(kw in thought for kw in ["risk", "limit", "constraint", "size_limit"]):
                        score += 0.3
                except (IndexError, AttributeError):
                    pass
                rewards.append(score)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def governance_reward_func_multiagent(prompts, completions, **kwargs) -> list[float]:
    """Updated governance verifier: checks compliance against *learned* RM constraints.

    The key differentiator: the governance verifier now checks compliance against
    RM's size_limit from the prompt, not a hardcoded position_limit.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        try:
            data = _extract_json_action(completion)
            if data is None:
                rewards.append(0.0)
                continue

            size = float(data.get("size", 0.0))
            direction = int(data.get("direction", 0))

            # Use RM's dynamic limit
            limit = _extract_signal_value(prompt, "rm_size_limit")
            if limit is None:
                limit = _extract_signal_value(prompt, "position_limit")
            if limit is None:
                limit = 1.0

            # Also check PM cap
            pm_cap = _extract_signal_value(prompt, "pm_cap_alloc")
            effective_limit = min(limit, pm_cap) if pm_cap is not None else limit

            score = 0.0

            # Core compliance: within both RM limit and PM cap
            if size <= effective_limit:
                score += 0.40
                if 0 < size <= effective_limit * 0.8:
                    score += 0.20
            else:
                score -= 0.50

            # Reasoning quality: governance-aware language
            try:
                thought = completion.split("<thought>")[1].split("</thought>")[0].lower()
                governance_keywords = [
                    "risk", "limit", "constraint", "compliance", "conservative",
                    "governance", "restrict", "drawdown", "cap", "position limit",
                    "size_limit", "risk manager", "portfolio manager", "allocation",
                ]
                if any(kw in thought for kw in governance_keywords):
                    score += 0.20
            except (IndexError, AttributeError):
                pass

            # Activity bonus
            if direction != 0:
                score += 0.20

            rewards.append(float(np.clip(score, 0.0, 1.0)))
        except Exception:
            rewards.append(0.0)
    return rewards


# ─── Model Loading ─────────────────────────────────────────────────────────────

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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
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


# ─── Trainer ───────────────────────────────────────────────────────────────────

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
        risk_reward_func_multiagent,      # Updated: reads RM's dynamic size_limit
        profit_reward_func,
        governance_reward_func_multiagent, # Updated: checks compliance vs learned RM constraints
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


# ─── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Agent GRPO Training for Trader (Qwen 2.5)")
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

    # 1. Generate scenarios from the PettingZoo env
    print(f"Generating {args.num_scenarios} scenarios from MultiAgentTradingEnv (difficulty={args.difficulty})...")
    scenarios = generate_pz_scenarios(n=args.num_scenarios, difficulty=args.difficulty)
    print(f"  Generated {len(scenarios)} scenarios.")

    # 2. Build dataset
    prompts = [{"prompt": build_prompt_multiagent(sc)} for sc in scenarios]
    dataset = Dataset.from_list(prompts)

    # 3. Load model
    torch_module = require_cuda()
    model, tokenizer = load_model(args.model_name, args.max_seq_length)

    # 4. Train
    trainer = make_trainer(model, tokenizer, dataset, args, torch_module)
    print(f"Starting multi-agent GRPO training on {len(dataset)} prompts...")
    train_result = trainer.train()

    # 5. Generate plots
    history = trainer.state.log_history
    rewards = [x["reward"] for x in history if "reward" in x]
    losses = [x["loss"] for x in history if "loss" in x]

    try:
        from utils.plotting import plot_training_results
        plot_training_results(rewards, losses)
    except Exception as e:
        print(f"  Warning: could not generate plots: {e}")

    # 6. Save model
    print(f"Saving GRPO policy to {args.output_dir}...")
    save_model(model, tokenizer, args.output_dir)

    # 7. Save training metrics
    metrics_path = Path(args.output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"rewards": rewards, "losses": losses}, f, indent=2)

    print("Multi-agent GRPO training complete.")
    print(f"  Model saved to:   {args.output_dir}")
    print(f"  Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
