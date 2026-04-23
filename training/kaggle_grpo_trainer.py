import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from datasets import Dataset
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOConfig, GRPOTrainer
import re
import json

# --- 1. CONFIGURATION ---
MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
OUTPUT_DIR = "qwen-0.5b-trading-grpo"
MAX_SEQ_LENGTH = 1024

# Reward Functions from env.reward (Inlined for Kaggle portability)
def _extract_json_action(completion: str):
    match = re.search(r"<action>\s*({.*?})\s*</action>", completion, re.DOTALL)
    return json.loads(match.group(1)) if match else None

def format_reward_func(completions, **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        # Check for tags and thought length
        if all(tag in completion for tag in ["<thought>", "</thought>", "<action>", "</action>"]):
            thought = completion.split("<thought>")[1].split("</thought>")[0].strip()
            score = 1.0 if len(thought) > 100 else 0.5
            if _extract_json_action(completion): score += 0.5
            rewards.append(score / 1.5)
        else:
            rewards.append(0.0)
    return rewards

def profit_reward_func(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for prompt, completion in zip(prompts, completions):
        action = _extract_json_action(completion)
        if not action:
            rewards.append(0.0); continue
        
        # Simple directional logic: check if 'state' prices went up or down
        # (This is a proxy for the model learning to follow the trend in the prompt)
        try:
            state_match = re.search(r'"state":\s*\[(.*?)\]', prompt)
            prices = [float(x) for x in state_match.group(1).split(",")]
            is_up = prices[-1] > prices[0]
            direction = int(action.get("direction", 0))
            
            if (direction == 1 and is_up) or (direction == 2 and not is_up):
                rewards.append(1.0)
            elif direction == 0:
                rewards.append(0.3)
            else:
                rewards.append(0.0)
        except:
            rewards.append(0.0)
    return rewards

# --- 2. DATASET BUILDING ---
def get_training_data():
    # We create scenarios where the answer is somewhat clear to bootstrap learning
    scenarios = [
        {"state": [100, 102, 105], "ta": 0.8, "fa": 0.5, "limit": 1.0}, # Clear Up
        {"state": [100, 98, 95], "ta": -0.8, "fa": -0.5, "limit": 1.0}, # Clear Down
        {"state": [100, 101, 100], "ta": 0.1, "fa": -0.1, "limit": 0.5}, # Sideways
    ] * 50 # Repeat to create a decent batch
    
    prompts = []
    for s in scenarios:
        p = f"You are a Quant Trader. Action must be in <action> JSON format.\n"
        p += f"Scenario: {json.dumps(s)}\nRespond with <thought> and <action>."
        prompts.append({"prompt": p})
    return Dataset.from_list(prompts)

# --- 3. TRAINING EXECUTION ---
def main():
    PatchFastRL("GRPO", "unsloth")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        use_gradient_checkpointing="unsloth",
    )

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=100,
        num_generations=4, # How many completions to compare per prompt
        max_prompt_length=256,
        max_completion_length=256,
        logging_steps=1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, profit_reward_func],
        args=training_args,
        train_dataset=get_training_data(),
    )

    print("Starting Training... Watch 'reward/format_reward_func' for tag compliance.")
    trainer.train()
    
    model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="merged_16bit")
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
