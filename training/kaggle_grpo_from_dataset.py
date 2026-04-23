import os
import sys
import torch
import re
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOTrainer, GRPOConfig

# 1. SETUP ENVIRONMENT FROM MOUNTED DATASET
# Replace 'mate-env' with whatever you named your Kaggle Dataset
KAGGLE_DATASET_PATH = "/kaggle/input/mate-env"
if KAGGLE_DATASET_PATH not in sys.path:
    sys.path.append(KAGGLE_DATASET_PATH)

# Test Import
try:
    from env.trading_env import TradingEnv
    from env.reward import format_reward_func, profit_reward_func, risk_reward_func, alignment_reward_func
    print("Successfully imported YOUR TradingEnv from dataset!")
except ImportError as e:
    print(f"Error: Could not find your environment in {KAGGLE_DATASET_PATH}")
    print("Make sure you uploaded your zip and added it as a Dataset to this notebook.")
    raise e

# 2. Patch for GRPO
PatchFastRL("GRPO", "unsloth")

# 3. Load Model
MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=1024,
    load_in_4bit=True,
    device_map={"": 0} # Fix for Kaggle T4 x 2 to avoid cross-device 4-bit error
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 4. Dataset Generation using YOUR TradingEnv
# We'll use your env to generate realistic starting states
dummy_df = pd.DataFrame({
    "open": np.random.randn(1000)+100, 
    "high": np.random.randn(1000)+101, 
    "low": np.random.randn(1000)+99, 
    "close": np.random.randn(1000)+100, 
    "volume": np.random.rand(1000)*1000
})
env = TradingEnv(dummy_df)

SYSTEM_PROMPT = """You are a Quant Trader. Analyze State and Signals.
Respond exactly in this format:
<thought> Reasoning </thought>
<action> {"direction": 0, "size": 0.0} </action>
"""

def get_training_data():
    data = []
    for _ in range(256):
        obs, _ = env.reset()
        # Randomize starting point if your env supports it, 
        # otherwise we just train from the start of the data.
        prompt = f"{SYSTEM_PROMPT}\nScenario State: {obs.tolist()}\nSignals: {{'ta': 0.5, 'position_limit': 1.0}}\n"
        data.append({"prompt": prompt})
    return Dataset.from_list(data)

# 5. Trainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        format_reward_func,
        risk_reward_func,
        profit_reward_func,
        alignment_reward_func,
    ],
    args=GRPOConfig(
        output_dir="qwen_trading_final",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=100,
        num_generations=4,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    ),
    train_dataset=get_training_data(),
)

# 6. Train
trainer.train()
model.save_pretrained("qwen_05b_final")
tokenizer.save_pretrained("qwen_05b_final")
