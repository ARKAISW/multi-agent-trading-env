
import os
import json
import random
import sys
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 1. Configuration
MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"
TRAJECTORY_PATH = "checkpoints/sft_trajectories.jsonl"
OUTPUT_DIR = "models/local_policy"

SYSTEM_PROMPT = """You are a Quant Trader. Analyze the scenario and return a single action.

Scenario:
{scenario}
"""

# 2. Load and Tokenize Data
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    prompt = SYSTEM_PROMPT.format(scenario=example["scenario"])
    text = (
        f"{prompt}\n"
        f"<thought>\n{example['reasoning']}\n</thought>\n"
        f"<action>\n{example['action']}\n</action>{tokenizer.eos_token}"
    )
    return tokenizer(text, truncation=True, max_length=512)

print(f"Loading data from {TRAJECTORY_PATH}...")
records = []
if os.path.exists(TRAJECTORY_PATH):
    with open(TRAJECTORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("final_grade", 0.0) >= 0.50:
                records.append({
                    "scenario": json.dumps({
                        "state": row["state"],
                        "signals": {
                            "ta": row["signals"]["ta_score"],
                            "fa": row["signals"]["fa_sentiment"],
                            "position_limit": row["signals"]["position_limit"],
                        },
                    }),
                    "action": json.dumps(row["action"]),
                    "reasoning": row["signals"].get("reasoning", {}).get(
                        "trader",
                        "Follow trend, respect the position limit, and size conservatively.",
                    ),
                })

if not records:
    print("No high-quality data found!")
    exit()

# Subset to save RAM
random.shuffle(records)
records = records[:10000] # Use top 10k samples only

dataset = Dataset.from_list(records)
tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)
print(f"Tokenized dataset ready: {len(tokenized_dataset)} samples.")

# 3. Load Model
print("Loading model to CPU...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    dtype=torch.float32, 
    device_map="cpu"
)

# 4. Train
print("Starting CPU Training (Lighter on RAM)...")
training_args = TrainingArguments(
    output_dir="outputs",
    max_steps=100, # Faster for CPU
    per_device_train_batch_size=1, # Lowest RAM usage
    gradient_accumulation_steps=8, # Maintain effective batch size of 8
    learning_rate=1e-4,
    logging_steps=10,
    save_strategy="no",
    use_cpu=True,
    report_to="none"
)

# Standard Trainer (skipping SFTTrainer specific helper args)
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()

# 5. Save
print(f"Saving fine-tuned model to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done! Your model is graduated.")
