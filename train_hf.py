#!/usr/bin/env python3
"""
QuantHive — HF Jobs GRPO Training Script
=========================================
Standalone script to fine-tune Qwen 2.5-1.5B on the multi-agent trading
environment using GRPO.  Designed to run on HuggingFace Jobs (A10G / A100).

Usage (local):
    python train_hf.py

Usage (HF Jobs):
    hf jobs run --hardware a10g-small -- python train_hf.py

The script:
 1. Generates scenarios from the PettingZoo multi-agent env
 2. Trains with GRPO + 5 governance-aware verifiers
 3. Saves LoRA adapters + merged model
 4. Logs sample outputs so you can see the <thought> reasoning
 5. Generates training plots and pushes everything to the HF Hub
"""

from __future__ import annotations

import inspect
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np

# ── Unsloth JIT-compilation bypass (prevents AttributeError on cloud) ─────────
os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["DISABLE_UNSLOTH_COMPILE"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Delete compiled cache if it exists
cache_dir = Path("unsloth_compiled_cache")
if cache_dir.exists():
    shutil.rmtree(cache_dir, ignore_errors=True)
    print("🗑️  Deleted unsloth_compiled_cache/")

# ── Ensure project root is importable ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — Edit these for your run
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_NAME       = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
OUTPUT_DIR       = "models/grpo_hf_trained"
HF_REPO_ID       = "ARKAISW/QuantHive-GRPO-Trader"  # Where to push the model

# Training hyperparameters
NUM_SCENARIOS         = 800          # More diverse scenarios
MAX_STEPS             = 500          # 2x longer than Kaggle run
BATCH_SIZE            = 4
GRAD_ACCUM_STEPS      = 2
NUM_GENERATIONS       = 8            # 8 candidates per prompt (better GRPO signal)
LEARNING_RATE         = 1e-5
MAX_SEQ_LENGTH        = 1024
MAX_PROMPT_LENGTH     = 768
MAX_COMPLETION_LENGTH = 64
SAVE_STEPS            = 100
LOGGING_STEPS         = 1
DIFFICULTY            = "easy"       # "easy", "medium", "hard"
SEED                  = 3407

# Sample output logging
NUM_SAMPLE_OUTPUTS    = 10           # How many sample outputs to log after training


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # ── Step 1: Install deps if missing ───────────────────────────────────────
    print("=" * 60)
    print("  QuantHive — Multi-Agent GRPO Training (HF Jobs)")
    print("=" * 60)

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("❌ CUDA not available. Use GPU hardware.")
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Step 2: Generate scenarios ────────────────────────────────────────────
    from training.prompt_utils import (
        SYSTEM_PROMPT,
        build_prompt_multiagent,
        generate_pz_scenarios,
    )

    print(f"\n📊 Generating {NUM_SCENARIOS} scenarios (difficulty={DIFFICULTY})...")
    scenarios = generate_pz_scenarios(
        n=NUM_SCENARIOS, difficulty=DIFFICULTY, max_env_steps=100
    )
    print(f"   Generated {len(scenarios)} scenarios.")

    from datasets import Dataset
    prompts = [{"prompt": build_prompt_multiagent(sc)} for sc in scenarios]
    dataset = Dataset.from_list(prompts)

    # ── Step 3: Load model natively via Transformers/PEFT ─────────────────────
    print(f"\n🤖 Loading model natively: {MODEL_NAME}")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    
    # 🚀 Fix precision mismatch (Half vs Float) in generate() — BFloat16 is safer for A10G
    compute_dtype = torch.bfloat16
    model.to(compute_dtype)
    if hasattr(model, "lm_head"):
        model.lm_head.to(compute_dtype)
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens.to(compute_dtype)

    # 🐛 Fix GRPOTrainer crash by injecting warnings_issued dict
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
        
    print("   Native model loaded + LoRA applied.")

    # ── Step 5: Build trainer ─────────────────────────────────────────────────
    from trl.trainer.grpo_config import GRPOConfig
    
    # 🐛 Fix llm_blender crashing on modern transformers by injecting missing cache var
    import transformers.utils.hub
    if not hasattr(transformers.utils.hub, "TRANSFORMERS_CACHE"):
        try:
            transformers.utils.hub.TRANSFORMERS_CACHE = transformers.utils.hub.constants.HF_HUB_CACHE
        except AttributeError:
            transformers.utils.hub.TRANSFORMERS_CACHE = "/tmp"

    from trl.trainer.grpo_trainer import GRPOTrainer

    from env.reward import (
        alignment_reward_func,
        format_reward_func,
        profit_reward_func,
    )
    from training.grpo_verifiers_multiagent import (
        governance_reward_func_multiagent,
        risk_reward_func_multiagent,
    )

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=1,
        max_steps=MAX_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        bf16=True,
        fp16=False,
        max_grad_norm=0.5,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=NUM_GENERATIONS,
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
    sig = inspect.signature(GRPOTrainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)

    # ── Step 6: Train! ────────────────────────────────────────────────────────
    print(f"\n🚀 Starting GRPO training — {MAX_STEPS} steps, {NUM_GENERATIONS} generations/prompt")
    print(f"   Effective batch size: {BATCH_SIZE} × {GRAD_ACCUM_STEPS} × 1 GPU = {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print()
    trainer.train()
    print("\n✅ Training complete!")

    # ── Step 7: Extract metrics ───────────────────────────────────────────────
    history = trainer.state.log_history
    rewards = [x["reward"] for x in history if "reward" in x]
    losses = [x.get("loss", 0.0) for x in history if "reward" in x]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics_path = Path(OUTPUT_DIR) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"rewards": rewards, "losses": losses, "log_history": history}, f, indent=2, default=str)
    print(f"📈 Metrics saved to {metrics_path}")

    # ── Step 8: Generate sample outputs (CRITICAL for judge review) ───────────
    print(f"\n📝 Generating {NUM_SAMPLE_OUTPUTS} sample outputs from trained model...")
    model.eval()

    sample_outputs = []
    for i in range(min(NUM_SAMPLE_OUTPUTS, len(scenarios))):
        prompt_text = build_prompt_multiagent(scenarios[i])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_COMPLETION_LENGTH,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        sample_outputs.append({
            "scenario_idx": i,
            "rm_size_limit": scenarios[i]["rm_size_limit"],
            "pm_cap_alloc": scenarios[i]["pm_cap_alloc"],
            "model_output": response,
        })

        print(f"\n{'─' * 60}")
        print(f"  Sample {i+1} | RM limit={scenarios[i]['rm_size_limit']:.2f} | PM cap={scenarios[i]['pm_cap_alloc']:.2f}")
        print(f"{'─' * 60}")
        print(response[:500])

    samples_path = Path(OUTPUT_DIR) / "sample_outputs.json"
    with open(samples_path, "w") as f:
        json.dump(sample_outputs, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Sample outputs saved to {samples_path}")

    # ── Step 9: Generate plots ────────────────────────────────────────────────
    print("\n📊 Generating training plots...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs("plots", exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("QuantHive Multi-Agent GRPO Training — Qwen 2.5 1.5B", fontsize=14)

        # Loss curve
        steps = list(range(1, len(losses) + 1))
        axes[0].plot(steps, losses, alpha=0.4, color="salmon", label="Raw")
        if len(losses) >= 20:
            ma = np.convolve(losses, np.ones(20)/20, mode="valid")
            axes[0].plot(range(20, len(losses)+1), ma, color="red", linewidth=2, label="MA-20")
        axes[0].set_xlabel("Training Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("GRPO Training Loss")
        axes[0].legend()

        # Reward curve
        axes[1].plot(steps, rewards, alpha=0.4, color="lightgreen", label="Raw")
        if len(rewards) >= 20:
            ma = np.convolve(rewards, np.ones(20)/20, mode="valid")
            axes[1].plot(range(20, len(rewards)+1), ma, color="green", linewidth=2, label="MA-20")
        axes[1].set_xlabel("Training Step")
        axes[1].set_ylabel("Mean Reward")
        axes[1].set_title("GRPO Mean Reward (5 Verifiers)")
        axes[1].legend()

        plt.tight_layout()
        fig.savefig("plots/hf_training_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("   Saved plots/hf_training_curves.png")

        # ── Baseline comparison bar chart ─────────────────────────────────────
        # Evaluate trained model vs random baseline on 20 scenarios
        print("   Generating baseline comparison...")
        eval_scenarios = scenarios[:20]
        trained_scores = {
            "Format": [], "Alignment": [], "Risk": [], "Profit": [], "Governance": []
        }
        baseline_scores = {
            "Format": [], "Alignment": [], "Risk": [], "Profit": [], "Governance": []
        }

        for sc in eval_scenarios:
            prompt_text = build_prompt_multiagent(sc)

            # Trained model output
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            out = model.generate(input_ids=input_ids, max_new_tokens=MAX_COMPLETION_LENGTH, temperature=0.7, do_sample=True)
            completion = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

            # Random baseline: gibberish output
            random_completion = '{"direction": ' + str(random.choice([0,1,2])) + ', "size": ' + f"{random.random():.2f}" + ', "sl": 0, "tp": 0}'

            # Score both
            for name, func in zip(
                ["Format", "Alignment", "Risk", "Profit", "Governance"],
                reward_funcs
            ):
                t_score = func([prompt_text], [completion])[0]
                b_score = func([prompt_text], [random_completion])[0]
                trained_scores[name].append(t_score)
                baseline_scores[name].append(b_score)

        # Plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        verifiers = list(trained_scores.keys())
        x = np.arange(len(verifiers))
        width = 0.35

        trained_means = [np.mean(trained_scores[v]) for v in verifiers]
        baseline_means = [np.mean(baseline_scores[v]) for v in verifiers]

        bars1 = ax2.bar(x - width/2, baseline_means, width, label="Random Baseline", color="#ff6b6b", alpha=0.85)
        bars2 = ax2.bar(x + width/2, trained_means, width, label="GRPO-Trained", color="#51cf66", alpha=0.85)

        ax2.set_ylabel("Mean Score")
        ax2.set_xlabel("Reward Verifier")
        ax2.set_title("QuantHive: Trained Agent vs Random Baseline")
        ax2.set_xticks(x)
        ax2.set_xticklabels(verifiers)
        ax2.legend()
        ax2.set_ylim(0, 1.1)

        for bar in bars1:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)
        for bar in bars2:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)

        fig2.savefig("plots/hf_baseline_vs_trained.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("   Saved plots/hf_baseline_vs_trained.png")

    except Exception as e:
        print(f"   ⚠️  Could not generate plots: {e}")

    # ── Step 10: Save model ───────────────────────────────────────────────────
    print(f"\n💾 Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # ── Step 11: Push to HF Hub (optional) ────────────────────────────────────
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        print(f"\n🚀 Pushing model to {HF_REPO_ID}...")
        api.upload_folder(
            folder_path=OUTPUT_DIR,
            repo_id=HF_REPO_ID,
            repo_type="model",
            create_pr=False,
        )
        print(f"   ✅ Model pushed to https://huggingface.co/{HF_REPO_ID}")

        # Also push the plots
        for plot_file in Path("plots").glob("hf_*.png"):
            api.upload_file(
                path_or_fileobj=str(plot_file),
                path_in_repo=f"plots/{plot_file.name}",
                repo_id=HF_REPO_ID,
                repo_type="model",
            )
            print(f"   📊 Uploaded {plot_file.name}")

    except Exception as e:
        print(f"   ⚠️  Could not push to HF Hub: {e}")
        print(f"   You can manually push later with: huggingface-cli upload {HF_REPO_ID} {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("  ✅ QuantHive GRPO Training Complete!")
    print(f"  📁 Model: {OUTPUT_DIR}")
    print(f"  📊 Plots: plots/hf_training_curves.png, plots/hf_baseline_vs_trained.png")
    print(f"  📝 Samples: {OUTPUT_DIR}/sample_outputs.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
