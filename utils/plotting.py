import matplotlib.pyplot as plt
plt.switch_backend('Agg') # Fix for Windows MemoryError/Display issues
import pandas as pd
import numpy as np
import os

def plot_training_results(reward_history, loss_history, output_dir="plots"):
    """
    Generate professional, readable plots for the training run.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('ggplot') # Clean, modern look
    
    # 1. Reward Curve
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history, label='Agent Reward', color='#3498db', linewidth=2)
    plt.xlabel('Training Steps / Episodes')
    plt.ylabel('Normalized Reward [0, 1]')
    plt.title('Agent Performance Over Time (GRPO)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "reward_curve.png"), dpi=300)
    plt.close()
    
    # 2. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Policy Loss', color='#e74c3c', linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss Value')
    plt.title('Convergence: Policy Loss Optimization')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300)
    plt.close()
    
    print(f"Plots saved to {output_dir}")

def plot_baseline_comparison(trained_grades, random_grades, output_dir="plots"):
    """
    Compare the trained agent vs a random baseline.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('ggplot')
    
    plt.figure(figsize=(10, 6))
    plt.hist(random_grades, bins=20, alpha=0.5, label='Random Baseline', color='#95a5a6')
    plt.hist(trained_grades, bins=20, alpha=0.7, label='Trained Agent', color='#2ecc71')
    
    plt.axvline(np.mean(random_grades), color='#7f8c8d', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(trained_grades), color='#27ae60', linestyle='dashed', linewidth=2)
    
    plt.xlabel('Performance Grade [0, 1]')
    plt.ylabel('Frequency (Episodes)')
    plt.title('Performance Distribution: Baseline vs. Trained')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "baseline_comparison.png"), dpi=300)
    plt.close()
    
    print(f"Comparison plot saved to {output_dir}")
