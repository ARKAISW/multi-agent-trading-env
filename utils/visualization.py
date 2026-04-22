"""
Visualization utilities for plotting training results.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")     # Non-interactive backend for scripts
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import os


PLOT_DIR = "plots"


def _ensure_plot_dir(save_dir: str = PLOT_DIR):
    os.makedirs(save_dir, exist_ok=True)


def plot_equity_curve(
    episode_values: List[float],
    title: str = "Equity Curve",
    save_path: Optional[str] = None,
):
    """Plot portfolio value over time within an episode."""
    _ensure_plot_dir()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(episode_values, color="#2196F3", linewidth=1.5)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Step")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True, alpha=0.3)
    ax.fill_between(range(len(episode_values)), episode_values,
                    alpha=0.1, color="#2196F3")
    plt.tight_layout()
    path = save_path or os.path.join(PLOT_DIR, "equity_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_drawdown(
    episode_values: List[float],
    title: str = "Drawdown Chart",
    save_path: Optional[str] = None,
):
    """Plot drawdown over time within an episode."""
    _ensure_plot_dir()
    values = np.array(episode_values)
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / (peak + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(range(len(drawdown)), drawdown, alpha=0.4, color="#F44336")
    ax.plot(drawdown, color="#F44336", linewidth=1)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Step")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    path = save_path or os.path.join(PLOT_DIR, "drawdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_reward_curve(
    metrics: List[Dict],
    baseline_metrics: Optional[List[Dict]] = None,
    title: str = "Reward Curve Across Episodes",
    save_path: Optional[str] = None,
):
    """Plot total reward per episode across training, optionally with baseline."""
    _ensure_plot_dir()
    rewards = [m["total_reward"] for m in metrics]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rewards, color="#4CAF50", linewidth=1.5, label="Trained Agent", alpha=0.8)

    # Smoothed trend
    if len(rewards) > 5:
        window = max(5, len(rewards) // 10)
        smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
        ax.plot(smoothed, color="#2E7D32", linewidth=2.5, label="Trend (smoothed)")

    # Baseline
    if baseline_metrics:
        bl_rewards = [m["total_reward"] for m in baseline_metrics]
        bl_mean = np.mean(bl_rewards)
        ax.axhline(y=bl_mean, color="#FF5722", linestyle="--", linewidth=2,
                    label=f"Random Baseline (avg={bl_mean:.3f})")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = save_path or os.path.join(PLOT_DIR, "reward_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_grade_progression(
    metrics: List[Dict],
    baseline_metrics: Optional[List[Dict]] = None,
    title: str = "Grade Progression (0 → 1)",
    save_path: Optional[str] = None,
):
    """Plot grade progression across episodes."""
    _ensure_plot_dir()
    grades = [m["final_grade"] for m in metrics]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(grades, color="#9C27B0", linewidth=1.5, label="Trained Agent", alpha=0.8)

    if len(grades) > 5:
        window = max(5, len(grades) // 10)
        smoothed = pd.Series(grades).rolling(window=window, min_periods=1).mean()
        ax.plot(smoothed, color="#6A1B9A", linewidth=2.5, label="Trend (smoothed)")

    if baseline_metrics:
        bl_grades = [m["final_grade"] for m in baseline_metrics]
        bl_mean = np.mean(bl_grades)
        ax.axhline(y=bl_mean, color="#FF5722", linestyle="--", linewidth=2,
                    label=f"Random Baseline (avg={bl_mean:.3f})")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Grade [0, 1]")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = save_path or os.path.join(PLOT_DIR, "grade_progression.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_comparison_table(
    trained_metrics: List[Dict],
    baseline_metrics: List[Dict],
    save_path: Optional[str] = None,
):
    """Create a comparison table figure: random agent vs trained agent."""
    _ensure_plot_dir()

    def avg(metrics, key):
        return np.mean([m[key] for m in metrics])

    data = {
        "Metric": ["Avg Reward", "Avg Grade", "Avg PnL %", "Avg Max DD", "Avg Sharpe"],
        "Random Agent": [
            f"{avg(baseline_metrics, 'total_reward'):.3f}",
            f"{avg(baseline_metrics, 'final_grade'):.3f}",
            f"{avg(baseline_metrics, 'pnl_pct'):.2%}",
            f"{avg(baseline_metrics, 'max_drawdown'):.3f}",
            f"{avg(baseline_metrics, 'sharpe_ratio'):.3f}",
        ],
        "Trained Agent": [
            f"{avg(trained_metrics, 'total_reward'):.3f}",
            f"{avg(trained_metrics, 'final_grade'):.3f}",
            f"{avg(trained_metrics, 'pnl_pct'):.2%}",
            f"{avg(trained_metrics, 'max_drawdown'):.3f}",
            f"{avg(trained_metrics, 'sharpe_ratio'):.3f}",
        ],
    }

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    table = ax.table(
        cellText=list(zip(data["Metric"], data["Random Agent"], data["Trained Agent"])),
        colLabels=["Metric", "Random Agent", "Trained Agent"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(3):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Random vs Trained Agent Comparison", fontsize=14, pad=20)
    plt.tight_layout()
    path = save_path or os.path.join(PLOT_DIR, "comparison_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path
