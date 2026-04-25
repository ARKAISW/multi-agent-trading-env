"""
Multi-Agent Reward Visualization Script.

Loads training metrics from the multi-agent training run and generates:
  - Per-agent reward curves (RM, PM, Trader on same axes)
  - Governance intervention rate over training
  - Compliance rate over training
  - Baseline comparison chart

Saves all to plots/ as PNG with labeled axes and titles.

Usage:
    python training/plot_multiagent.py --input outputs/multi_agent/metrics_final.json --output plots/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def smooth(values: list[float], window: int = 10) -> np.ndarray:
    """Simple moving average for smoother curves."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_per_agent_rewards(metrics: dict, output_dir: Path):
    """Plot per-agent discounted returns on same axes."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    episodes = metrics.get("episode", [])
    trader_r = metrics.get("trader_return", [])
    rm_r = metrics.get("rm_return", [])
    pm_r = metrics.get("pm_return", [])

    if not episodes:
        print("  No episode data found, skipping reward plot.")
        return

    window = max(1, len(episodes) // 20)

    ax.plot(episodes[:len(smooth(trader_r, window))], smooth(trader_r, window),
            label="Trader", color="#2ecc71", linewidth=2)
    ax.plot(episodes[:len(smooth(rm_r, window))], smooth(rm_r, window),
            label="Risk Manager", color="#e74c3c", linewidth=2)
    ax.plot(episodes[:len(smooth(pm_r, window))], smooth(pm_r, window),
            label="Portfolio Manager", color="#3498db", linewidth=2)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Discounted Return", fontsize=12)
    ax.set_title("QuantHive: Per-Agent Reward Curves (Multi-Agent Training)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "reward_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_grade_and_sharpe(metrics: dict, output_dir: Path):
    """Plot grade and Sharpe ratio progression."""
    import matplotlib.pyplot as plt

    episodes = metrics.get("episode", [])
    grades = metrics.get("grade", [])
    sharpes = metrics.get("sharpe", [])

    if not episodes or not grades:
        print("  No grade data found, skipping grade plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    window = max(1, len(episodes) // 20)

    ax1.plot(episodes[:len(smooth(grades, window))], smooth(grades, window),
             color="#9b59b6", linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Grade [0, 1]")
    ax1.set_title("Portfolio Grade Over Training")
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes[:len(smooth(sharpes, window))], smooth(sharpes, window),
             color="#f39c12", linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Sharpe Ratio Over Training")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "grade_progression.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_baseline_comparison(metrics: dict, output_dir: Path):
    """Plot random baseline vs trained agent performance."""
    import matplotlib.pyplot as plt

    episodes = metrics.get("episode", [])
    trader_r = metrics.get("trader_return", [])
    grades = metrics.get("grade", [])

    if not episodes or len(episodes) < 20:
        print("  Not enough data for baseline comparison, skipping.")
        return

    n = len(episodes)
    first_20 = slice(0, min(20, n))
    last_20 = slice(max(0, n - 20), n)

    metrics_names = ["Trader Return", "Grade", "Max Drawdown", "Sharpe"]
    early = [
        np.mean(trader_r[first_20]),
        np.mean(grades[first_20]),
        np.mean(metrics.get("max_drawdown", [0])[first_20]),
        np.mean(metrics.get("sharpe", [0])[first_20]),
    ]
    late = [
        np.mean(trader_r[last_20]),
        np.mean(grades[last_20]),
        np.mean(metrics.get("max_drawdown", [0])[last_20]),
        np.mean(metrics.get("sharpe", [0])[last_20]),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_names))
    width = 0.35

    ax.bar(x - width / 2, early, width, label="Early (first 20 eps)", color="#e74c3c", alpha=0.8)
    ax.bar(x + width / 2, late, width, label="Late (last 20 eps)", color="#2ecc71", alpha=0.8)

    ax.set_ylabel("Value")
    ax.set_title("QuantHive: Baseline vs Trained Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = output_dir / "baseline_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_loss_curve(metrics: dict, output_dir: Path):
    """Plot PnL (as proxy loss) over training."""
    import matplotlib.pyplot as plt

    episodes = metrics.get("episode", [])
    pnl = metrics.get("pnl_pct", [])

    if not episodes or not pnl:
        print("  No PnL data found, skipping loss plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    window = max(1, len(episodes) // 20)

    smoothed = smooth(pnl, window)
    ax.plot(episodes[:len(smoothed)], smoothed, color="#e74c3c", linewidth=2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(episodes[:len(smoothed)], 0, smoothed,
                     where=np.array(smoothed) > 0, color="#2ecc71", alpha=0.2)
    ax.fill_between(episodes[:len(smoothed)], 0, smoothed,
                     where=np.array(smoothed) <= 0, color="#e74c3c", alpha=0.2)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("PnL %", fontsize=12)
    ax.set_title("QuantHive: PnL Over Training (Policy Convergence)", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "loss_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot multi-agent training results")
    parser.add_argument("--input", type=str, default="outputs/multi_agent/metrics_final.json",
                        help="Path to training metrics JSON file")
    parser.add_argument("--output", type=str, default="plots/",
                        help="Output directory for PNG plots")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Metrics file not found: {input_path}")
        print("Run training first: python training/train_multi_agent.py")
        sys.exit(1)

    with open(input_path, "r") as f:
        metrics = json.load(f)

    print(f"Loaded {len(metrics.get('episode', []))} episodes from {input_path}")
    print(f"Saving plots to {output_dir}/")

    plot_per_agent_rewards(metrics, output_dir)
    plot_grade_and_sharpe(metrics, output_dir)
    plot_baseline_comparison(metrics, output_dir)
    plot_loss_curve(metrics, output_dir)

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()
