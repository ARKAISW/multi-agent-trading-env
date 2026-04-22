"""
Evaluation utilities for comparing trained vs random agents.
"""

import numpy as np
from typing import List, Dict

from training.config import TrainingConfig
from training.train import train, run_random_baseline
from utils.visualization import (
    plot_reward_curve,
    plot_grade_progression,
    plot_comparison_table,
)


def evaluate(
    config: TrainingConfig = None,
    trained_metrics: List[Dict] = None,
    baseline_episodes: int = 10,
    df=None,
) -> Dict:
    """
    Run full evaluation: train agent, run random baseline, compare, and plot.

    Args:
        config: Training configuration (uses default if None).
        trained_metrics: Pre-computed training metrics (skips training if provided).
        baseline_episodes: Number of random baseline episodes.

    Returns:
        Evaluation results dict.
    """
    if config is None:
        config = TrainingConfig()

    # Run training if needed
    if trained_metrics is None:
        print("Running training...")
        trained_metrics = train(config, df=df)

    # Run random baseline
    print(f"\nRunning random baseline ({baseline_episodes} episodes)...")
    baseline_metrics = run_random_baseline(config, df=df, num_episodes=baseline_episodes)

    # Print comparison
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")

    def avg(metrics, key):
        return np.mean([m[key] for m in metrics])

    print(f"\n{'Metric':<20} {'Random':>12} {'Trained':>12} {'Improvement':>14}")
    print("-" * 60)

    for key, label in [
        ("total_reward", "Avg Reward"),
        ("final_grade", "Avg Grade"),
        ("pnl_pct", "Avg PnL %"),
        ("max_drawdown", "Avg Max DD"),
        ("sharpe_ratio", "Avg Sharpe"),
    ]:
        r = avg(baseline_metrics, key)
        t = avg(trained_metrics, key)
        imp = t - r
        sign = "+" if imp > 0 else ""
        print(f"  {label:<18} {r:>12.4f} {t:>12.4f} {sign}{imp:>13.4f}")

    # Generate plots
    print("\nGenerating plots...")
    plot_reward_curve(trained_metrics, baseline_metrics)
    plot_grade_progression(trained_metrics, baseline_metrics)
    plot_comparison_table(trained_metrics, baseline_metrics)

    results = {
        "trained_metrics": trained_metrics,
        "baseline_metrics": baseline_metrics,
        "trained_avg_grade": avg(trained_metrics, "final_grade"),
        "baseline_avg_grade": avg(baseline_metrics, "final_grade"),
        "grade_improvement": avg(trained_metrics, "final_grade") - avg(baseline_metrics, "final_grade"),
    }
    return results


if __name__ == "__main__":
    evaluate()
