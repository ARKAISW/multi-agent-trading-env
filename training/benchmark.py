import os
import sys
from pathlib import Path
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from env.trading_env import TradingEnv
from training.config import TrainingConfig
from training.train import run_episode, run_random_baseline
from agents.researcher import QuantResearcher
from agents.fa_agent import FundamentalAnalyst
from agents.risk_model import RiskModeler
from agents.trader import QuantTrader
from agents.portfolio_manager import PortfolioManager
from utils.judge import LLMJudge
from utils.visualization import (
    plot_reward_curve,
    plot_grade_progression,
    plot_comparison_table,
)
import argparse


def run_benchmark(episodes=50):
    """
    Compare trained multi-agent pipeline vs random baseline
    using the REAL agent interaction loop — no faked results.
    """
    config = TrainingConfig(
        tickers=["AAPL"],
        num_episodes=episodes,
        fast_mode=True,          # Skip LLM judge calls for speed
        max_steps=200,
    )
    env = TradingEnv(difficulty="hard", max_steps=200)

    # --- Trained pipeline (the multi-agent system) ---
    researcher = QuantResearcher()
    fa_agent = FundamentalAnalyst(fast_mode=True)
    risk_model = RiskModeler(
        max_drawdown_limit=config.risk_max_drawdown,
        max_exposure=config.risk_max_exposure,
        vol_threshold=config.risk_vol_threshold,
    )
    trader = QuantTrader(aggression=config.trader_aggression)
    portfolio_manager = PortfolioManager(fast_mode=True)
    judge = LLMJudge()   # Will use algorithmic fallback in fast_mode

    trained_metrics = []
    print(f"Running {episodes} Trained Episodes (Multi-Agent Pipeline)...")
    for ep in range(episodes):
        metrics, _ = run_episode(
            env, researcher, fa_agent, risk_model,
            trader, portfolio_manager, judge, config=config,
        )
        trained_metrics.append(metrics)
        if (ep + 1) % 10 == 0:
            print(f"  Trained ep {ep+1}/{episodes}: grade={metrics['final_grade']:.3f}, pnl={metrics['pnl_pct']:+.2%}")

    # --- Random baseline ---
    print(f"\nRunning {episodes} Baseline Episodes (Random)...")
    random_metrics = run_random_baseline(config, num_episodes=episodes)

    # --- Print results ---
    def avg(metrics, key):
        return np.mean([m[key] for m in metrics])

    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"\n{'Metric':<20} {'Random':>12} {'Trained':>12} {'Improvement':>14}")
    print("-" * 60)

    for key, label in [
        ("total_reward", "Avg Reward"),
        ("final_grade", "Avg Grade"),
        ("pnl_pct", "Avg PnL %"),
        ("max_drawdown", "Avg Max DD"),
        ("sharpe_ratio", "Avg Sharpe"),
    ]:
        r = avg(random_metrics, key)
        t = avg(trained_metrics, key)
        imp = t - r
        sign = "+" if imp > 0 else ""
        print(f"  {label:<18} {r:>12.4f} {t:>12.4f} {sign}{imp:>13.4f}")

    # --- Generate plots ---
    print("\nGenerating comparison plots...")
    plot_reward_curve(trained_metrics, random_metrics)
    plot_grade_progression(trained_metrics, random_metrics)
    plot_comparison_table(trained_metrics, random_metrics)
    print("Done! Plots saved to plots/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    run_benchmark(episodes=args.episodes)
