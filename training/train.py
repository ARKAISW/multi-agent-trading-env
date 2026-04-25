"""
Training loop for the multi-agent trading environment.
Runs episodic simulation with the full agent interaction loop.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from env.trading_env import TradingEnv
from agents.researcher import QuantResearcher
from agents.fa_agent import FundamentalAnalyst
from agents.risk_model import RiskModeler
from agents.trader import QuantTrader
from agents.portfolio_manager import PortfolioManager
from training.config import TrainingConfig
from utils.judge import LLMJudge


def _to_jsonable(value):
    """Convert nested numpy scalars/arrays into plain Python values."""
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _append_trajectory_batch(path: str, trajectories: List[Dict]) -> None:
    """Append one episode of SFT trajectories to a JSONL file."""
    if not trajectories:
        return

    with open(path, "a", encoding="utf-8") as handle:
        for row in trajectories:
            handle.write(json.dumps(_to_jsonable(row)) + "\n")


def run_episode(
    env: TradingEnv,
    researcher: QuantResearcher,
    fa_agent: FundamentalAnalyst,
    risk_model: RiskModeler,
    trader: QuantTrader,
    portfolio_manager: PortfolioManager,
    judge: LLMJudge,
    config: Optional[TrainingConfig] = None,
) -> tuple[Dict, List[Dict]]:
    """
    Run a single episode of the multi-agent trading loop.
    Collects text-reasoning for SFT and uses LLM Judge for RL rewards.
    """
    obs, info = env.reset()
    fa_agent.reset()
    portfolio_manager.reset()

    total_reward = 0.0
    step_rewards = []
    
    # Storage for SFT Data Collection
    episode_trajectories = []

    done = False
    step_count = 0
    while not done:
        step_count += 1
        state_snapshot = obs.tolist()
        current_price = env.market.current_price()

        # 1. Researcher: TA signal + Reasoning
        res_signal, res_conf, res_reasoning = researcher(obs)

        # 2. FA Agent: sentiment bias + Reasoning
        fa_sentiment, fa_reasoning = fa_agent(obs)

        # 3. Risk Model: constraints + Reasoning
        risk_limit, risk_constraints, risk_reasoning = risk_model(obs)
        risk_constraints["raw_price"] = current_price

        # 4. Trader: action + reasoning
        direction, size, sl, tp, trader_reasoning = trader(
            obs, 
            (res_signal, res_conf, res_reasoning),
            (fa_sentiment, fa_reasoning),
            (risk_limit, risk_constraints, risk_reasoning)
        )

        # 5. Portfolio Manager: review
        capital_allocation, override = portfolio_manager(obs, (direction, size), info)
        if override is not None:
            direction, size = override

        # 6. Environment step
        action = {
            "direction": direction, "size": np.array([size], dtype=np.float32),
            "sl": np.array([sl], dtype=np.float32), "tp": np.array([tp], dtype=np.float32),
        }
        obs, env_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- JUDGE: LLM-based Quality Reward ---
        # The judge evaluates the "Inter-agent reasoning" and "Action Alignment"
        agent_reasoning = {
            "researcher": res_reasoning,
            "fundamental": fa_reasoning,
            "risk": risk_reasoning,
            "trader": trader_reasoning
        }
        
        # We only call the judge periodically or in 'high-stakes' steps to save API tokens
        judge_reward = 0.5
        if not (config and config.fast_mode) and (step_count % 5 == 0 or direction != 0):
            state_brief = f"Price: {current_price:.2f}, Vol: {obs[12]:.4f}, PnL: {info.get('pnl_pct', 0):.2%}"
            judge_reward = judge.evaluate_step(state_brief, agent_reasoning, action, info)

        # Combined RL Reward: Environment (PnL) + Judge (Professionalism)
        # Weighting can be tuned; 70% env, 30% judge is a good start
        final_reward = 0.7 * env_reward + 0.3 * judge_reward
        
        total_reward += final_reward
        step_rewards.append(final_reward)

        # Log for SFT data
        episode_trajectories.append({
            "step": step_count,
            "state": state_snapshot,
            "signals": {
                "ta_score": res_conf if res_signal == "bullish" else (-res_conf if res_signal == "bearish" else 0.0),
                "fa_sentiment": (fa_sentiment * 2.0) - 1.0,
                "position_limit": risk_limit,
                "constraints": {k: v for k, v in risk_constraints.items() if k != "raw_price"},
                "reasoning": agent_reasoning,
            },
            "action": {
                "direction": int(direction),
                "size": float(size),
                "sl": float(sl),
                "tp": float(tp),
            },
            "env_reward": float(env_reward),
            "judge_reward": float(judge_reward),
            "reward": float(final_reward),
        })
        
        if not (config and config.fast_mode):
            print(f"  Step {step_count:>3d} | Reward: {final_reward:.3f} | Env: {env_reward:.2f} | Judge: {judge_reward:.2f}", end="\r")

    if not (config and config.fast_mode):
        print() 
    
    # Save SFT data if needed (logic omitted for brevity)
    
    metrics = {
        "total_reward": total_reward,
        "mean_reward": float(np.mean(step_rewards)) if step_rewards else 0.0,
        "final_grade": info.get("grade", 0.0),
        "final_value": info.get("portfolio_value", 0.0),
        "pnl_pct": info.get("pnl_pct", 0.0),
        "max_drawdown": info.get("max_drawdown", 0.0),
        "sharpe_ratio": info.get("sharpe_ratio", 0.0),
        "trade_count": info.get("trade_count", 0),
    }
    for row in episode_trajectories:
        row["final_grade"] = metrics["final_grade"]
        row["episode_total_reward"] = metrics["total_reward"]
    return metrics, episode_trajectories


def train(
    config: TrainingConfig,
    df: Optional[pd.DataFrame] = None,
) -> List[Dict]:
    """
    Run the full training loop with LLM Judge integration.
    """
    np.random.seed(config.seed)

    env = TradingEnv(
        df=df, initial_cash=config.initial_cash,
        ticker=config.tickers[0] if config.tickers else "default",
        commission=config.commission,
        reward_weights=config.reward_weights,
        max_steps=config.max_steps,
    )

    # Initialize agents
    researcher = QuantResearcher()
    fa_agent = FundamentalAnalyst(fast_mode=config.fast_mode)
    risk_model = RiskModeler(
        max_drawdown_limit=config.risk_max_drawdown,
        max_exposure=config.risk_max_exposure,
        vol_threshold=config.risk_vol_threshold,
    )
    trader = QuantTrader(aggression=config.trader_aggression)
    portfolio_manager = PortfolioManager(fast_mode=config.fast_mode)
    judge = LLMJudge()

    all_metrics = []
    trajectory_path = os.path.join(config.save_dir, config.trajectories_file)
    print(f"\nStarting training with LLM Judge (Llama 3.3 70B)")
    os.makedirs(config.save_dir, exist_ok=True)
    if config.save_trajectories and os.path.exists(trajectory_path):
        os.remove(trajectory_path)

    for episode in range(config.num_episodes):
        metrics, trajectories = run_episode(
            env,
            researcher,
            fa_agent,
            risk_model,
            trader,
            portfolio_manager,
            judge,
            config=config,
        )
        metrics["episode"] = episode
        all_metrics.append(metrics)
        if config.save_trajectories:
            for row in trajectories:
                row["episode"] = episode
            _append_trajectory_batch(trajectory_path, trajectories)

        if (episode + 1) % config.log_every == 0 or episode == 0:
            print(f"Ep {episode+1:>4d} | Reward: {metrics['total_reward']:>8.3f} | PnL: {metrics['pnl_pct']:>+7.2%} | Grade: {metrics['final_grade']:.3f}")

    # Save results
    pd.DataFrame(all_metrics).to_csv(os.path.join(config.save_dir, config.metrics_file), index=False)
    return all_metrics


def run_random_baseline(
    config: TrainingConfig,
    df: Optional[pd.DataFrame] = None,
    num_episodes: int = 10,
) -> List[Dict]:
    """
    Run episodes with random actions as a baseline for comparison.
    """
    env = TradingEnv(
        df=df,
        initial_cash=config.initial_cash,
        ticker=config.tickers[0] if config.tickers else "default",
        commission=config.commission,
        reward_weights=config.reward_weights,
        max_steps=config.max_steps,
    )

    all_metrics = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action_space: Any = env.action_space
            action = {
                "direction": action_space["direction"].sample(),
                "size": action_space["size"].sample(),
                "sl": np.array([0.0], dtype=np.float32),
                "tp": np.array([0.0], dtype=np.float32),
            }
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        metrics = {
            "episode": ep,
            "total_reward": total_reward,
            "final_grade": info.get("grade", 0.0),
            "pnl_pct": info.get("pnl_pct", 0.0),
            "max_drawdown": info.get("max_drawdown", 0.0),
            "sharpe_ratio": info.get("sharpe_ratio", 0.0),
            "trade_count": info.get("trade_count", 0),
        }
        all_metrics.append(metrics)

    return all_metrics
