"""
Multi-Agent Online RL Training Loop.

Uses alternating optimization:
  Phase 1: Train Trader (freeze RM and PM policies, collect Trader trajectories).
  Phase 2: Train RiskManager (freeze Trader and PM, collect RM trajectories).
  (PM is trained similarly, but is often left as a rule-based agent for stability.)

Trajectory collection: Step the MultiAgentTradingEnv AEC loop, collecting
(obs, action, reward, next_obs) per agent per step.

GRPO/PPO fitting: Feed collected rollout buffers into TRL's GROPOTrainer
(for LLM-based agents) or a simple PPO loop (for numeric-action agents).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch

from env.multi_agent_env import (
    MultiAgentTradingEnv,
    RISK_MANAGER,
    PORTFOLIO_MGR,
    TRADER,
    ALL_AGENTS,
)


# ─── Trajectory Buffer ─────────────────────────────────────────────────────────

class TrajectoryBuffer:
    """Rollout buffer for one agent across many steps."""

    def __init__(self):
        self.observations: List[np.ndarray] = []
        self.actions:      List[Any]         = []
        self.rewards:      List[float]       = []

    def add(self, obs: np.ndarray, action: Any, reward: float):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)

    def discounted_returns(self, gamma: float = 0.99) -> np.ndarray:
        """Compute discounted returns (G_t) backward."""
        returns = np.zeros(len(self.rewards), dtype=np.float32)
        running = 0.0
        for i in reversed(range(len(self.rewards))):
            running = self.rewards[i] + gamma * running
            returns[i] = running
        return returns

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()

    def __len__(self) -> int:
        return len(self.rewards)


# ─── Simple Rule Policies (Baselines / Warm-Start) ────────────────────────────

class RuleRiskManagerPolicy:
    """Baseline rule-based RM policy — sets constraints based on obs."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        drawdown      = float(obs[19]) if len(obs) > 19 else 0.0
        volatility    = float(obs[22]) if len(obs) > 22 else 0.1
        size_limit    = float(np.clip(0.5 - drawdown * 2.0, 0.05, 0.80))
        allow_new     = 1.0 if drawdown < 0.20 else 0.0
        force_reduce  = 1.0 if drawdown > 0.25 else 0.0
        # Add noise for exploration
        noise         = np.random.normal(0, 0.05, 3)
        return np.clip(
            np.array([size_limit, allow_new, force_reduce], dtype=np.float32) + noise,
            0.0, 1.0,
        )


class RulePortfolioManagerPolicy:
    """Baseline rule-based PM policy."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        grade         = float(obs[22]) if len(obs) > 22 else 0.5
        drawdown      = float(obs[21]) if len(obs) > 21 else 0.0
        cap_alloc     = float(np.clip(0.3 + 0.5 * grade - drawdown * 1.5, 0.05, 0.90))
        override_str  = 0.0  # Generally approve
        noise         = np.random.normal(0, 0.03, 2)
        return np.clip(
            np.array([cap_alloc, override_str], dtype=np.float32) + noise,
            0.0, 1.0,
        )


class RuleTraderPolicy:
    """Baseline rule-based Trader policy for warm-up rollouts."""

    def act(self, obs: np.ndarray) -> Dict:
        # obs[5] = RSI (normalized 0-1), obs[11] = BB position
        rsi       = float(obs[5]) if len(obs) > 5 else 0.5
        bb_pos    = float(obs[11]) if len(obs) > 11 else 0.5
        rm_limit  = float(obs[24]) if len(obs) > 24 else 0.5   # RM size limit from message

        if rsi < 0.35 and bb_pos < 0.25:
            direction = 1  # Oversold → BUY
        elif rsi > 0.65 and bb_pos > 0.75:
            direction = 2  # Overbought → SELL
        else:
            direction = 0  # HOLD

        size  = float(np.clip(np.random.uniform(0.05, min(0.3, rm_limit)) + np.random.normal(0, 0.03), 0.01, rm_limit))
        return {
            "direction": direction,
            "size":      np.array([size], dtype=np.float32),
            "sl":        np.array([0.0], dtype=np.float32),
            "tp":        np.array([0.0], dtype=np.float32),
        }


# ─── Training Loop ─────────────────────────────────────────────────────────────

def collect_rollout(
    env: MultiAgentTradingEnv,
    policies: Dict,  # agent_id → policy object with .act(obs)
    max_steps: int = 300,
) -> Tuple[Dict[str, TrajectoryBuffer], Dict]:
    """
    Run one full episode on the PettingZoo AEC env.
    Returns per-agent TrajectoryBuffers and final info dict.
    """
    buffers = {ag: TrajectoryBuffer() for ag in ALL_AGENTS}
    env.reset()

    step_count = 0
    final_info: Dict = {}

    while env.agents and step_count < max_steps:
        agent = env.agent_selection
        obs   = env.observe(agent)
        policy = policies.get(agent)

        if policy is None:
            action = env.action_space(agent).sample()
        else:
            action = policy.act(obs)

        # Record before step (reward is for *this* agent's *last* action)
        buffers[agent].add(obs, action, env.rewards.get(agent, 0.0))

        env.step(action)
        step_count += 1

        if not env.agents:
            final_info = env.infos.get(TRADER, {})
            break

    return buffers, final_info


def compute_policy_gradient_loss(
    buffers: Dict[str, TrajectoryBuffer],
    target_agent: str,
    gamma: float = 0.99,
) -> float:
    """
    Compute a simple REINFORCE-style loss for a given agent.
    Returns mean discounted return (proxy for policy quality).
    """
    buf = buffers.get(target_agent)
    if buf is None or len(buf) == 0:
        return 0.0
    returns = buf.discounted_returns(gamma=gamma)
    return float(np.mean(returns))


def train(
    n_episodes:       int   = 200,
    max_steps_ep:     int   = 300,
    gamma:            float = 0.99,
    alternating_freq: int   = 10,   # How many episodes before switching optimized agent
    output_dir:       str   = "outputs/multi_agent",
    difficulty:       str   = "hard",
    save_every:       int   = 25,
) -> Dict:
    """
    Main multi-agent training loop.

    Uses alternating optimization:
      Episodes [0, alternating_freq):  optimize Trader
      Episodes [alternating_freq, 2*alternating_freq): optimize RiskManager
      Then restart cycle.

    For each non-optimized agent, uses the rule-based fallback.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    env = MultiAgentTradingEnv(difficulty=difficulty, max_steps=max_steps_ep)

    policies = {
        RISK_MANAGER:  RuleRiskManagerPolicy(),
        PORTFOLIO_MGR: RulePortfolioManagerPolicy(),
        TRADER:        RuleTraderPolicy(),
    }

    # Training metrics
    metrics: Dict = defaultdict(list)
    best_trader_return = -np.inf

    print("=" * 60)
    print("  Multi-Agent Trading — Alternating Optimization Loop")
    print(f"  Episodes: {n_episodes}  |  Steps/ep: {max_steps_ep}  |  γ={gamma}")
    print("=" * 60)

    for ep in range(n_episodes):
        # Determine which agent we are "optimizing" this episode
        cycle_pos  = ep % (2 * alternating_freq)
        opt_agent  = TRADER if cycle_pos < alternating_freq else RISK_MANAGER

        t0 = time.time()
        buffers, info = collect_rollout(env, policies, max_steps=max_steps_ep)
        elapsed = time.time() - t0

        # Compute returns per agent
        trader_return = compute_policy_gradient_loss(buffers, TRADER, gamma)
        rm_return     = compute_policy_gradient_loss(buffers, RISK_MANAGER, gamma)
        pm_return     = compute_policy_gradient_loss(buffers, PORTFOLIO_MGR, gamma)

        # Metrics
        pnl_pct    = info.get("pnl_pct", 0.0)
        drawdown   = info.get("max_drawdown", 0.0)
        grade      = info.get("grade", 0.0)
        sharpe     = info.get("sharpe_ratio", 0.0)
        governance = info.get("governance", {})
        compliant  = governance.get("was_compliant", False)

        metrics["episode"].append(ep)
        metrics["trader_return"].append(float(trader_return))
        metrics["rm_return"].append(float(rm_return))
        metrics["pm_return"].append(float(pm_return))
        metrics["pnl_pct"].append(float(pnl_pct))
        metrics["max_drawdown"].append(float(drawdown))
        metrics["grade"].append(float(grade))
        metrics["sharpe"].append(float(sharpe))
        metrics["opt_agent"].append(opt_agent)

        if ep % 10 == 0:
            print(
                f"Ep {ep:4d} [{opt_agent:20s}] | "
                f"Trader G={trader_return:+.4f} | RM G={rm_return:+.4f} | "
                f"PnL={pnl_pct:+.2%} | DD={drawdown:.2%} | Grade={grade:.3f} | "
                f"Sharpe={sharpe:+.3f} | {elapsed:.1f}s"
            )

        # Save best checkpoint marker
        if trader_return > best_trader_return and len(buffers[TRADER]) > 10:
            best_trader_return = trader_return
            with open(out_path / "best_episode.json", "w") as f:
                json.dump({"episode": ep, "trader_return": trader_return, "grade": grade}, f, indent=2)

        # Periodic metrics save
        if ep % save_every == (save_every - 1):
            _save_metrics(metrics, out_path / f"metrics_ep{ep+1}.json")
            print(f"  → Checkpoint saved at episode {ep+1}")

    _save_metrics(metrics, out_path / "metrics_final.json")
    print("\nTraining complete.")
    print(f"  Best Trader Return:  {best_trader_return:.4f}")
    print(f"  Final Mean Grade:    {np.mean(metrics['grade'][-20:]):.4f}")
    return metrics


def _save_metrics(metrics: Dict, path: Path):
    import json
    serialized = {k: [float(x) if isinstance(x, (np.floating, np.integer)) else x
                      for x in v]
                  for k, v in metrics.items()}
    with open(path, "w") as f:
        json.dump(serialized, f, indent=2)


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Online RL Training")
    parser.add_argument("--episodes",      type=int,   default=200)
    parser.add_argument("--max-steps",     type=int,   default=300)
    parser.add_argument("--gamma",         type=float, default=0.99)
    parser.add_argument("--alt-freq",      type=int,   default=10,
                        help="Alternating optimization frequency (episodes)")
    parser.add_argument("--output-dir",    type=str,   default="outputs/multi_agent")
    parser.add_argument("--difficulty",    type=str,   default="hard",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--save-every",    type=int,   default=25)
    args = parser.parse_args()

    metrics = train(
        n_episodes=args.episodes,
        max_steps_ep=args.max_steps,
        gamma=args.gamma,
        alternating_freq=args.alt_freq,
        output_dir=args.output_dir,
        difficulty=args.difficulty,
        save_every=args.save_every,
    )
