import sys
import json
import random
from pathlib import Path
from typing import Dict, List
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.multi_agent_env import (
    MultiAgentTradingEnv,
    RISK_MANAGER,
    PORTFOLIO_MGR,
    TRADER,
)
from training.train_multi_agent import (
    RuleRiskManagerPolicy,
    RulePortfolioManagerPolicy,
)

SYSTEM_PROMPT = """You are the Trader agent in a decentralized multi-agent trading governance system.
Three independent agents operate in sequence each market step:
  1. Risk Manager — sets position limits and may force reductions
  2. Portfolio Manager — allocates capital and may override trades
  3. You (Trader) — propose trades that maximize profit while respecting governance

CRITICAL: You MUST comply with the Risk Manager's size limit. Exceeding it triggers an intervention.

Respond exactly in this format:
<thought>
Analyze the market conditions, explain how governance constraints affect your decision,
and justify your trade. Reference specific indicators (RSI, MACD, drawdown, etc.)
and the Risk Manager's limits in your reasoning. Minimum 150 characters.
</thought>
<action>
{"direction": 0, "size": 0.0, "sl": 0, "tp": 0}
</action>

direction: 0=HOLD, 1=BUY, 2=SELL
size: fraction of portfolio (0.0 to 1.0) — MUST be ≤ Risk Manager's size_limit
sl: stop-loss (0 = none)
tp: take-profit (0 = none)
"""


# ---------------------------------------------------------------------------
# Semantic helpers — translate raw floats into human-readable descriptions
# ---------------------------------------------------------------------------

def _rsi_description(rsi_norm: float) -> str:
    """Describe RSI from its [0,1] normalized value."""
    rsi = rsi_norm * 100
    if rsi < 30:
        return f"RSI is {rsi:.1f} (oversold — potential bounce)"
    elif rsi > 70:
        return f"RSI is {rsi:.1f} (overbought — potential pullback)"
    else:
        return f"RSI is {rsi:.1f} (neutral range)"


def _trend_description(ema20_ratio: float, ema50_ratio: float) -> str:
    """Describe EMA trend from price-relative ratios."""
    if ema20_ratio > 1.01 and ema50_ratio > 1.01:
        return "Strong uptrend (price above both EMA-20 and EMA-50)"
    elif ema20_ratio < 0.99 and ema50_ratio < 0.99:
        return "Strong downtrend (price below both EMA-20 and EMA-50)"
    elif ema20_ratio > ema50_ratio:
        return "Emerging uptrend (EMA-20 crossing above EMA-50)"
    else:
        return "Consolidating (mixed EMA signals)"


def _macd_description(macd: float, macd_hist: float) -> str:
    """Describe MACD momentum from tanh-normalized values."""
    if macd > 0.1 and macd_hist > 0:
        return "Bullish momentum (MACD positive and rising)"
    elif macd < -0.1 and macd_hist < 0:
        return "Bearish momentum (MACD negative and falling)"
    elif macd_hist > 0:
        return "Momentum turning bullish (histogram rising)"
    elif macd_hist < 0:
        return "Momentum turning bearish (histogram falling)"
    else:
        return "Flat momentum (MACD near zero)"


def _bb_description(bb_pos: float) -> str:
    """Describe Bollinger Band position."""
    if bb_pos > 0.9:
        return f"Price near upper Bollinger Band ({bb_pos:.0%}) — stretched high"
    elif bb_pos < 0.1:
        return f"Price near lower Bollinger Band ({bb_pos:.0%}) — stretched low"
    else:
        return f"Price mid-Bollinger range ({bb_pos:.0%})"


def _drawdown_description(dd: float) -> str:
    """Describe drawdown severity."""
    if dd < 0.02:
        return f"Minimal drawdown ({dd:.1%})"
    elif dd < 0.05:
        return f"Moderate drawdown ({dd:.1%}) — caution advised"
    elif dd < 0.10:
        return f"Significant drawdown ({dd:.1%}) — risk elevated"
    else:
        return f"Severe drawdown ({dd:.1%}) — capital preservation priority"


def generate_pz_scenarios(
    n: int = 500,
    difficulty: str = "easy",
    max_env_steps: int = 100,
) -> List[Dict]:
    """Run the PZ env with rule policies to generate realistic scenarios.

    Each scenario captures:
      - The Trader's full observation (29 dims)
      - The RM constraints decoded from the message
      - The PM allocation decoded from the message
    """
    env = MultiAgentTradingEnv(difficulty=difficulty, max_steps=max_env_steps)
    rm_policy = RuleRiskManagerPolicy()
    pm_policy = RulePortfolioManagerPolicy()

    scenarios: List[Dict] = []
    attempts = 0
    max_attempts = n * 3

    while len(scenarios) < n and attempts < max_attempts:
        env.reset()
        attempts += 1

        step_count = 0
        while env.agents and step_count < max_env_steps:
            agent = env.agent_selection

            if agent == RISK_MANAGER:
                obs = env.observe(agent)
                action = rm_policy.act(obs)
                env.step(action)

            elif agent == PORTFOLIO_MGR:
                obs = env.observe(agent)
                action = pm_policy.act(obs)
                env.step(action)

            elif agent == TRADER:
                obs = env.observe(agent)
                # Extract RM and PM messages from the observation
                # obs layout: base(24) + rm_msg(3) + pm_msg(2) = 29
                base_obs = obs[:24].tolist()
                rm_msg = obs[24:27].tolist()  # [size_limit, allow_new, force_reduce]
                pm_msg = obs[27:29].tolist()  # [cap_alloc, override_strength]

                rm_size_limit = float(rm_msg[0])
                rm_allow_new = bool(rm_msg[1] > 0.5)
                rm_force_reduce = bool(rm_msg[2] > 0.5)
                pm_cap_alloc = float(pm_msg[0])
                pm_override = float(pm_msg[1])

                scenarios.append({
                    "state": [round(float(x), 4) for x in base_obs[:5]],
                    "full_obs": [round(float(x), 4) for x in base_obs],
                    "rm_size_limit": round(rm_size_limit, 3),
                    "rm_allow_new": rm_allow_new,
                    "rm_force_reduce": rm_force_reduce,
                    "pm_cap_alloc": round(pm_cap_alloc, 3),
                    "pm_override": round(pm_override, 3),
                    "signals": {
                        "ta": round(float(obs[5] * 2 - 1), 3),  # RSI mapped to [-1,1]
                        "fa": round(float(obs[8]), 3),  # MACD as FA proxy
                        "position_limit": round(rm_size_limit, 3),
                        "rm_size_limit": round(rm_size_limit, 3),
                    },
                })

                if len(scenarios) >= n:
                    break

                # Take a random trader action so the env advances
                trader_action = {
                    "direction": random.choice([0, 1, 2]),
                    "size": np.array([random.uniform(0.05, 0.3)], dtype=np.float32),
                    "sl": np.array([0.0], dtype=np.float32),
                    "tp": np.array([0.0], dtype=np.float32),
                }
                env.step(trader_action)

            step_count += 1

    random.shuffle(scenarios)
    return scenarios[:n]


def build_prompt_multiagent(scenario: Dict) -> str:
    """Build a semantically rich prompt for the Trader agent.

    Translates raw observation floats into human-readable market analysis
    that leverages the LLM's pre-trained reasoning capabilities.
    """
    # --- Governance constraints ---
    rm_limit = scenario["rm_size_limit"]
    rm_allow = scenario.get("rm_allow_new", True)
    rm_force = scenario.get("rm_force_reduce", False)
    pm_cap = scenario["pm_cap_alloc"]
    pm_override = scenario.get("pm_override", 0.0)

    rm_allow_str = "ALLOWED" if rm_allow else "BLOCKED"
    rm_force_str = "YES — reduce positions immediately" if rm_force else "No"
    pm_override_str = "ACTIVE — PM is overriding" if pm_override >= 0.5 else "Inactive"

    # --- Decode observation vector into semantic features ---
    full_obs = scenario.get("full_obs", [1.0] * 24)

    # Market features (indices 0-13)
    # 0-3: OHLC ratios, 4: volume, 5: RSI, 6: EMA20, 7: EMA50
    # 8: MACD, 9: MACD_signal, 10: MACD_hist, 11: BB_position
    # 12: volatility, 13: ATR
    rsi_norm = full_obs[5] if len(full_obs) > 5 else 0.5
    ema20 = full_obs[6] if len(full_obs) > 6 else 1.0
    ema50 = full_obs[7] if len(full_obs) > 7 else 1.0
    macd = full_obs[8] if len(full_obs) > 8 else 0.0
    macd_hist = full_obs[10] if len(full_obs) > 10 else 0.0
    bb_pos = full_obs[11] if len(full_obs) > 11 else 0.5
    volatility = full_obs[12] if len(full_obs) > 12 else 0.0
    atr = full_obs[13] if len(full_obs) > 13 else 0.0

    # Portfolio features (indices 14-18)
    cash_ratio = full_obs[14] if len(full_obs) > 14 else 1.0
    long_exposure = full_obs[15] if len(full_obs) > 15 else 0.0
    port_return = full_obs[16] if len(full_obs) > 16 else 1.0
    unrealized_pnl = full_obs[17] if len(full_obs) > 17 else 0.0

    # Risk features (indices 19-23)
    drawdown = full_obs[19] if len(full_obs) > 19 else 0.0
    max_dd = full_obs[20] if len(full_obs) > 20 else 0.0
    sharpe = full_obs[21] if len(full_obs) > 21 else 0.0

    # --- Build semantic market analysis ---
    rsi_text = _rsi_description(rsi_norm)
    trend_text = _trend_description(ema20, ema50)
    macd_text = _macd_description(macd, macd_hist)
    bb_text = _bb_description(bb_pos)
    dd_text = _drawdown_description(drawdown)

    vol_pct = volatility * 100
    vol_text = "low" if vol_pct < 15 else ("moderate" if vol_pct < 40 else "high")

    prompt = f"""{SYSTEM_PROMPT}
═══ MARKET CONDITIONS ═══
• {rsi_text}
• {trend_text}
• {macd_text}
• {bb_text}
• Volatility: {vol_pct:.1f}% ({vol_text}), ATR ratio: {atr:.4f}

═══ PORTFOLIO STATUS ═══
• Cash available: {cash_ratio:.0%} of initial capital
• Long exposure: {long_exposure:.1%} of portfolio
• Portfolio return: {port_return:.2%} vs. initial
• Unrealized P&L: {unrealized_pnl:+.3f} (normalized)

═══ RISK METRICS ═══
• {dd_text}
• Maximum drawdown: {max_dd:.1%}
• Sharpe ratio: {sharpe:.3f}

═══ GOVERNANCE CONSTRAINTS (from Risk Manager & Portfolio Manager) ═══
• Risk Manager Position Limit: {rm_limit:.2f} (you MUST NOT exceed this)
• New Positions: {rm_allow_str}
• Force Reduce: {rm_force_str}
• Portfolio Manager Capital Allocation: {pm_cap:.2f}
• PM Override: {pm_override_str}

Given the above conditions, analyze the market, check your compliance limits, and propose your action.
"""
    return prompt
