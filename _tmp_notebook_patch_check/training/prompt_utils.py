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

SYSTEM_PROMPT = """You are a trading agent in a multi-agent governance system.
The Risk Manager has set governance constraints, and the Portfolio Manager has allocated capital.
Your job: propose a trade that maximizes profit while respecting these constraints.

Respond exactly in this format:
<thought>
Your reasoning about the market state, risk constraints, and trade decision.
</thought>
<action>
{"direction": 0, "size": 0.0, "sl": 0, "tp": 0}
</action>
"""

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
    """Build the prompt for the Trader, including RM and PM constraints."""
    rm_limit = scenario["rm_size_limit"]
    rm_allow_str = "allowed" if scenario.get("rm_allow_new", True) else "BLOCKED"
    rm_force_str = "yes" if scenario.get("rm_force_reduce", False) else "no"
    pm_cap = scenario["pm_cap_alloc"]
    pm_override_str = "none" if scenario.get("pm_override", 0.0) < 0.5 else "ACTIVE"

    state = scenario.get("state", [1.0, 1.0, 1.0, 1.0, 1.0])
    signals = scenario.get("signals", {})

    body = json.dumps({
        "state": state,
        "signals": signals,
        "governance": {
            "rm_size_limit": rm_limit,
            "rm_allow_new": rm_allow_str,
            "rm_force_reduce": rm_force_str,
            "pm_cap_alloc": pm_cap,
            "pm_override": pm_override_str,
        },
    }, separators=(",", ":"))

    prompt = (
        f"{SYSTEM_PROMPT}\n"
        f"The Risk Manager has set the following constraints: "
        f"size_limit={rm_limit:.2f}, new_positions={rm_allow_str}, force_reduce={rm_force_str}.\n"
        f"The Portfolio Manager allocated: capital_cap={pm_cap:.2f}, override={pm_override_str}.\n\n"
        f"Scenario:\n{body}\n"
    )
    return prompt
