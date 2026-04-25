"""
Multi-Agent Trading Environment using PettingZoo AEC API.

Three independent RL agents operate in a decentralized governance framework:
  - risk_manager_0:    Rewarded for restricting dangerous trades. Penalized when Trader loses.
  - portfolio_manager_0: Oversees capital allocation. Rewarded for portfolio growth + drawdown control.
  - trader_0:          Rewarded purely for PnL. Sees Risk/PM constraints as observations.

The AEC (Agent-Environment Cycle) loop alternates agent turns each step.
Agent Negotiation: Each agent's *output message* (constraints, allocations) becomes
part of the next agent's observation, creating an emergent negotiation dynamic.
"""

from __future__ import annotations

import functools
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from gymnasium import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from env.state import MarketState, PortfolioState, RiskState, get_observation
from env.reward import compute_raw_reward, normalize_reward, compute_grade
from utils.indicators import compute_indicators


# â”€â”€â”€ Agent IDs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
RISK_MANAGER    = "risk_manager_0"
PORTFOLIO_MGR   = "portfolio_manager_0"
TRADER          = "trader_0"
ALL_AGENTS      = [RISK_MANAGER, PORTFOLIO_MGR, TRADER]

# â”€â”€â”€ Observation Sizes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Base market+portfolio+risk obs size: 14 + 5 + 5 = 24
BASE_OBS_SIZE = 24
# Risk Manager message appended to PM and Trader observations: [size_limit, allow_new, force_reduce]
RM_MSG_SIZE = 3
# PM message appended to Trader observations: [cap_allocation, is_override_signaled]
PM_MSG_SIZE = 2


class MultiAgentTradingEnv(AECEnv):
    """
    A PettingZoo AEC environment for decentralized multi-agent trading governance.

    Turn order per step: risk_manager_0 â†’ portfolio_manager_0 â†’ trader_0
    On each full cycle, the market advances by one candle.

    Observations:
      risk_manager_0:   base_obs (24,)
      portfolio_mgr_0:  base_obs + rm_message (24 + 3 = 27,)
      trader_0:         base_obs + rm_message + pm_message (24 + 3 + 2 = 29,)

    Actions:
      risk_manager_0:   Box(3,) â€” [size_limit, allow_new_positions, force_reduce] â€” continuous
      portfolio_mgr_0:  Box(2,) â€” [capital_allocation_fraction, override_flag] â€” continuous
      trader_0:         Dict â€” direction (Discrete 3), size (Box 1), sl (Box 1), tp (Box 1)
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "multi_agent_trading_v1",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        initial_cash: float = 100_000.0,
        ticker: str = "default",
        commission: float = 0.001,
        max_steps: Optional[int] = None,
        difficulty: str = "hard",
    ):
        super().__init__()

        self.difficulty = difficulty
        if df is None:
            df = self._make_dummy_data(difficulty=difficulty)
        self.raw_df = df.copy()
        self.df = compute_indicators(df)
        self.ticker = ticker
        self.initial_cash = initial_cash
        self.commission = commission
        self.max_steps = max_steps or (len(self.df) - 1)

        # â”€â”€ PettingZoo required attributes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.agents = ALL_AGENTS[:]
        self.possible_agents = ALL_AGENTS[:]

        # â”€â”€ Observation spaces â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.observation_spaces = {
            RISK_MANAGER:   spaces.Box(low=-np.inf, high=np.inf,
                                       shape=(BASE_OBS_SIZE,), dtype=np.float32),
            PORTFOLIO_MGR:  spaces.Box(low=-np.inf, high=np.inf,
                                       shape=(BASE_OBS_SIZE + RM_MSG_SIZE,), dtype=np.float32),
            TRADER:         spaces.Box(low=-np.inf, high=np.inf,
                                       shape=(BASE_OBS_SIZE + RM_MSG_SIZE + PM_MSG_SIZE,), dtype=np.float32),
        }

        # â”€â”€ Action spaces â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.action_spaces = {
            RISK_MANAGER:  spaces.Box(low=np.array([0.01, 0.0, 0.0], dtype=np.float32),
                                      high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                                      shape=(3,), dtype=np.float32),
            PORTFOLIO_MGR: spaces.Box(low=np.array([0.0, 0.0], dtype=np.float32),
                                      high=np.array([1.0, 1.0], dtype=np.float32),
                                      shape=(2,), dtype=np.float32),
            TRADER:        spaces.Dict({
                "direction": spaces.Discrete(3),          # 0=Hold, 1=Buy, 2=Sell/Short
                "size":      spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "sl":        spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
                "tp":        spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            }),
        }

        # â”€â”€ Internal state (reset before first use) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self._agent_selector = agent_selector(ALL_AGENTS)
        self._reset_internal_state()

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # PettingZoo required API
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = ALL_AGENTS[:]
        self._agent_selector.reinit(ALL_AGENTS)

        self._reset_internal_state()
        self._generate_observations()

        self.agent_selection = self._agent_selector.reset()

        # Zero-fill all rewards/terminations/truncations/infos for PZ compliance
        self.rewards         = {ag: 0.0 for ag in self.agents}
        self._cumulative_rewards = {ag: 0.0 for ag in self.agents}
        self.terminations    = {ag: False for ag in self.agents}
        self.truncations     = {ag: False for ag in self.agents}
        self.infos           = {ag: {} for ag in self.agents}

    def step(self, action):
        """Process one agent's action in the AEC turn order."""
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            # Dead-step: PZ compliance requires we handle this
            self._was_dead_step(action)
            return

        # â”€â”€ Route action to the correct handler â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if agent == RISK_MANAGER:
            self._step_risk_manager(action)
        elif agent == PORTFOLIO_MGR:
            self._step_portfolio_manager(action)
        elif agent == TRADER:
            self._step_trader(action)
            # After the trader acts, the market cycle is complete â†’ advance step
            self._advance_market()

        # Advance to next agent
        self._accumulate_rewards()
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent: str) -> np.ndarray:
        return self._observations[agent]

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def render(self):
        price = self._market.current_price()
        val   = self._portfolio.total_value(price, self.ticker)
        print(
            f"Step {self._current_step:4d} | "
            f"Price: {price:10,.2f} | "
            f"Value: {val:12,.2f} | "
            f"Agent: {self.agent_selection}"
        )

    def close(self):
        pass

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Per-Agent Step Handlers
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _step_risk_manager(self, action: np.ndarray):
        """
        Risk Manager decides governance constraints.
        action = [size_limit (0-1), allow_new_positions (0-1), force_reduce (0-1)]

        Reward logic (adversarial):
          +0.2  for restricting a dangerous action (high drawdown â†’ low size_limit)
          -0.3  for each $ portfolio value LOST since it last acted (it shares downside pain)
          +0.05 for being compliant (not overriding a healthy portfolio)
        """
        size_limit, allow_new_raw, force_reduce_raw = float(action[0]), float(action[1]), float(action[2])
        allow_new  = allow_new_raw  > 0.5
        force_reduce = force_reduce_raw > 0.5

        # Store message to pass to PM and Trader
        self._rm_message = np.array(
            [size_limit, float(allow_new), float(force_reduce)], dtype=np.float32
        )

        # Compute RM's step reward
        drawdown = self._risk.current_drawdown
        rm_reward = 0.0

        # Rewarded for restricting size when portfolio is underwater
        if drawdown > 0.10 and size_limit < 0.30:
            rm_reward += 0.20   # RM correctly capped risk during drawdown

        if force_reduce and drawdown > 0.20:
            rm_reward += 0.15   # Correct force-reduce under severe drawdown

        # Penalize for allowing reckless sizing when at risk
        if drawdown > 0.15 and size_limit > 0.70:
            rm_reward -= 0.20   # RM being reckless during drawdown

        # Shared downside: RM suffers when portfolio loses money this step
        prev_val = self._prev_portfolio_value
        curr_price = self._market.current_price()
        curr_val   = self._portfolio.total_value(curr_price, self.ticker)
        portfolio_delta_pct = (curr_val - prev_val) / (self.initial_cash + 1e-10)
        rm_reward += min(portfolio_delta_pct * 0.5, 0.0)  # Only downside pain

        self._pending_rewards[RISK_MANAGER] = rm_reward

    def _step_portfolio_manager(self, action: np.ndarray):
        """
        Portfolio Manager decides capital allocation and optionally signals override.
        action = [capital_allocation (0-1), override_strength (0-1)]

        Reward logic:
          Aligned with overall portfolio performance (grade-based).
          Penalized for excessive overrides that don't improve outcomes.
        """
        cap_alloc  = float(np.clip(action[0], 0.0, 1.0))
        override_s = float(action[1])

        self._pm_message = np.array([cap_alloc, override_s], dtype=np.float32)
        self._pm_capital_allocation = cap_alloc
        self._pm_override_strength  = override_s

        # PM reward deferred to after trader executes (knows the outcome)
        self._pending_rewards[PORTFOLIO_MGR] = 0.0  # Will be updated in _advance_market

    def _step_trader(self, action: Dict):
        """
        Trader proposes a trade using the constrained action space.
        Receives both RM and PM guidance in its observation.

        Reward logic (adversarial):
          Rewarded purely on PnL.
          Penalized when governance overrides (RM size cap, PM force-close) are triggered.
          Bonus for proposing compliant actions that need no governance intervention.
        """
        direction = int(action["direction"])
        size_raw  = float(action["size"][0]) if hasattr(action["size"], "__len__") else float(action["size"])
        sl_input  = float(action["sl"][0])   if hasattr(action["sl"],   "__len__") else float(action.get("sl", 0.0))
        tp_input  = float(action["tp"][0])   if hasattr(action["tp"],   "__len__") else float(action.get("tp", 0.0))

        size = float(np.clip(size_raw, 0.0, 1.0))

        # â”€â”€ Apply Risk Manager constraints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        rm_size_limit  = float(self._rm_message[0])
        rm_allow_new   = bool(self._rm_message[1] > 0.5)
        rm_force_reduce = bool(self._rm_message[2] > 0.5)

        interventions: List[Dict] = []

        if direction != 0 and size > rm_size_limit:
            interventions.append({
                "agent": "RiskManager",
                "type":  "size_clamp",
                "original_size":  size,
                "enforced_size":  rm_size_limit,
            })
            size = rm_size_limit

        if direction in (1, 2) and not rm_allow_new:
            interventions.append({
                "agent": "RiskManager",
                "type":  "no_new_positions",
                "reason": "RM blocked new positions during drawdown",
            })
            direction = 0  # Force hold

        if rm_force_reduce and direction == 1:
            interventions.append({
                "agent": "RiskManager",
                "type":  "force_reduce",
                "reason": "RM signaling to reduce longs",
            })
            direction = 2  # Flip to reduce

        # â”€â”€ Apply Portfolio Manager override â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        cap_alloc  = self._pm_capital_allocation
        if direction != 0 and size > cap_alloc:
            interventions.append({
                "agent": "PortfolioManager",
                "type":  "capital_cap",
                "original_size": size,
                "enforced_size": cap_alloc,
            })
            size = min(size, cap_alloc)

        # PM strong override_strength >0.7 means PM wants to force hold
        if self._pm_override_strength > 0.7 and direction != 0:
            interventions.append({
                "agent": "PortfolioManager",
                "type":  "pm_veto",
                "reason": "PM vetoed trade (insufficient conviction signal)",
            })
            direction = 0

        # â”€â”€ Auto SL/TP (governance baseline) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        current_price = self._market.current_price()
        DEFAULT_SL = 0.02
        if direction != 0 and sl_input <= 0:
            if direction == 1:
                sl_input = current_price * (1 - DEFAULT_SL)
            else:
                sl_input = current_price * (1 + DEFAULT_SL)
            interventions.append({"agent": "RiskManager", "type": "auto_sl"})
        if direction != 0 and tp_input <= 0 and sl_input > 0:
            sl_dist = abs(current_price - sl_input)
            tp_input = (current_price + sl_dist * 2.0) if direction == 1 else (current_price - sl_dist * 2.0)
            interventions.append({"agent": "RiskManager", "type": "auto_tp"})

        # Store pending trade for market advance
        self._pending_trade = {
            "direction": direction,
            "size": size,
            "sl": sl_input,
            "tp": tp_input,
            "interventions": interventions,
            "original_direction": int(action["direction"]),
            "original_size": size_raw,
        }

        # Compliance reward/penalty â€” will be finalized after market moves
        n_interventions = len(interventions)
        compliance_bonus = 0.15 if (n_interventions == 0 and direction != 0) else (-0.05 * n_interventions)
        self._trader_compliance_bonus = compliance_bonus

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Market Advance (called after Trader acts)
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _advance_market(self):
        """Execute the pending trade, advance market, compute final rewards."""
        if not hasattr(self, "_pending_trade") or self._pending_trade is None:
            # No trade was staged (edge case)
            self._pending_trade = {"direction": 0, "size": 0.0, "sl": 0.0, "tp": 0.0,
                                   "interventions": [], "original_direction": 0, "original_size": 0.0}

        trade = self._pending_trade
        direction = trade["direction"]
        size      = trade["size"]
        sl_input  = trade["sl"]
        tp_input  = trade["tp"]

        current_price = self._market.current_price()
        prev_value    = self._portfolio.total_value(current_price, self.ticker)

        # Check SL/TP before executing new action
        self._check_sl_tp(current_price)

        # Execute trade in portfolio state
        traded = self._execute_trade(direction, size, sl_input, tp_input, current_price)

        # Advance market step
        self._current_step += 1
        self._market.current_step = self._current_step

        # Update risk state
        new_price = self._market.current_price() if self._current_step < len(self.df) else current_price
        new_value = self._portfolio.total_value(new_price, self.ticker)
        self._risk.update(new_value)
        self._episode_values.append(new_value)

        # Compute portfolio delta
        profit = (new_value - prev_value) / (self.initial_cash + 1e-10)
        price_trend = (new_price - current_price) / (current_price + 1e-10)

        raw_r = compute_raw_reward(
            profit=profit,
            drawdown=self._risk.current_drawdown,
            volatility=self._risk.return_volatility(),
            sharpe=self._risk.sharpe_ratio(),
            trade_count=int(traded),
            direction=direction,
            price_trend=price_trend,
        )

        # â”€â”€ Trader reward â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        trader_reward = normalize_reward(raw_r + self._trader_compliance_bonus)
        self._pending_rewards[TRADER] = float(trader_reward)
        self._episode_rewards.append(trader_reward)

        # â”€â”€ PM reward: grade-based portfolio performance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        normalized_profit  = float(np.clip((profit + 1.0) / 2.0, 0.0, 1.0))
        normalized_sharpe  = float(np.clip((self._risk.sharpe_ratio() + 2.0) / 4.0, 0.0, 1.0))
        consistency = float(np.mean(np.diff(np.array(self._episode_values)) > 0)) if len(self._episode_values) > 2 else 0.5
        grade = float(compute_grade({
            "profit": normalized_profit,
            "sharpe": normalized_sharpe,
            "drawdown": float(self._risk.max_drawdown),
            "consistency": consistency,
        }))
        pm_reward = (grade - 0.5) * 0.4   # Grade in [0,1] â†’ centered reward
        if self._risk.max_drawdown > 0.20:
            pm_reward -= 0.15              # PM penalized for deep drawdown
        self._pending_rewards[PORTFOLIO_MGR] = float(pm_reward)

        # â”€â”€ RM: shared downside with final portfolio value â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # We ADD to whatever penalty was already set in _step_risk_manager
        rm_pain = min(profit * 0.5, 0.0)   # Only share downside
        self._pending_rewards[RISK_MANAGER] = float(self._pending_rewards.get(RISK_MANAGER, 0.0) + rm_pain)

        # â”€â”€ Termination Check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        terminated = (
            self._current_step >= self.max_steps or
            new_value < self.initial_cash * 0.10   # Blowup condition
        )
        if terminated:
            for ag in self.agents:
                self.terminations[ag] = True

        # Rebuild observations for the next cycle
        self._generate_observations()

        # Update governance log
        gov_record = {
            "step": self._current_step,
            "proposed": {"direction": trade["original_direction"], "size": trade["original_size"]},
            "executed": {"direction": direction, "size": size, "sl": sl_input, "tp": tp_input},
            "interventions": trade["interventions"],
            "was_compliant": len(trade["interventions"]) == 0,
            "rm_message": self._rm_message.tolist(),
            "pm_message": self._pm_message.tolist(),
        }
        self._governance_log.append(gov_record)

        # Expose info for the Trader (most info-rich agent)
        self.infos[TRADER] = {
            "step": self._current_step,
            "portfolio_value": float(new_value),
            "cash": float(self._portfolio.cash),
            "pnl": float(new_value - self.initial_cash),
            "pnl_pct": float(profit),
            "max_drawdown": float(self._risk.max_drawdown),
            "sharpe_ratio": float(self._risk.sharpe_ratio()),
            "grade": grade,
            "governance": gov_record,
            "rewards": dict(self._pending_rewards),
        }
        self.infos[RISK_MANAGER]  = {"step": self._current_step, "drawdown": float(self._risk.max_drawdown)}
        self.infos[PORTFOLIO_MGR] = {"step": self._current_step, "grade": grade}

        self._prev_portfolio_value = new_value
        self._pending_trade = None

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Observation Generation
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _generate_observations(self):
        base_obs = get_observation(self._market, self._portfolio, self._risk, self.ticker)
        self._observations = {
            RISK_MANAGER:  base_obs.copy(),
            PORTFOLIO_MGR: np.concatenate([base_obs, self._rm_message]),
            TRADER:        np.concatenate([base_obs, self._rm_message, self._pm_message]),
        }

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Internal Helpers
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _reset_internal_state(self):
        self._market    = MarketState(prices=self.df, current_step=0)
        self._portfolio = PortfolioState(initial_cash=self.initial_cash, cash=self.initial_cash)
        self._risk      = RiskState(peak_value=self.initial_cash)
        self._current_step = 0

        # Inter-agent messages (start neutral)
        self._rm_message = np.array([0.5, 1.0, 0.0], dtype=np.float32)  # [size_limit=50%, allow=yes, force_reduce=no]
        self._pm_message = np.array([0.5, 0.0], dtype=np.float32)        # [cap_alloc=50%, override_strength=0]
        self._pm_capital_allocation = 0.5
        self._pm_override_strength  = 0.0

        self._pending_trade  = None
        self._pending_rewards = {ag: 0.0 for ag in ALL_AGENTS}
        self._trader_compliance_bonus = 0.0

        self._episode_values  = [self.initial_cash]
        self._episode_rewards = []
        self._governance_log: List[Dict] = []
        self._prev_portfolio_value = self.initial_cash

        # PZ state dictionaries
        self._observations = {ag: np.zeros(self.observation_spaces[ag].shape, dtype=np.float32)
                              for ag in ALL_AGENTS}

    def _accumulate_rewards(self):
        """Move pending rewards into PZ cumulative reward tracking."""
        for ag in self.agents:
            self.rewards[ag] = self._pending_rewards.get(ag, 0.0)
            self._cumulative_rewards[ag] += self.rewards[ag]

    def _execute_trade(
        self, direction: int, size: float, sl: float, tp: float, current_price: float
    ) -> bool:
        """Execute trade on portfolio state. Returns True if a trade was made."""
        traded = False

        if direction == 1:  # BUY / Cover Short
            pos = self._portfolio.positions.get(self.ticker, 0.0)
            if pos < 0:
                # Cover short
                abs_qty = abs(pos)
                cover_cost = abs_qty * current_price * (1 + self.commission)
                margin_return = abs_qty * self._portfolio.avg_costs.get(self.ticker, current_price)
                self._portfolio.cash += margin_return - cover_cost
                self._portfolio.positions[self.ticker] = 0.0
                self._portfolio.avg_costs[self.ticker] = 0.0
                self._portfolio.stop_losses[self.ticker] = None
                self._portfolio.take_profits[self.ticker] = None
                traded = True
            else:
                trade_qty = (self._portfolio.cash * size) / (current_price * (1 + self.commission) + 1e-10)
                if trade_qty > 1e-8:
                    cost = trade_qty * current_price * (1 + self.commission)
                    self._portfolio.cash -= cost
                    prev_qty = pos
                    prev_avg  = self._portfolio.avg_costs.get(self.ticker, 0.0)
                    new_qty  = prev_qty + trade_qty
                    new_avg  = ((prev_qty * prev_avg) + (trade_qty * current_price)) / (new_qty + 1e-10)
                    self._portfolio.positions[self.ticker]   = new_qty
                    self._portfolio.avg_costs[self.ticker]   = new_avg
                    if sl > 0: self._portfolio.stop_losses[self.ticker]  = sl
                    if tp > 0: self._portfolio.take_profits[self.ticker] = tp
                    traded = True

        elif direction == 2:  # SELL / Short
            pos = self._portfolio.positions.get(self.ticker, 0.0)
            if pos > 0:
                sell_qty = min(pos, pos * size)
                if sell_qty > 1e-8:
                    revenue = sell_qty * current_price * (1 - self.commission)
                    self._portfolio.cash += revenue
                    remaining = pos - sell_qty
                    self._portfolio.positions[self.ticker] = max(remaining, 0.0)
                    if remaining <= 1e-8:
                        self._portfolio.avg_costs[self.ticker] = 0.0
                        self._portfolio.stop_losses[self.ticker] = None
                        self._portfolio.take_profits[self.ticker] = None
                    traded = True
            else:
                margin = self._portfolio.cash * size
                short_qty = margin / (current_price * (1 + self.commission) + 1e-10)
                if short_qty > 1e-8:
                    self._portfolio.cash -= short_qty * current_price
                    prev_qty  = abs(pos)
                    prev_avg  = self._portfolio.avg_costs.get(self.ticker, 0.0)
                    new_qty   = prev_qty + short_qty
                    new_avg   = ((prev_qty * prev_avg) + (short_qty * current_price)) / (new_qty + 1e-10)
                    self._portfolio.positions[self.ticker]   = -new_qty
                    self._portfolio.avg_costs[self.ticker]   = new_avg
                    if sl > 0: self._portfolio.stop_losses[self.ticker]  = sl
                    if tp > 0: self._portfolio.take_profits[self.ticker] = tp
                    traded = True

        if traded:
            self._risk.trade_count += 1
        return traded

    def _check_sl_tp(self, current_price: float):
        """Check and execute SL/TP orders."""
        ticker  = self.ticker
        pos_qty = self._portfolio.positions.get(ticker, 0.0)
        sl      = self._portfolio.stop_losses.get(ticker)
        tp      = self._portfolio.take_profits.get(ticker)
        if abs(pos_qty) < 1e-8:
            return

        hit = False
        if pos_qty > 0:
            if sl and current_price <= sl: hit = True
            if tp and current_price >= tp: hit = True
            if hit:
                revenue = pos_qty * current_price * (1 - self.commission)
                self._portfolio.cash += revenue
                self._portfolio.positions[ticker] = 0.0
                self._portfolio.avg_costs[ticker] = 0.0
                self._portfolio.stop_losses[ticker] = None
                self._portfolio.take_profits[ticker] = None
                self._risk.trade_count += 1
        elif pos_qty < 0:
            abs_qty = abs(pos_qty)
            if sl and current_price >= sl: hit = True
            if tp and current_price <= tp: hit = True
            if hit:
                avg_cost   = self._portfolio.avg_costs.get(ticker, current_price)
                cover_cost = abs_qty * current_price * (1 + self.commission)
                margin_ret = abs_qty * avg_cost
                self._portfolio.cash += margin_ret - cover_cost
                self._portfolio.positions[ticker] = 0.0
                self._portfolio.avg_costs[ticker] = 0.0
                self._portfolio.stop_losses[ticker] = None
                self._portfolio.take_profits[ticker] = None
                self._risk.trade_count += 1

    def _make_dummy_data(self, n: int = 500, difficulty: str = "hard") -> pd.DataFrame:
        """Delegate to TradingEnv's proven synthetic data generator."""
        from env.trading_env import TradingEnv
        tmp = TradingEnv.__new__(TradingEnv)
        return tmp._generate_market_data(n=n, difficulty=difficulty)

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Convenience
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @functools.lru_cache(maxsize=None)
    def _obs_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def _act_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def state(self) -> Dict:
        """Return the full shared environment state (for visualization)."""
        price = self._market.current_price()
        return {
            "step":            self._current_step,
            "price":           float(price),
            "portfolio_value": float(self._portfolio.total_value(price, self.ticker)),
            "cash":            float(self._portfolio.cash),
            "positions":       {k: float(v) for k, v in self._portfolio.positions.items()},
            "max_drawdown":    float(self._risk.max_drawdown),
            "sharpe_ratio":    float(self._risk.sharpe_ratio()),
            "trade_count":     self._risk.trade_count,
            "rm_message":      self._rm_message.tolist(),
            "pm_message":      self._pm_message.tolist(),
            "governance_log":  self._governance_log[-10:],
        }
