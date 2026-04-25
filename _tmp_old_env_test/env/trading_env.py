"""
Multi-Agent Trading Environment built on Gymnasium.
Integrates MarketState, PortfolioState, RiskState with the agent interaction loop.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from openenv.env import Env as OpenEnvBase

from env.state import MarketState, PortfolioState, RiskState, get_observation
from env.reward import compute_raw_reward, normalize_reward, compute_grade
from utils.indicators import compute_indicators


class TradingEnv(OpenEnvBase, gym.Env):
    """
    A multi-agent RL trading environment.

    Observation: concatenated normalized features from market, portfolio, and risk state.
    Action: Dict with 'direction' (0=Hold, 1=Buy, 2=Sell), 'size' [0, 1], 'sl' (price), 'tp' (price).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        initial_cash: float = 100_000.0,
        ticker: str = "default",
        commission: float = 0.001,
        reward_weights: Optional[Dict[str, float]] = None,
        max_steps: Optional[int] = None,
        difficulty: str = "hard",
    ):
        """
        Args:
            df: OHLCV DataFrame.
            initial_cash: Starting cash.
            ticker: Asset identifier.
            commission: Trading commission.
            reward_weights: Custom weights.
            max_steps: Max steps.
            difficulty: 'easy', 'medium', or 'hard' for curriculum learning.
        """
        self.difficulty = difficulty
        # Data setup
        if df is None:
            df = self._make_dummy_data(difficulty=self.difficulty)
        self.raw_df = df.copy()
        self.df = compute_indicators(df)
        self.ticker = ticker
        self.initial_cash = initial_cash
        self.commission = commission
        self.reward_weights = reward_weights
        self.max_steps = max_steps or (len(self.df) - 1)

        # State objects
        self.market = MarketState(prices=self.df)
        self.portfolio = PortfolioState(initial_cash=initial_cash, cash=initial_cash)
        self.risk = RiskState(peak_value=initial_cash)

        # Observation and action spaces
        obs_size = self.market.feature_size + self.portfolio.feature_size + self.risk.feature_size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Dict({
            "direction": spaces.Discrete(3),  # 0=Hold, 1=Buy, 2=Sell
            "size": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "sl": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "tp": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
        })
        OpenEnvBase.__init__(
            self,
            name="TradingEnv",
            state_space=self.observation_space,
            action_space=self.action_space,
            episode_max_length=self.max_steps,
        )

        # Episode tracking
        self.current_step = 0
        self.done = False
        self.episode_rewards = []
        self.episode_values = []
        self.margin_call_threshold = 0.5  # Force-close short if loss > 50% of initial cash

        # Governance tracking
        self.governance_log: list = []  # Per-step governance records
        self.episode_interventions = 0  # Total interventions this episode
        self.episode_compliant_actions = 0  # Actions that passed without intervention

    def _make_dummy_data(self, n=500, difficulty="hard") -> pd.DataFrame:
        """
        Generate synthetic price data with realistic market regimes.
        Easy: Trending (bull_steady, recovery).
        Medium: Sideways, mean-reverting, volatile bull.
        Hard: Crashes, bubble pops, bear markets + regime switching.
        """
        return self._generate_market_data(n=n, difficulty=difficulty)

    def _generate_market_data(
        self,
        n: int = 500,
        difficulty: str = "hard",
    ) -> pd.DataFrame:
        """Multi-regime synthetic market data generator.

        Supports 8 realistic market regimes with calibrated parameters,
        jump diffusion, fat tails, and volume spikes.
        """
        rng = np.random.default_rng()
        dt = 1 / (24 * 365)  # Hourly steps

        # ── Regime Definitions ──
        regimes = {
            "bull_steady":     {"mu": 0.30, "sigma": 0.08, "jump_prob": 0.0,  "jump_std": 0.0,  "df": 30},
            "bull_volatile":   {"mu": 0.40, "sigma": 0.35, "jump_prob": 0.02, "jump_std": 0.04, "df": 5},
            "bear_steady":     {"mu": -0.20, "sigma": 0.15, "jump_prob": 0.01, "jump_std": 0.03, "df": 8},
            "crash":           {"mu": -0.80, "sigma": 0.60, "jump_prob": 0.05, "jump_std": 0.10, "df": 3},
            "sideways_choppy": {"mu": 0.0,  "sigma": 0.25, "jump_prob": 0.01, "jump_std": 0.03, "df": 6},
            "mean_revert":     {"mu": 0.0,  "sigma": 0.12, "jump_prob": 0.0,  "jump_std": 0.0,  "df": 15},
            "bubble_pop":      {"mu": 1.00, "sigma": 0.50, "jump_prob": 0.0,  "jump_std": 0.0,  "df": 4},
            "recovery":        {"mu": 0.50, "sigma": 0.20, "jump_prob": 0.01, "jump_std": 0.02, "df": 10},
        }

        # ── Difficulty → regime selection ──
        if difficulty == "easy":
            regime_pool = ["bull_steady", "recovery"]
        elif difficulty == "medium":
            regime_pool = ["sideways_choppy", "mean_revert", "bull_volatile", "recovery"]
        else:  # hard
            regime_pool = list(regimes.keys())

        # ── Regime switching: split episode into 1-3 regimes ──
        if difficulty == "hard":
            num_regimes = rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
        elif difficulty == "medium":
            num_regimes = rng.choice([1, 2], p=[0.5, 0.5])
        else:
            num_regimes = 1

        chosen_regimes = rng.choice(regime_pool, size=num_regimes)
        splits = sorted(rng.integers(50, n - 50, size=max(0, num_regimes - 1)))
        boundaries = [0] + list(splits) + [n]

        # ── Generate returns per regime segment ──
        all_returns = np.zeros(n)
        for i, regime_name in enumerate(chosen_regimes):
            start_idx, end_idx = boundaries[i], boundaries[i + 1]
            seg_len = end_idx - start_idx
            params = regimes[regime_name]

            # Fat-tailed noise via Student-t distribution
            noise = rng.standard_t(df=params["df"], size=seg_len) * params["sigma"] * np.sqrt(dt)

            # Drift
            drift = (params["mu"] - 0.5 * params["sigma"] ** 2) * dt

            # Jump diffusion
            jump_mask = rng.random(seg_len) < params["jump_prob"]
            jumps = jump_mask * rng.normal(0, params["jump_std"], seg_len)

            # Special handling for bubble_pop: parabolic rise then crash
            if regime_name == "bubble_pop":
                midpoint = seg_len // 2
                # First half: parabolic rise (accelerating drift)
                accel = np.linspace(1.0, 3.0, midpoint)
                noise[:midpoint] *= 0.5  # Lower noise during rise
                drift_arr = np.full(seg_len, drift)
                drift_arr[:midpoint] *= accel
                # Second half: crash
                drift_arr[midpoint:] = -abs(drift) * 2.5
                noise[midpoint:] *= 2.0  # Higher noise during crash
                jumps[midpoint:] += rng.normal(-0.05, 0.08, seg_len - midpoint) * (rng.random(seg_len - midpoint) > 0.9)
                all_returns[start_idx:end_idx] = drift_arr + noise + jumps
            elif regime_name == "mean_revert":
                # Mean-reverting overlay: pull returns toward zero
                raw = drift + noise + jumps
                cumulative = np.cumsum(raw)
                reversion = -0.05 * cumulative * dt
                all_returns[start_idx:end_idx] = raw + reversion
            else:
                all_returns[start_idx:end_idx] = drift + noise + jumps

        # ── Convert returns to prices ──
        s0 = 50000.0
        prices = s0 * np.exp(np.cumsum(all_returns))

        # ── Volume: correlated with absolute returns (spikes on big moves) ──
        base_volume = rng.integers(100_000_000, 500_000_000, n).astype(float)
        abs_rets = np.abs(all_returns)
        vol_multiplier = 1.0 + 10.0 * (abs_rets / (abs_rets.max() + 1e-10))
        volume = (base_volume * vol_multiplier).astype(int)

        # ── Build OHLCV ──
        intrabar_noise = rng.normal(0, 0.003, n)
        high_noise = np.abs(rng.normal(0, 0.008, n))
        low_noise = np.abs(rng.normal(0, 0.008, n))

        df = pd.DataFrame({
            "open": prices * (1 + intrabar_noise),
            "high": prices * (1 + high_noise),
            "low": prices * (1 - low_noise),
            "close": prices,
            "volume": volume,
        }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

        df.index.name = "date"
        return df

    def _make_dummy_data_from_profile(
        self,
        n: int = 500,
        difficulty: str = "hard",
        mu: float | None = None,
        sigma: float | None = None,
    ) -> pd.DataFrame:
        """Generate data with explicit mu/sigma (for backward compatibility)."""
        if mu is not None and sigma is not None:
            rng = np.random.default_rng()
            dt = 1 / (24 * 365)
            Z = rng.standard_normal(n)
            returns = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            s0 = 50000.0
            prices = s0 * np.cumprod(returns)
            df = pd.DataFrame({
                "open": prices * (1 + np.random.randn(n) * 0.005),
                "high": prices * (1 + abs(np.random.randn(n) * 0.01)),
                "low": prices * (1 - abs(np.random.randn(n) * 0.01)),
                "close": prices,
                "volume": np.random.randint(100_000_000, 1_000_000_000, n),
            }, index=pd.date_range("2024-01-01", periods=n, freq="h"))
            df.index.name = "date"
            return df
        return self._generate_market_data(n=n, difficulty=difficulty)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.done = False
        self.market = MarketState(prices=self.df, current_step=0)
        self.portfolio = PortfolioState(
            initial_cash=self.initial_cash, cash=self.initial_cash
        )
        self.risk = RiskState(peak_value=self.initial_cash)
        self.episode_rewards = []
        self.episode_values = [self.initial_cash]
        self.governance_log = []
        self.episode_interventions = 0
        self.episode_compliant_actions = 0

        obs = get_observation(self.market, self.portfolio, self.risk, self.ticker)
        info = self._get_info()
        return obs, info

    def _check_sl_tp(self, current_price: float):
        """Check if any open position hit SL or TP, and apply trailing updates.
        
        Long positions: SL triggers when price falls to SL; TP when price rises to TP.
        Short positions: SL triggers when price rises to SL; TP when price falls to TP.
        """
        atr = self.df["atr"].iloc[self.current_step]
        
        for ticker, position_qty in list(self.portfolio.positions.items()):
            if abs(position_qty) < 1e-8:
                continue

            sl = self.portfolio.stop_losses.get(ticker)
            tp = self.portfolio.take_profits.get(ticker)
            
            # --- 1. ATR Trailing Stop Update ---
            if sl is not None:
                if position_qty > 0:  # Long
                    trailing_level = current_price - (atr * 2.0)
                    if trailing_level > sl and current_price > self.portfolio.avg_costs.get(ticker, current_price):
                        self.portfolio.stop_losses[ticker] = trailing_level
                elif position_qty < 0:  # Short
                    trailing_level = current_price + (atr * 2.0)
                    if trailing_level < sl and current_price < self.portfolio.avg_costs.get(ticker, current_price):
                        self.portfolio.stop_losses[ticker] = trailing_level
            # -----------------------------------

        exit_triggered = False
        exit_price = current_price
        reason = ""

        # Only process SL/TP for the primary ticker to maintain original logic
        qty = self.portfolio.positions.get(self.ticker, 0.0)
        sl = self.portfolio.stop_losses.get(self.ticker)
        tp = self.portfolio.take_profits.get(self.ticker)

        if qty > 0:  # Long position
            if sl is not None and current_price <= sl:
                exit_triggered = True
                exit_price = sl
                reason = "stop_loss"
            elif tp is not None and current_price >= tp:
                exit_triggered = True
                exit_price = tp
                reason = "take_profit"

            if exit_triggered:
                revenue = qty * exit_price * (1 - self.commission)
                self.portfolio.cash += revenue
                self.portfolio.positions[self.ticker] = 0.0
                self.portfolio.avg_costs[self.ticker] = 0.0
                self.portfolio.stop_losses[self.ticker] = None
                self.portfolio.take_profits[self.ticker] = None
                self.portfolio.trade_history.append({
                    "step": self.current_step,
                    "action": "sell",
                    "ticker": self.ticker,
                    "price": exit_price,
                    "quantity": qty,
                    "reason": reason
                })
                self.risk.trade_count += 1
                return True

        elif qty < 0:  # Short position
            abs_qty = abs(qty)
            if sl is not None and current_price >= sl:
                exit_triggered = True
                exit_price = sl
                reason = "stop_loss"
            elif tp is not None and current_price <= tp:
                exit_triggered = True
                exit_price = tp
                reason = "take_profit"

            if exit_triggered:
                # Cover the short: buy back at exit_price
                avg_cost = self.portfolio.avg_costs.get(self.ticker, exit_price)
                cover_cost = abs_qty * exit_price * (1 + self.commission)
                # Return margin (original short proceeds)
                margin_return = abs_qty * avg_cost
                self.portfolio.cash += margin_return - cover_cost
                self.portfolio.positions[self.ticker] = 0.0
                self.portfolio.avg_costs[self.ticker] = 0.0
                self.portfolio.stop_losses[self.ticker] = None
                self.portfolio.take_profits[self.ticker] = None
                self.portfolio.trade_durations[self.ticker] = 0
                self.portfolio.trade_history.append({
                    "step": self.current_step,
                    "action": "cover",
                    "ticker": self.ticker,
                    "price": exit_price,
                    "quantity": abs_qty,
                    "reason": reason
                })
                self.risk.trade_count += 1
                return True

        return False

    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the multi-agent governance environment.

        The environment acts as a governance framework: the agent proposes
        an action, and internal Risk/Compliance agents may modify or
        override it.  Every intervention is logged so the agent can learn
        to self-regulate (propose compliant actions that pass governance
        without modification).
        """
        if self.done:
            obs = get_observation(self.market, self.portfolio, self.risk, self.ticker)
            return obs, 0.0, True, False, self._get_info()

        current_price = self.market.current_price()
        prev_value = self.portfolio.total_value(current_price, self.ticker)
        
        # 1. Check SL/TP before executing new action
        sl_tp_hit = self._check_sl_tp(current_price)

        # 2. Extract action components
        direction = int(action["direction"])
        size = action.get("size", [0.0])
        if hasattr(size, "__len__"):
            size = float(size[0])
        else:
            size = float(size)
        size = float(np.clip(size, 0.0, 1.0))
        
        sl_input = float(action["sl"][0]) if "sl" in action and hasattr(action["sl"], '__len__') else float(action.get("sl", 0.0))
        tp_input = float(action["tp"][0]) if "tp" in action and hasattr(action["tp"], '__len__') else float(action.get("tp", 0.0))

        # ═══════════════════════════════════════════════════
        #  GOVERNANCE FRAMEWORK — track all interventions
        # ═══════════════════════════════════════════════════
        original_direction = direction
        original_size = size
        original_sl = sl_input
        original_tp = tp_input
        interventions: list = []

        # --- 2. Market Impact & Funding Cost ---
        volatility = self.df["volatility"].iloc[self.current_step]
        # Slippage scales with trade size and current market volatility
        effective_commission = self.commission + (size * volatility * 0.25)
        
        # Funding cost: small fee deducted for holding shorts overnight/per step
        time_penalty = 0.0
        for ticker, pos_qty in list(self.portfolio.positions.items()):
            if abs(pos_qty) > 1e-8:
                # Increment holding duration
                dur = self.portfolio.trade_durations.get(ticker, 0) + 1
                self.portfolio.trade_durations[ticker] = dur
                
                # Deduct borrow fee for shorts
                if pos_qty < 0:
                    borrow_fee = abs(pos_qty) * current_price * 0.00005  # 0.5 bps per tick
                    self.portfolio.cash -= borrow_fee
                    
                # Time decay penalty factor for RL reward (capital velocity)
                time_penalty += (dur * 0.0001)
        # ---------------------------------------

        # ═══════════════════════════════════════════════════
        # GOVERNANCE ENFORCEMENT — Risk Manager Agent
        # ═══════════════════════════════════════════════════
        # 1. Auto-SL: If no SL provided, set one at 2% from entry
        DEFAULT_SL_RATIO = 0.02
        if direction != 0 and sl_input <= 0:
            if direction == 1:  # BUY
                sl_input = current_price * (1.0 - DEFAULT_SL_RATIO)
            elif direction == 2:  # SHORT
                sl_input = current_price * (1.0 + DEFAULT_SL_RATIO)
            interventions.append({
                "agent": "RiskManager",
                "type": "auto_stop_loss",
                "reason": "No stop-loss provided — governance auto-set 2% SL",
                "enforced_sl": float(sl_input),
            })

        # 2. Auto-TP: If no TP provided, set one at 2:1 RRR
        if direction != 0 and tp_input <= 0 and sl_input > 0:
            sl_dist = abs(current_price - sl_input)
            if direction == 1:
                tp_input = current_price + sl_dist * 2.0
            elif direction == 2:
                tp_input = current_price - sl_dist * 2.0
            interventions.append({
                "agent": "RiskManager",
                "type": "auto_take_profit",
                "reason": "No take-profit provided — governance auto-set 2:1 RRR",
                "enforced_tp": float(tp_input),
            })

        # 3. Hard 1% risk cap: clamp position size so max loss ≤ 1% of portfolio
        # Only apply risk cap if OPENING or ADDING to a position
        position_qty = self.portfolio.positions.get(self.ticker, 0.0)
        is_opening = (direction == 1 and position_qty >= 0) or (direction == 2 and position_qty <= 0)

        HARD_RISK_CAP = 0.01
        if direction != 0 and sl_input > 0 and is_opening:
            portfolio_value = self.portfolio.total_value(current_price, self.ticker)
            sl_distance = abs(current_price - sl_input)
            if sl_distance > 1e-10:
                max_loss = portfolio_value * HARD_RISK_CAP
                max_qty = max_loss / sl_distance
                max_size = (max_qty * current_price) / (portfolio_value + 1e-10)
                if size > max_size:
                    interventions.append({
                        "agent": "RiskManager",
                        "type": "size_clamp",
                        "original_size": float(size),
                        "enforced_size": float(max_size),
                        "reason": f"Position size {size:.2%} exceeded Kelly 1% risk cap — clamped to {max_size:.2%}",
                    })
                size = min(size, max_size)

        traded = False
        step_trade_count = int(sl_tp_hit)

        if direction == 1:  # BUY
            position_qty = self.portfolio.positions.get(self.ticker, 0.0)
            
            if position_qty < 0:
                # ── Cover existing short position ──
                abs_qty = abs(position_qty)
                cover_qty = min(abs_qty, abs_qty * size) if size < 1.0 else abs_qty
                avg_cost = self.portfolio.avg_costs.get(self.ticker, current_price)
                cover_cost = cover_qty * current_price * (1 + self.commission)
                margin_return = cover_qty * avg_cost
                self.portfolio.cash += margin_return - cover_cost
                remaining = position_qty + cover_qty  # Moves toward 0
                if abs(remaining) <= 1e-8:
                    remaining = 0.0
                    self.portfolio.avg_costs[self.ticker] = 0.0
                    self.portfolio.stop_losses[self.ticker] = None
                    self.portfolio.take_profits[self.ticker] = None
                    self.portfolio.trade_durations[self.ticker] = 0
                self.portfolio.positions[self.ticker] = remaining
                self.portfolio.trade_history.append({
                    "step": self.current_step,
                    "action": "cover",
                    "ticker": self.ticker,
                    "price": current_price,
                    "quantity": cover_qty,
                })
                traded = True
            else:
                # ── Open/add to long position ──
                trade_qty = (self.portfolio.cash * size) / (current_price * (1 + self.commission) + 1e-10)
                if trade_qty > 1e-8:
                    cost = trade_qty * current_price * (1 + self.commission)
                    self.portfolio.cash -= cost
                    prev_qty = position_qty
                    prev_avg_cost = self.portfolio.avg_costs.get(self.ticker, 0.0)
                    new_qty = prev_qty + trade_qty
                    new_avg_cost = (
                        ((prev_qty * prev_avg_cost) + (trade_qty * current_price)) / (new_qty + 1e-10)
                    )
                    self.portfolio.positions[self.ticker] = new_qty
                    self.portfolio.avg_costs[self.ticker] = new_avg_cost
                    
                    # Update SL/TP for the position
                    if sl_input > 0: self.portfolio.stop_losses[self.ticker] = sl_input
                    if tp_input > 0: self.portfolio.take_profits[self.ticker] = tp_input
                    
                    self.portfolio.trade_history.append({
                        "step": self.current_step,
                        "action": "buy",
                        "ticker": self.ticker,
                        "price": current_price,
                        "quantity": trade_qty,
                    })
                    traded = True

        elif direction == 2:  # SELL / SHORT
            position_qty = self.portfolio.positions.get(self.ticker, 0.0)
            
            if position_qty > 0:
                # ── Close/reduce existing long position ──
                sell_qty = min(position_qty, position_qty * size)
                if sell_qty > 1e-8:
                    revenue = sell_qty * current_price * (1 - self.commission)
                    self.portfolio.cash += revenue
                    remaining_qty = position_qty - sell_qty
                    if remaining_qty <= 1e-8:
                        remaining_qty = 0.0
                    self.portfolio.positions[self.ticker] = remaining_qty
                    
                    # Clear SL/TP if position closed
                    if remaining_qty == 0.0:
                        self.portfolio.avg_costs[self.ticker] = 0.0
                        self.portfolio.stop_losses[self.ticker] = None
                        self.portfolio.take_profits[self.ticker] = None

                    self.portfolio.trade_history.append({
                        "step": self.current_step,
                        "action": "sell",
                        "ticker": self.ticker,
                        "price": current_price,
                        "quantity": sell_qty,
                    })
                    traded = True
            else:
                # ── Open/add to short position ──
                # Margin required: qty * price locked as collateral
                margin_available = self.portfolio.cash * size
                short_qty = margin_available / (current_price * (1 + self.commission) + 1e-10)
                if short_qty > 1e-8:
                    margin_cost = short_qty * current_price  # Lock as collateral
                    self.portfolio.cash -= margin_cost
                    prev_qty = abs(position_qty)  # existing short size
                    prev_avg_cost = self.portfolio.avg_costs.get(self.ticker, 0.0)
                    new_qty = prev_qty + short_qty
                    new_avg_cost = (
                        ((prev_qty * prev_avg_cost) + (short_qty * current_price)) / (new_qty + 1e-10)
                    )
                    self.portfolio.positions[self.ticker] = -(new_qty)  # Negative = short
                    self.portfolio.avg_costs[self.ticker] = new_avg_cost
                    
                    # SL/TP for shorts: SL above entry, TP below entry
                    if sl_input > 0: self.portfolio.stop_losses[self.ticker] = sl_input
                    if tp_input > 0: self.portfolio.take_profits[self.ticker] = tp_input
                    
                    self.portfolio.trade_history.append({
                        "step": self.current_step,
                        "action": "short",
                        "ticker": self.ticker,
                        "price": current_price,
                        "quantity": short_qty,
                    })
                    traded = True

        if traded:
            self.risk.trade_count += 1
            step_trade_count += 1

        # Advance market
        self.current_step += 1
        self.market.current_step = self.current_step

        # Update portfolio and risk
        new_price = self.market.current_price()
        new_value = self.portfolio.total_value(new_price, self.ticker)
        self.risk.update(new_value)
        self.episode_values.append(new_value)

        # Compute reward
        profit = (new_value - prev_value) / (self.initial_cash + 1e-10)
        price_trend = (new_price - current_price) / (current_price + 1e-10)
        raw_r = compute_raw_reward(
            profit=profit,
            drawdown=self.risk.current_drawdown,
            volatility=self.risk.return_volatility(),
            sharpe=self.risk.sharpe_ratio(),
            trade_count=step_trade_count,
            weights=self.reward_weights,
            direction=direction,
            price_trend=price_trend,
        )
        
        # Combine raw profit reward with our multiple behavior signals
        step_reward = raw_r
        
        # Apply Time Penalty
        step_reward -= time_penalty

        # ═══════════════════════════════════════════════════
        # GOVERNANCE REWARD SIGNAL
        # ═══════════════════════════════════════════════════
        # Bonus for self-regulation: agent proposed compliant action
        # Penalty for triggering governance interventions
        n_interventions = len(interventions)
        if n_interventions == 0 and direction != 0:
            step_reward += 0.15  # Compliance bonus
            self.episode_compliant_actions += 1
        elif n_interventions > 0:
            step_reward -= 0.05 * n_interventions  # Per-intervention penalty
            self.episode_interventions += n_interventions

        reward = normalize_reward(step_reward)
        self.episode_rewards.append(reward)

        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = False
        if new_value < self.initial_cash * 0.1:
            terminated = True
        # Margin call: force-close short if unrealized loss exceeds threshold
        position_qty = self.portfolio.positions.get(self.ticker, 0.0)
        if position_qty < 0:
            short_pnl = self.portfolio.unrealized_pnl(new_price, self.ticker)
            if short_pnl < -(self.initial_cash * self.margin_call_threshold):
                # Force cover the short
                abs_qty = abs(position_qty)
                avg_cost = self.portfolio.avg_costs.get(self.ticker, new_price)
                cover_cost = abs_qty * new_price * (1 + self.commission)
                margin_return = abs_qty * avg_cost
                self.portfolio.cash += margin_return - cover_cost
                self.portfolio.positions[self.ticker] = 0.0
                self.portfolio.avg_costs[self.ticker] = 0.0
                self.portfolio.stop_losses[self.ticker] = None
                self.portfolio.take_profits[self.ticker] = None
                self.portfolio.trade_history.append({
                    "step": self.current_step,
                    "action": "margin_call",
                    "ticker": self.ticker,
                    "price": new_price,
                    "quantity": abs_qty,
                    "reason": "margin_call",
                })
                self.risk.trade_count += 1
                interventions.append({
                    "agent": "ComplianceOfficer",
                    "type": "margin_call",
                    "reason": f"Unrealized short loss exceeded {self.margin_call_threshold:.0%} threshold — forced liquidation",
                })
                self.episode_interventions += 1
                terminated = True
        if terminated:
            self.done = True

        # ═══════════════════════════════════════════════════
        # BUILD GOVERNANCE RECORD
        # ═══════════════════════════════════════════════════
        governance_record = {
            "step": self.current_step,
            "proposed": {
                "direction": original_direction,
                "size": original_size,
                "sl": original_sl,
                "tp": original_tp,
            },
            "executed": {
                "direction": direction,
                "size": size,
                "sl": sl_input,
                "tp": tp_input,
            },
            "interventions": interventions,
            "was_compliant": len(interventions) == 0,
        }
        self.governance_log.append(governance_record)

        obs = get_observation(self.market, self.portfolio, self.risk, self.ticker)
        info = self._get_info()
        info["governance"] = governance_record
        info["governance_stats"] = {
            "episode_interventions": self.episode_interventions,
            "episode_compliant_actions": self.episode_compliant_actions,
            "compliance_rate": (
                self.episode_compliant_actions / max(self.current_step, 1)
            ),
        }
        return obs, reward, terminated, truncated, info

    def _get_info(self) -> dict:
        """Return diagnostic info dict."""
        current_price = self.market.current_price()
        total_value = self.portfolio.total_value(current_price, self.ticker)

        # Compute grade metrics
        profit_ratio = (total_value - self.initial_cash) / (self.initial_cash + 1e-10)
        normalized_profit = np.clip((profit_ratio + 1.0) / 2.0, 0.0, 1.0)
        normalized_sharpe = np.clip((self.risk.sharpe_ratio() + 2.0) / 4.0, 0.0, 1.0)

        if len(self.episode_values) > 1:
            vals = np.array(self.episode_values)
            returns = np.diff(vals) / (vals[:-1] + 1e-10)
            consistency = np.mean(returns > 0)
        else:
            consistency = 0.5

        grade = compute_grade({
            "profit": float(normalized_profit),
            "sharpe": float(normalized_sharpe),
            "drawdown": float(self.risk.max_drawdown),
            "consistency": float(consistency),
        })

        return {
            "step": self.current_step,
            "portfolio_value": float(total_value),
            "cash": float(self.portfolio.cash),
            "positions": {ticker: float(qty) for ticker, qty in self.portfolio.positions.items()},
            "pnl": float(total_value - self.initial_cash),
            "pnl_pct": float(profit_ratio),
            "max_drawdown": float(self.risk.max_drawdown),
            "sharpe_ratio": float(self.risk.sharpe_ratio()),
            "normalized_profit": float(normalized_profit),
            "normalized_sharpe": float(normalized_sharpe),
            "normalized_drawdown_inverse": float(1.0 - np.clip(self.risk.max_drawdown, 0.0, 1.0)),
            "consistency": float(consistency),
            "trade_count": self.risk.trade_count,
            "grade": float(grade),
            "episode_reward_sum": float(sum(self.episode_rewards)) if self.episode_rewards else 0.0,
            "episode_reward_mean": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
        }

    def sample_action(self) -> dict:
        """Sample a random action (convenience method)."""
        action_space: Any = self.action_space
        return {
            "direction": action_space["direction"].sample(),
            "size": action_space["size"].sample(),
            "sl": np.array([0.0], dtype=np.float32),
            "tp": np.array([0.0], dtype=np.float32),
        }
