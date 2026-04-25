"""
State management for the trading environment.
Defines MarketState, PortfolioState, RiskState, and observation construction.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class MarketState:
    """Holds current market data and technical indicators for the observation."""

    prices: pd.DataFrame  # OHLCV + indicators dataframe
    current_step: int = 0

    def current_row(self) -> pd.Series:
        return self.prices.iloc[self.current_step]

    def current_price(self) -> float:
        return float(self.prices.iloc[self.current_step]["close"])

    def observation_vector(self) -> np.ndarray:
        """Return a normalized vector of market features."""
        row = self.current_row()
        features = []

        # Normalized price features (relative to close)
        close = row["close"]
        for col in ["open", "high", "low", "close"]:
            features.append(row[col] / (close + 1e-10))

        # Volume — log-normalize
        features.append(np.log1p(row["volume"]) / 20.0)

        # RSI normalized to [0, 1]
        features.append(row["rsi"] / 100.0)

        # EMAs relative to close
        features.append(row["ema_20"] / (close + 1e-10))
        features.append(row["ema_50"] / (close + 1e-10))

        # MACD features normalized
        features.append(np.tanh(row["macd"] / (close + 1e-10) * 100))
        features.append(np.tanh(row["macd_signal"] / (close + 1e-10) * 100))
        features.append(np.tanh(row["macd_hist"] / (close + 1e-10) * 100))

        # Bollinger Band position: where is price within bands
        bb_range = row["bb_upper"] - row["bb_lower"] + 1e-10
        features.append((close - row["bb_lower"]) / bb_range)

        # Volatility — clip to reasonable range
        features.append(min(row["volatility"] * 100, 1.0))

        # ATR relative to close (normalized)
        features.append(row["atr"] / (close + 1e-10))

        return np.array(features, dtype=np.float32)

    @property
    def feature_size(self) -> int:
        return 14  # Number of features in observation_vector


@dataclass
class PortfolioState:
    """Tracks portfolio holdings and cash."""

    initial_cash: float = 100_000.0
    cash: float = 100_000.0
    positions: Dict[str, float] = field(default_factory=dict)  # ticker -> quantity
    avg_costs: Dict[str, float] = field(default_factory=dict)  # ticker -> average entry price
    trade_durations: Dict[str, int] = field(default_factory=dict) # ticker -> steps held
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Professional risk management: Stop Loss and Take Profit
    # Format: {ticker: price}
    stop_losses: Dict[str, "Optional[float]"] = field(default_factory=dict)
    take_profits: Dict[str, "Optional[float]"] = field(default_factory=dict)

    def reset(self):
        self.cash = self.initial_cash
        self.positions = {}
        self.avg_costs = {}
        self.trade_history = []
        self.stop_losses = {}
        self.take_profits = {}

    def total_value(self, current_price: float, ticker: str = "default") -> float:
        """Total portfolio value = cash + position mark-to-market.

        For longs:  value = cash + qty * price
        For shorts: value = cash + qty * (avg_cost - price) + qty * avg_cost
                  which simplifies to cash + qty * (2 * avg_cost - price)
        But since qty is negative for shorts, we use the unified formula:
          value = cash + qty * price  (for longs)
          value = cash + margin_held + unrealized_pnl  (for shorts)
        """
        position_qty = self.positions.get(ticker, 0.0)
        if position_qty >= 0:
            # Long position
            return self.cash + position_qty * current_price
        else:
            # Short position: cash already reduced by margin (|qty| * avg_cost)
            # Unrealized P&L = |qty| * (avg_cost - current_price)
            avg_cost = self.avg_costs.get(ticker, current_price)
            unrealized = abs(position_qty) * (avg_cost - current_price)
            return self.cash + unrealized

    def unrealized_pnl(self, current_price: float, ticker: str = "default") -> float:
        """
        Unrealized profit/loss from open positions using tracked average cost.
        Supports both long (positive qty) and short (negative qty) positions.
        """
        position_qty = self.positions.get(ticker, 0.0)
        if abs(position_qty) < 1e-10:
            return 0.0

        avg_entry = self.avg_costs.get(ticker, 0.0)
        if position_qty > 0:
            # Long: profit when price goes up
            return position_qty * (current_price - avg_entry)
        else:
            # Short: profit when price goes down
            return abs(position_qty) * (avg_entry - current_price)

    def observation_vector(self, current_price: float, ticker: str = "default") -> np.ndarray:
        """Return normalized portfolio features."""
        total_val = self.total_value(current_price, ticker)
        position_qty = self.positions.get(ticker, 0.0)
        long_value = max(position_qty, 0.0) * current_price
        short_value = abs(min(position_qty, 0.0)) * current_price

        features = [
            self.cash / (self.initial_cash + 1e-10),       # cash ratio
            long_value / (total_val + 1e-10),              # long exposure ratio
            total_val / (self.initial_cash + 1e-10),       # portfolio return ratio
            np.tanh(self.unrealized_pnl(current_price, ticker) / (self.initial_cash + 1e-10) * 10),  # normalized PnL
            short_value / (self.initial_cash + 1e-10),     # short exposure ratio
        ]
        return np.array(features, dtype=np.float32)

    @property
    def feature_size(self) -> int:
        return 5


@dataclass
class RiskState:
    """Tracks risk metrics: drawdown, exposure."""

    peak_value: float = 100_000.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    return_history: List[float] = field(default_factory=list)
    trade_count: int = 0

    def reset(self, initial_value: float = 100_000.0):
        self.peak_value = initial_value
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.return_history = []
        self.trade_count = 0

    def update(self, portfolio_value: float):
        """Update risk metrics with latest portfolio value."""
        # Track returns
        if self.return_history:
            prev = self.return_history[-1]
            ret = (portfolio_value - prev) / (prev + 1e-10)
        else:
            ret = 0.0
        self.return_history.append(portfolio_value)

        # Update peak and drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        self.current_drawdown = (self.peak_value - portfolio_value) / (self.peak_value + 1e-10)
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Compute Sharpe ratio from return history."""
        if len(self.return_history) < 2:
            return 0.0
        values = np.array(self.return_history)
        returns = np.diff(values) / (values[:-1] + 1e-10)
        if len(returns) == 0 or np.std(returns) < 1e-10:
            return 0.0
        return float((np.mean(returns) - risk_free_rate) / (np.std(returns) + 1e-10))

    def return_volatility(self) -> float:
        """Compute rolling return volatility."""
        if len(self.return_history) < 2:
            return 0.0
        values = np.array(self.return_history)
        returns = np.diff(values) / (values[:-1] + 1e-10)
        return float(np.std(returns))

    def observation_vector(self) -> np.ndarray:
        """Return normalized risk features."""
        features = [
            min(self.current_drawdown, 1.0),   # current drawdown [0, 1]
            min(self.max_drawdown, 1.0),        # max drawdown [0, 1]
            np.tanh(self.sharpe_ratio()),        # sharpe ratio [-1, 1] -> tanh
            min(self.return_volatility() * 100, 1.0),  # volatility
            min(self.trade_count / 100.0, 1.0),  # normalized trade count
        ]
        return np.array(features, dtype=np.float32)

    @property
    def feature_size(self) -> int:
        return 5


def get_observation(market: MarketState, portfolio: PortfolioState,
                    risk: RiskState, ticker: str = "default") -> np.ndarray:
    """Concatenate all state observations into a single flat vector."""
    current_price = market.current_price()
    obs = np.concatenate([
        market.observation_vector(),
        portfolio.observation_vector(current_price, ticker),
        risk.observation_vector(),
    ])
    return obs


def get_observation_size(market: MarketState, portfolio: PortfolioState,
                         risk: RiskState) -> int:
    """Total observation vector size."""
    return market.feature_size + portfolio.feature_size + risk.feature_size
