"""
Training configuration for the multi-agent trading environment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for training."""

    # ─── Data ───
    data_source: str = "ccxt"             # Use CCXT by default for Crypto
    tickers: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    train_split: float = 0.8

    # ─── Environment ───
    initial_cash: float = 100_000.0
    commission: float = 0.0005           # Lower commissions for high-volume crypto
    max_steps: Optional[int] = None

    # ─── Reward Weights ───
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "profit": 1.0,
        "drawdown": 0.8,                 # Heavier penalty for crypto drawdowns
        "volatility": 0.2,
        "sharpe": 0.5,
        "overtrading": 0.05,
        "hold_penalty": 0.01,            # Small cost for inaction
        "directional_bonus": 0.3,        # Reward matching market trend
    })

    # ─── Training Loop ───
    num_episodes: int = 200
    learning_rate: float = 1e-4
    gamma: float = 0.99
    seed: int = 42

    # ─── Agent Settings ───
    trader_aggression: float = 0.6
    risk_max_drawdown: float = 0.30      # Higher threshold for crypto
    risk_max_exposure: float = 0.90
    risk_vol_threshold: float = 0.8      # Crypto-specific volatility threshold

    # ─── Logging ───
    log_every: int = 10
    save_dir: str = "checkpoints"
    metrics_file: str = "training_metrics.csv"
    trajectories_file: str = "sft_trajectories.jsonl"
    save_trajectories: bool = True
    fast_mode: bool = False

    # ─── Reward Strategy ───
    reward_strategy: str = "shared"


# Default config instance
DEFAULT_CONFIG = TrainingConfig()
