"""
Risk Modeler Agent.
Reads portfolio + volatility → outputs position size limits, constraints + reasoning.
"""

import numpy as np
from typing import Dict, Tuple


class RiskModeler:
    """
    Risk management agent that sets position size limits and constraints
    based on current portfolio state and market volatility.
    """

    def __init__(
        self,
        max_drawdown_limit: float = 0.20,
        max_exposure: float = 0.80,
        vol_threshold: float = 0.5,
    ):
        self.name = "RiskModeler"
        self.max_drawdown_limit = max_drawdown_limit
        self.max_exposure = max_exposure
        self.vol_threshold = vol_threshold

    def __call__(self, observation: np.ndarray) -> Tuple[float, Dict, str]:
        """
        Compute risk constraints and reasoning.
        """
        # Market features
        close_price = observation[3]  # un-normalized close is index 3 in market_vec
        # Wait, check state.py indexing for MarketState.observation_vector()
        # 0: open/close, 1: high/close, 2: low/close, 3: close/close=1.0
        # Actually, state.py observation_vector() doesn't include raw price.
        # I need to get it from the market_row if passed, or use the normalized version.
        # But wait, observation[0:4] are price ratios. 
        # I should probably have the environment pass the raw price in some other way 
        # or include it in the observation.
        
        atr_ratio = observation[13]

        # Portfolio features
        exposure_ratio = observation[15]
        portfolio_return = observation[16]

        # Risk features
        current_drawdown = observation[18]
        volatility = observation[21]

        # --- 1% Risk Rule ---
        sl_distance_ratio = 2.0 * atr_ratio
        suggested_size = 0.01 / sl_distance_ratio if sl_distance_ratio > 0 else 0.1
        
        reasons = [f"1% Rule suggests {suggested_size:.1%} allocation based on {sl_distance_ratio:.1%} volatility-stop"]

        # --- Compute position size limit ---
        position_limit = suggested_size

        if current_drawdown > self.max_drawdown_limit * 0.5:
            reduction = 1.0 - (current_drawdown / self.max_drawdown_limit)
            position_limit *= max(0.1, reduction)
            reasons.append(f"Drawdown ({current_drawdown:.1%}) is high; reducing risk")

        if volatility > self.vol_threshold:
            position_limit *= 0.5
            reasons.append("High market volatility detected; slashing size")

        if portfolio_return < 0.90:
            position_limit *= 0.5
            reasons.append("Significant portfolio losses; defensive sizing active")

        position_limit = float(np.clip(position_limit, 0.01, self.max_exposure))
        reasoning = "; ".join(reasons)

        # --- Build constraints ---
        constraints = {
            "position_size_limit": position_limit,
            "suggested_sl_ratio": sl_distance_ratio,
            "allow_new_positions": current_drawdown < self.max_drawdown_limit,
            "force_reduce": current_drawdown > self.max_drawdown_limit * 0.9,
            "raw_price": None, # Environment will fill this if needed, or we use a ratio
        }

        return position_limit, constraints, reasoning
