"""
Quant Trader (Executor Agent).
Reads state + signals + constraints → outputs actions + reasoning.
"""

import numpy as np
from typing import Dict, Tuple, Any


from policy.local_model import LocalPolicyModel

class QuantTrader:
    """
    Execution agent that decides final trade direction and size
    based on signals from Researcher/FA and constraints from Risk Model.
    """

    def __init__(self, aggression: float = 0.5):
        self.name = "QuantTrader"
        self.aggression = aggression
        self.policy = LocalPolicyModel()

    def __call__(
        self,
        observation: np.ndarray,
        researcher_data: Tuple[str, float, str],
        fa_data: Tuple[float, str],
        risk_data: Tuple[float, Dict, str],
    ) -> Tuple[int, float, float, float, str]:
        """
        Decide action based on signals and reasoning.

        Returns:
            (direction, size, sl, tp, reasoning)
        """
        signal, confidence, researcher_reasoning = researcher_data
        fa_sentiment, fa_reasoning = fa_data
        position_limit, constraints, risk_reasoning = risk_data
        current_exposure = float(observation[15]) if len(observation) > 15 else 0.0
        short_exposure = float(observation[18]) if len(observation) > 18 else 0.0
        
        raw_price = constraints.get("raw_price")
        sl_ratio = constraints.get("suggested_sl_ratio", 0.02)

        # --- Format signals for the policy model ---
        if signal == "bullish":
            ta_score = confidence
        elif signal == "bearish":
            ta_score = -confidence
        else:
            ta_score = 0.0

        signals = {
            "raw_state": observation.tolist() if hasattr(observation, "tolist") else observation,
            "ta_score": ta_score,
            "fa_sentiment": (fa_sentiment * 2.0) - 1.0, # Map back to [-1, 1]
            "position_limit": position_limit,
            "constraints": constraints,
            "text_context": {
                "researcher": researcher_reasoning,
                "fundamental": fa_reasoning,
                "risk": risk_reasoning
            }
        }

        # --- Delegate to policy (Fine-tuned model will use text_context) ---
        direction, size = self.policy.predict(observation, signals)
        
        # ═══════════════════════════════════════════════════
        # STRICT 1% RISK RULE + FIXED 2:1 RRR
        # ═══════════════════════════════════════════════════
        # Max loss per trade = 1% of current portfolio value.
        # RRR is fixed at 2:1 (TP distance = 2 × SL distance).
        sl = 0.0
        tp = 0.0
        RISK_PER_TRADE = 0.01    # 1% of portfolio
        REWARD_RISK_RATIO = 2.0  # Fixed 2:1 RRR
        reasoning = f"Trader executing {['HOLD', 'BUY', 'SELL/SHORT'][direction]} based on signals. Risk check: {risk_reasoning}."

        if raw_price is not None and raw_price > 0 and direction != 0:
            # SL distance from ATR-based ratio (from risk model)
            sl_distance = raw_price * sl_ratio  # e.g. 2% of price

            if direction == 1:  # BUY / Cover short
                sl = raw_price - sl_distance
                tp = raw_price + (sl_distance * REWARD_RISK_RATIO)
            elif direction == 2:  # SELL long / Open short
                sl = raw_price + sl_distance
                tp = raw_price - (sl_distance * REWARD_RISK_RATIO)

            # ── 1% Risk Position Sizing ──
            # If SL is hit, loss = qty × sl_distance
            # We want: qty × sl_distance ≤ portfolio_value × 0.01
            # So: qty ≤ (portfolio_value × 0.01) / sl_distance
            # And: size = qty × price / portfolio_value
            # → size ≤ (0.01 × price) / (sl_distance × (1 + commission))
            # Simplified: size = RISK_PER_TRADE / (sl_ratio × (1 + 0.001))
            max_risk_size = RISK_PER_TRADE / (sl_ratio + 1e-10)

            # ── Volatility-Adjusted Target Sizing ──
            # Target a constant volatility layout. If market vol is 2x normal, halve position.
            current_volatility = float(observation[12]) if len(observation) > 12 else 0.015
            vol_target = 0.015  # Benchmark normal volatility
            vol_scalar = np.clip(vol_target / max(current_volatility, 1e-4), 0.2, 1.5)
            
            size = min(size * vol_scalar, max_risk_size, position_limit)

            reasoning = (
                f"Trader executing {['HOLD', 'BUY', 'SELL/SHORT'][direction]} | "
                f"Vol Adjusted (x{vol_scalar:.2f}), Size: {size:.1%} of portfolio | "
                f"SL: {sl:.2f}, TP: {tp:.2f} (2:1 RRR) | "
                f"{risk_reasoning}"
            )
        else:
            size = min(size, position_limit)
            
        size = float(np.clip(size, 0.0, 1.0))
        return direction, size, sl, tp, reasoning
