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
        
        # Apply professional risk rules
        sl = 0.0
        tp = 0.0
        reasoning = f"Trader executing {['HOLD', 'BUY', 'SELL'][direction]} based on signals. Risk check: {risk_reasoning}."

        if raw_price is not None:
            rrr = 2.0 if confidence > 0.7 else 1.5
            if direction == 1:  # BUY
                size = min(size, position_limit)
                sl = raw_price * (1.0 - sl_ratio)
                risk_dist = raw_price - sl
                tp = raw_price + (risk_dist * rrr)
            elif direction == 2:  # SELL (for longs, this closes; for shorts, it opens)
                # Note: Currently env is long-only, but logic should be robust
                sl = raw_price * (1.0 + sl_ratio)
                risk_dist = sl - raw_price
                tp = raw_price - (risk_dist * rrr)
        else:
            # Fallback if raw_price is missing: use ratios (0.98, 1.05) 
            # environment will need to handle this or it might break SL/TP logic
            pass
            
        size = float(np.clip(size, 0.0, 1.0))
        return direction, size, sl, tp, reasoning

