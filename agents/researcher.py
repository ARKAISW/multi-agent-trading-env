"""
Quant Researcher (Technical Analysis Agent).
Reads price data + indicators → outputs trade signals with confidence + reasoning.
"""

import numpy as np
from typing import Tuple, Optional


class QuantResearcher:
    """
    Technical analysis agent that generates trading signals
    based on market indicators.
    """

    def __init__(self):
        self.name = "QuantResearcher"

    def __call__(self, observation: np.ndarray, market_row: Optional[dict] = None) -> Tuple[str, float, str]:
        """
        Generate a signal and reasoning from the observation.
        Mapping follows env/state.py:
        - [5]: RSI (normalized 0-1)
        - [6]: EMA20 ratio
        - [7]: EMA50 ratio
        - [10]: MACD histogram (tanh)
        - [11]: BB position (0-1)
        """
        rsi = observation[5]          
        ema20_ratio = observation[6]  
        ema50_ratio = observation[7]  
        macd_hist = observation[10]   
        bb_position = observation[11] 

        bullish_score = 0.0
        bearish_score = 0.0
        reasons = []

        # RSI signal
        if rsi < 0.30:
            bullish_score += 0.3
            reasons.append(f"RSI is oversold at {rsi*100:.1f}")
        elif rsi > 0.70:
            bearish_score += 0.3
            reasons.append(f"RSI is overbought at {rsi*100:.1f}")

        # EMA crossover
        if ema20_ratio > ema50_ratio:
            bullish_score += 0.25
            reasons.append("Price is above short-term EMA trend")
        else:
            bearish_score += 0.25
            reasons.append("Price is below short-term EMA trend")

        # MACD histogram
        if macd_hist > 0.05:
            bullish_score += 0.25
            reasons.append("MACD momentum is increasing")
        elif macd_hist < -0.05:
            bearish_score += 0.25
            reasons.append("MACD momentum is decreasing")

        # Bollinger Band position
        if bb_position < 0.2:
            bullish_score += 0.2
            reasons.append("Price is near the lower Bollinger Band")
        elif bb_position > 0.8:
            bearish_score += 0.2
            reasons.append("Price is near the upper Bollinger Band")

        # Determine signal
        if bullish_score > bearish_score + 0.1:
            signal = "bullish"
            confidence = min(bullish_score, 1.0)
        elif bearish_score > bullish_score + 0.1:
            signal = "bearish"
            confidence = min(bearish_score, 1.0)
        else:
            signal = "neutral"
            confidence = 0.3

        reasoning = "; ".join(reasons) if reasons else "No strong technical signals."
        return signal, confidence, reasoning
