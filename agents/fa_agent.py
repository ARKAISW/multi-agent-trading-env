import os
import numpy as np
from typing import Tuple

class FundamentalAnalyst:
    """
    Fundamental analysis agent that generates a sentiment bias.
    Outputs (sentiment, reasoning) for SFT training.
    """

    def __init__(self, fast_mode: bool = False):
        self.name = "FundamentalAnalyst"
        self.fast_mode = fast_mode
        self._momentum_window = []
        self._window_size = 5

    def __call__(self, observation: np.ndarray) -> Tuple[float, str]:
        """
        Returns (sentiment_score [0,1], reasoning_brief).
        """
        volatility = observation[12]   
        ema20_ratio = observation[6]   
        ema50_ratio = observation[7]   

        # Trend component
        trend = (1.0 - ema20_ratio) * 10
        self._momentum_window.append(ema20_ratio)
        if len(self._momentum_window) > self._window_size:
            self._momentum_window.pop(0)
        momentum = -(self._momentum_window[-1] - self._momentum_window[0]) if len(self._momentum_window) >= 2 else 0.0

        # Rule-based reasoning for SFT
        reasons = []
        if trend > 0.05: reasons.append("Price action is in an uptrend")
        elif trend < -0.05: reasons.append("Price action is in a downtrend")
        
        if momentum > 0: reasons.append("Bullish momentum increasing")
        else: reasons.append("Bearish momentum increasing")
        
        if volatility > 0.4: reasons.append("High volatility dampening conviction")

        reasoning = "; ".join(reasons) if reasons else "No clear fundamental trend."

        # Rule-based sentiment
        vol_dampener = 1.0 - min(volatility * 2, 0.8)
        raw_sentiment = (trend * 0.5 + momentum * 50 * 0.5) * vol_dampener
        sentiment_final = (np.tanh(raw_sentiment) + 1.0) / 2.0
        
        return float(np.clip(sentiment_final, 0.0, 1.0)), reasoning

    def reset(self):
        self._momentum_window = []

