import os
import numpy as np
from typing import Dict, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()

class PortfolioManager:
    """
    Meta-agent that oversees all other agents.
    Uses an optional OpenAI-compatible LLM for strategic overrides.
    """

    def __init__(self, performance_window: int = 10, override_threshold: float = 0.15, fast_mode: bool = False):
        self.name = "PortfolioManager"
        self.fast_mode = fast_mode
        self.performance_window = performance_window
        self.override_threshold = override_threshold
        self._reward_history = []
        self._grade_history = []
        
        remote_enabled = os.getenv("ENABLE_REMOTE_PM", "false").lower() == "true"

        # Initialize OpenAI client only when explicitly enabled.
        self.client = None
        if remote_enabled and os.getenv("OPENAI_API_KEY"):
            try:
                self.client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL")
                )
                self.model = os.getenv("PM_MODEL", "gpt-4o") # Default to GPT-4o for PM
                self.system_prompt = "You are a Portfolio Manager. Decide if you should OVERRIDE the trader's action. Input will be grade, drawdown, pnl, and trader's action. Respond with JSON: {\"override\": bool, \"new_direction\": int, \"new_size\": float, \"reason\": str}. Directions: 0=Hold, 1=Buy, 2=Sell."
                print(f"LLM initialized for PortfolioManager (OpenAI: {self.model})")
            except Exception as e:
                print(f"Failed to init OpenAI for PM: {e}")

    def __call__(
        self,
        observation: np.ndarray,
        trader_action: Tuple[int, float],
        info: dict,
    ) -> Tuple[float, Optional[Tuple[int, float]]]:
        direction, size = trader_action
        grade = info.get("grade", 0.5)
        max_drawdown = info.get("max_drawdown", 0.0)
        pnl_pct = info.get("pnl_pct", 0.0)

        self._reward_history.append(info.get("episode_reward_mean", 0.0))
        self._grade_history.append(grade)

        # Capital Allocation
        if len(self._grade_history) >= self.performance_window:
            recent_grades = self._grade_history[-self.performance_window:]
            avg_grade = np.mean(recent_grades)
        else:
            avg_grade = grade

        capital_allocation = float(np.clip(0.3 + 0.5 * avg_grade, 0.1, 0.9))
        if max_drawdown > 0.15:
            capital_allocation *= max(0.3, 1.0 - max_drawdown * 2)

        # Try OpenAI Override
        if self.client and not self.fast_mode:
            try:
                user_content = f"Grade: {grade:.2f}, Drawdown: {max_drawdown:.2f}, PnL: {pnl_pct:.2f}, Trader: dir={direction}, size={size}"
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                if content:
                    data = json.loads(content)
                    if data.get("override"):
                        print(f"[PM OVERRIDE] {data.get('reason', '')}")
                        return capital_allocation, (data.get("new_direction", 0), data.get("new_size", 0.0))
            except Exception as e:
                # print(f"PM LLM Error: {e}") # Suppress to avoid flooding
                pass 

        # Rule-based Fallback
        override_action = None
        if max_drawdown > self.override_threshold:
            # High drawdown: force close any open position
            if direction == 1:
                override_action = (2, 0.3)  # Close long / reduce buying
            elif direction == 2:
                override_action = (1, 0.3)  # Cover short / reduce shorting
        elif direction in (1, 2) and size > capital_allocation:
            override_action = (direction, capital_allocation)

        return capital_allocation, override_action

    def reset(self):
        self._reward_history = []
        self._grade_history = []
