import os
import json
import numpy as np
from openai import OpenAI
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


def _algorithmic_score(
    action: Dict[str, Any],
    agent_reasoning: Dict[str, str],
    outcome: Dict[str, Any],
    state_brief: str,
) -> float:
    """
    Deterministic scoring function that replaces the LLM judge when the
    remote API is unavailable or rate-limited.  Scores on four axes:

      1. Direction matches TA signal sentiment      (0.3)
      2. Position size respects risk limit           (0.2)
      3. SL/TP are set for non-hold trades           (0.2)
      4. Reasoning quality (length + keyword check)  (0.3)

    Returns a score in [0, 1].
    """
    score = 0.0

    # --- 1. Direction plausibility  (0.30) ---
    direction = action.get("direction", 0)
    if hasattr(direction, 'item'):
        direction = int(direction)
    pnl_pct = outcome.get("pnl_pct", 0.0)

    if direction == 1 and pnl_pct >= 0:
        score += 0.30
    elif direction == 2 and pnl_pct <= 0:
        score += 0.30
    elif direction == 0:
        score += 0.15        # Neutral — acceptable but not rewarded

    # --- 2. Position sizing  (0.20) ---
    size_raw = action.get("size", 0.0)
    size = float(size_raw[0]) if hasattr(size_raw, '__len__') else float(size_raw)
    max_dd = outcome.get("max_drawdown", 0.0)

    if 0.0 <= size <= 1.0:
        score += 0.10
    if size <= 0.5 or max_dd < 0.10:
        score += 0.10        # Conservative sizing rewarded

    # --- 3. SL / TP presence  (0.20) ---
    sl_raw = action.get("sl", 0.0)
    tp_raw = action.get("tp", 0.0)
    sl = float(sl_raw[0]) if hasattr(sl_raw, '__len__') else float(sl_raw)
    tp = float(tp_raw[0]) if hasattr(tp_raw, '__len__') else float(tp_raw)

    if direction != 0:
        if sl > 0:
            score += 0.10
        if tp > 0:
            score += 0.10
    else:
        score += 0.20        # Hold doesn't need SL/TP

    # --- 4. Reasoning quality  (0.30) ---
    all_reasoning = " ".join(str(v) for v in agent_reasoning.values()).lower()
    word_count = len(all_reasoning.split())

    if word_count > 20:
        score += 0.10
    if word_count > 50:
        score += 0.05

    quality_keywords = [
        "rsi", "ema", "macd", "volatility", "drawdown",
        "risk", "trend", "bullish", "bearish", "momentum",
        "support", "resistance", "limit", "exposure",
    ]
    hits = sum(1 for kw in quality_keywords if kw in all_reasoning)
    score += min(hits * 0.03, 0.15)

    return float(np.clip(score, 0.0, 1.0))


class LLMJudge:
    """
    Evaluates agent interactions and provides a normalized reward.

    Primary:  Llama 3.3 70B (or compatible) via OpenAI-compatible API.
    Fallback: Deterministic algorithmic scorer (no API calls, no rate limits).
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "")
        remote_enabled = os.getenv("ENABLE_REMOTE_JUDGE", "false").lower() == "true"
        resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not resolved_key and self.base_url and "groq.com" in self.base_url:
            resolved_key = os.getenv("GROQ_API_KEY", "")

        self.enabled = remote_enabled and bool(resolved_key)
        self.client = None
        if self.enabled:
            self.client = OpenAI(
                api_key=resolved_key,
                base_url=self.base_url if self.base_url else None
            )
        self.model = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
        self._warned = False
        self._rate_limit_hits = 0
        self._max_rate_limit_hits = 3  # Fall back after 3 consecutive rate limits

    def evaluate_step(self,
                      state_brief: str,
                      agent_reasoning: Dict[str, str],
                      action: Dict[str, Any],
                      outcome: Dict[str, Any]) -> float:
        """
        Evaluate a single step and return a reward [0, 1].

        Tries the remote LLM judge first; on failure or rate-limit,
        falls back to the algorithmic scorer automatically.
        """
        # If remote judge is disabled or rate-limited, use algorithmic fallback
        if not self.enabled or self._rate_limit_hits >= self._max_rate_limit_hits:
            return _algorithmic_score(action, agent_reasoning, outcome, state_brief)

        # Ensure action and outcome are JSON serializable
        serializable_action = {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in action.items()
        }
        serializable_outcome = {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in outcome.items()
            if k not in ["positions"]
        }
        serializable_outcome["positions"] = outcome.get("positions", {})

        prompt = f"""
        Analyze this trade execution for a professional quant firm.
        
        MARKET STATE:
        {state_brief}
        
        AGENT REASONING:
        {json.dumps(agent_reasoning, indent=2)}
        
        ACTION TAKEN:
        {json.dumps(serializable_action, indent=2)}
        
        OUTCOME:
        {json.dumps(serializable_outcome, indent=2)}
        
        CRITERIA:
        1. Professionalism: Did they follow the 1% risk rule and SL/TP constraints?
        2. Alignment: Does the action match the agents' reasoning?
        3. Logic: Was the trade direction sound given the indicators?

        Respond with ONLY a JSON object: {{"score": float, "reason": str}}. 
        The score MUST be between 0.0 and 1.0.
        """

        try:
            if not self.client:
                return _algorithmic_score(action, agent_reasoning, outcome, state_brief)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if not content:
                return _algorithmic_score(action, agent_reasoning, outcome, state_brief)

            data = json.loads(content)
            self._rate_limit_hits = 0  # Reset on success
            return float(np.clip(data.get("score", 0.5), 0.0, 1.0))

        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str or "limit" in err_str:
                self._rate_limit_hits += 1
                if self._rate_limit_hits >= self._max_rate_limit_hits:
                    print(f"Judge: rate-limited {self._rate_limit_hits}× — switching to algorithmic fallback permanently.")
            elif not self._warned:
                print(f"Judge error: {e} — using algorithmic fallback.")
                self._warned = True

            return _algorithmic_score(action, agent_reasoning, outcome, state_brief)

    def get_episode_reward(self, metrics: Dict[str, Any]) -> float:
        """Evaluate overall episode performance."""
        return 0.0  # Placeholder
