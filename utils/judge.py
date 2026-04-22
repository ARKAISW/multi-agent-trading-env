import os
import json
import numpy as np
import openai
from openai import OpenAI
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class LLMJudge:
    """
    The Llama 3.3 70B Judge that evaluates agent interactions and provides 
    a normalized reward for Reinforcement Learning.
    """

    def __init__(self, api_key: str = None, base_url: str = None):
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key and self.base_url and "groq.com" in self.base_url:
            resolved_key = os.getenv("GROQ_API_KEY")

        self.enabled = bool(resolved_key)
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=self.base_url # For Groq/Together/Samba
        )
        self.model = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
        self._warned = False

    def evaluate_step(self, 
                      state_brief: str, 
                      agent_reasoning: Dict[str, str], 
                      action: Dict[str, Any], 
                      outcome: Dict[str, Any]) -> float:
        """
        Evaluate a single step and return a reward [0, 1].
        """
        if not self.enabled:
            if not self._warned:
                print("Judge disabled: no valid remote API key configured. Using neutral fallback reward.")
                self._warned = True
            return 0.5

        # Ensure action and outcome are JSON serializable (handling numpy arrays)
        serializable_action = {
            k: (v.tolist() if hasattr(v, "tolist") else v) 
            for k, v in action.items()
        }
        serializable_outcome = {
            k: (v.tolist() if hasattr(v, "tolist") else v) 
            for k, v in outcome.items()
            if k not in ["positions"] # Positions is already a dict of floats
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            return float(np.clip(data.get("score", 0.5), 0.0, 1.0))
        except Exception as e:
            if not self._warned:
                print(f"Judge error: {e}")
                self._warned = True
            return 0.5

    def get_episode_reward(self, metrics: Dict[str, Any]) -> float:
        """
        Evaluate overall episode performance.
        """
        # We can use the judge for final grading too
        return 0.0 # Placeholder
