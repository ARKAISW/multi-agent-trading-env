import os
import json
import importlib
import time
import numpy as np
from typing import Dict, Tuple, Any

class LocalPolicyModel:
    """
    The brain of the Quant Trader. Uses a 135M-300M parameter model
    (e.g. Qwen2.5-1.5B) to process agent reasoning and make decisions.
    """
    
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.getenv("LOCAL_MODEL_PATH", "models/local_policy")
        self.is_active = os.getenv("USE_LOCAL_POLICY", "false").lower() == "true"
        self.max_new_tokens = int(os.getenv("LOCAL_POLICY_MAX_NEW_TOKENS", "64"))
        self.allow_cpu_policy = os.getenv("ALLOW_CPU_LOCAL_POLICY", "false").lower() == "true"

        self.model: Any = None
        self.tokenizer: Any = None
        self.device = "cpu"
        self._torch: Any = None
        self._auto_model_cls: Any = None
        self._auto_tokenizer_cls: Any = None        
        if self.is_active:
            self._load_model()
            
    def _load_model(self):
        """Loads the local transformer model if available."""
        try:
            self._load_runtime_dependencies()
            if self.device == "cpu" and not self.allow_cpu_policy:
                print("Local policy disabled on CPU. Set ALLOW_CPU_LOCAL_POLICY=true to force-enable.")
                self.is_active = False
                return
            if os.path.exists(self.model_path):
                print(f"Loading local policy model from {self.model_path}...")
                self.tokenizer = self._auto_tokenizer_cls.from_pretrained(self.model_path)
                self.model = self._auto_model_cls.from_pretrained(
                    self.model_path,
                    dtype=self._torch.float16 if self.device == "cuda" else self._torch.float32,
                    device_map="auto"
                )
            else:
                print(f"Local model not found at {self.model_path}. Using fallback.")
                self.is_active = False
        except Exception as e:
            print(f"Error loading model: {e}. Using fallback.")
            self.is_active = False

    def _load_runtime_dependencies(self):
        """Import heavyweight ML dependencies only when the local policy is enabled."""
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        self._torch = torch
        self._auto_model_cls = getattr(transformers, "AutoModelForCausalLM")
        self._auto_tokenizer_cls = getattr(transformers, "AutoTokenizer")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _debug_log(self, hypothesis_id: str, location: str, message: str, data: dict) -> None:
        # region agent log
        payload = {
            "sessionId": "85370c",
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        try:
            with open("debug-85370c.log", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except Exception:
            pass
        # endregion

    def predict(self, observation: np.ndarray, signals: Dict[str, Any]) -> Tuple[int, float]:
        """
        Processes text reasoning + numerical signals to output (direction, size).
        Uses <thought> ... <action> tags for GRPO-compatible reasoning.
        """
        self._debug_log(
            "H12",
            "policy/local_model.py:80",
            "predict_mode",
            {"is_active": bool(self.is_active), "model_loaded": bool(self.model is not None), "device": self.device},
        )
        if not self.is_active or self.model is None:
            return self._fallback_logic(signals)

        text_ctx = signals.get("text_context", {})
        prompt = self._build_prompt(text_ctx, signals)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 1. Extract the <action> block
            import re
            action_match = re.search(r'<action>\s*({.*?})\s*</action>', full_text, re.DOTALL)
            
            if action_match:
                json_str = action_match.group(1)
                data = json.loads(json_str)
                direction = int(data.get("direction", 0))
                size = float(data.get("size", 0.0))
                return direction, size
            
            # 2. Fallback: try finding any JSON if tags are missing/malformed
            json_start = full_text.rfind("{")
            json_end = full_text.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                try:
                    json_str = full_text[json_start:json_end]
                    data = json.loads(json_str)
                    direction = int(data.get("direction", 0))
                    size = float(data.get("size", 0.0))
                    return direction, size
                except json.JSONDecodeError:
                    pass
            
            return self._fallback_logic(signals)
        except json.JSONDecodeError:
            return self._fallback_logic(signals)
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_logic(signals)

    def _build_prompt(self, text_ctx: Dict, signals: Dict) -> str:
        state_str = json.dumps(signals.get("raw_state", []))
        signals_str = json.dumps({
            "ta": signals.get("ta_score"),
            "fa": signals.get("fa_sentiment"),
            "position_limit": signals.get("position_limit")
        })
        
        return f"""You are a Quant Trader. Analyze the scenario and return a single action.

Scenario:
{{"state": {state_str}, "signals": {signals_str}}}

Respond exactly in this format:
<thought>
(Provide concise reasoning here)
</thought>
<action>
{{
  "direction": 0,
  "size": (0.0 to 1.0)
}}
</action>
"""

    def _fallback_logic(self, signals: Dict[str, Any]) -> Tuple[int, float]:
        """Indicator-aware fallback policy when model is unavailable.
        
        Uses RSI, EMA crossover, MACD, BB position from the observation
        vector for smarter, more conservative decision-making.
        """
        ta_score = signals.get("ta_score", 0.0)
        fa_sentiment = signals.get("fa_sentiment", 0.0)
        position_limit = signals.get("position_limit", 1.0)
        constraints = signals.get("constraints", {})
        raw_state = signals.get("raw_state", [])
        
        # Extract key indicators from observation vector
        # Market features: indices 0-13
        rsi = float(raw_state[5]) if isinstance(raw_state, list) and len(raw_state) > 5 else 0.5
        ema20_ratio = float(raw_state[6]) if isinstance(raw_state, list) and len(raw_state) > 6 else 1.0
        ema50_ratio = float(raw_state[7]) if isinstance(raw_state, list) and len(raw_state) > 7 else 1.0
        macd_hist = float(raw_state[10]) if isinstance(raw_state, list) and len(raw_state) > 10 else 0.0
        bb_position = float(raw_state[11]) if isinstance(raw_state, list) and len(raw_state) > 11 else 0.5
        volatility = float(raw_state[12]) if isinstance(raw_state, list) and len(raw_state) > 12 else 0.0
        
        # Portfolio features
        long_exposure = float(raw_state[15]) if isinstance(raw_state, list) and len(raw_state) > 15 else 0.0
        short_exposure = float(raw_state[18]) if isinstance(raw_state, list) and len(raw_state) > 18 else 0.0
        
        if constraints.get("force_reduce", False):
            if long_exposure > 1e-6:
                return 2, min(0.5, position_limit)
            elif short_exposure > 1e-6:
                return 1, min(0.5, position_limit)

        # ── Composite signal from indicators ──
        bullish_points = 0.0
        bearish_points = 0.0
        
        # RSI (strong mean-reversion signal)
        if rsi < 0.25:
            bullish_points += 0.35  # Oversold → buy
        elif rsi < 0.35:
            bullish_points += 0.15
        elif rsi > 0.75:
            bearish_points += 0.35  # Overbought → sell/short
        elif rsi > 0.65:
            bearish_points += 0.15
        
        # EMA crossover (trend-following)
        if ema20_ratio > ema50_ratio * 1.001:
            bullish_points += 0.25  # Short-term above long-term
        elif ema20_ratio < ema50_ratio * 0.999:
            bearish_points += 0.25
        
        # MACD histogram (momentum)
        if macd_hist > 0.05:
            bullish_points += 0.20
        elif macd_hist < -0.05:
            bearish_points += 0.20
        
        # Bollinger Band position
        if bb_position < 0.15:
            bullish_points += 0.20  # Near lower band → bounce likely
        elif bb_position > 0.85:
            bearish_points += 0.20  # Near upper band → pullback likely
        
        # Agent signals (from Researcher + FA)
        combined = 0.6 * ta_score + 0.4 * fa_sentiment
        if combined > 0.1:
            bullish_points += 0.20
        elif combined < -0.1:
            bearish_points += 0.20

        # Volatility dampener: reduce size in high vol
        vol_scale = max(0.3, 1.0 - volatility * 2.0)
        
        # ── Decision logic ──
        net_signal = bullish_points - bearish_points
        
        # Conservative sizing: scale by signal strength, cap at 50% of limit
        base_size = min(abs(net_signal) * 0.5, position_limit * 0.5) * vol_scale
        
        if long_exposure > (position_limit * 1.05):
            direction = 2
            size = min(0.3, position_limit)
        elif short_exposure > (position_limit * 1.05):
            direction = 1
            size = min(0.3, position_limit)
        elif net_signal > 0.15 and constraints.get("allow_new_positions", True):
            if short_exposure > 1e-6:
                direction = 1  # Cover short first
                size = base_size
            else:
                direction = 1  # Open/add long
                size = base_size
        elif net_signal < -0.15 and constraints.get("allow_new_positions", True):
            if long_exposure > 1e-6:
                direction = 2  # Close long first
                size = base_size
            else:
                direction = 2  # Open short
                size = base_size * 0.7  # Slightly more conservative for shorts
        elif net_signal < -0.05 and long_exposure > 1e-6:
            direction = 2  # Mild bearish: trim long
            size = base_size * 0.5
        elif net_signal > 0.05 and short_exposure > 1e-6:
            direction = 1  # Mild bullish: cover short
            size = base_size * 0.5
        else:
            direction = 0
            size = 0.0

        self._debug_log(
            "H13",
            "policy/local_model.py:fallback",
            "fallback_decision",
            {
                "rsi": float(rsi),
                "ema_cross": float(ema20_ratio - ema50_ratio),
                "macd_hist": float(macd_hist),
                "bb_position": float(bb_position),
                "net_signal": float(net_signal),
                "bullish": float(bullish_points),
                "bearish": float(bearish_points),
                "vol_scale": float(vol_scale),
                "direction": int(direction),
                "size": float(size),
            },
        )
        return direction, float(np.clip(size, 0.0, 1.0))
