import os
import torch
import re
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datasets import Dataset
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOTrainer, GRPOConfig

# ==========================================
# 0. Patch for GRPO
# ==========================================
PatchFastRL("GRPO", "unsloth")

# ==========================================
# 1. Technical Indicators (from utils/indicators.py)
# ==========================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta).where(delta < 0, 0.0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    # EMA
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    # MACD
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    # BB
    middle = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["bb_upper"] = middle + 2 * std
    df["bb_lower"] = middle - 2 * std
    # Vol/ATR
    df["volatility"] = close.pct_change().rolling(20).std()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - close.shift(1)).abs(), (df["low"] - close.shift(1)).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df.bfill().fillna(0)

# ==========================================
# 2. State Management (from env/state.py)
# ==========================================
@dataclass
class MarketState:
    prices: pd.DataFrame
    current_step: int = 0
    def current_price(self): return float(self.prices.iloc[self.current_step]["close"])
    def observation_vector(self):
        row = self.prices.iloc[self.current_step]
        close = row["close"]
        return np.array([
            row["open"]/close, row["high"]/close, row["low"]/close, 1.0,
            np.log1p(row["volume"])/20.0, row["rsi"]/100.0,
            row["ema_20"]/close, row["ema_50"]/close,
            np.tanh(row["macd"]/close*100), np.tanh(row["macd_signal"]/close*100),
            (close - row["bb_lower"])/(row["bb_upper"] - row["bb_lower"] + 1e-10),
            min(row["volatility"]*100, 1.0), row["atr"]/close
        ], dtype=np.float32)

@dataclass
class PortfolioState:
    initial_cash: float = 100000.0
    cash: float = 100000.0
    positions: Dict[str, float] = field(default_factory=dict)
    def total_value(self, price): return self.cash + self.positions.get("default", 0.0) * price
    def observation_vector(self, price):
        tv = self.total_value(price)
        return np.array([self.cash/self.initial_cash, (tv-self.cash)/tv, tv/self.initial_cash, 0.0], dtype=np.float32)

@dataclass
class RiskState:
    peak_value: float = 100000.0
    max_drawdown: float = 0.0
    def update(self, val):
        self.peak_value = max(self.peak_value, val)
        self.max_drawdown = max(self.max_drawdown, (self.peak_value - val)/self.peak_value)
    def observation_vector(self):
        return np.array([0.0, self.max_drawdown, 0.0, 0.0, 0.0], dtype=np.float32)

# ==========================================
# 3. Environment (from env/trading_env.py)
# ==========================================
class TradingEnv:
    def __init__(self, df):
        self.df = compute_indicators(df)
        self.market = MarketState(self.df)
        self.portfolio = PortfolioState()
        self.risk = RiskState()
    def reset(self, step=0):
        self.market.current_step = step
        self.portfolio.cash = 100000.0
        self.portfolio.positions = {}
        self.risk.max_drawdown = 0.0
        return self._get_obs()
    def _get_obs(self):
        price = self.market.current_price()
        return np.concatenate([self.market.observation_vector(), self.portfolio.observation_vector(price), self.risk.observation_vector()])
    def step(self, action):
        price = self.market.current_price()
        # Simple step logic for GRPO reward calc
        dir, size = action["direction"], action["size"]
        if dir == 1: # Buy
            qty = (self.portfolio.cash * size) / price
            self.portfolio.cash -= qty * price * 1.001
            self.portfolio.positions["default"] = self.portfolio.positions.get("default", 0.0) + qty
        elif dir == 2: # Sell
            qty = self.portfolio.positions.get("default", 0.0) * size
            self.portfolio.cash += qty * price * 0.999
            self.portfolio.positions["default"] -= qty
        
        self.market.current_step += 1
        new_price = self.market.current_price()
        new_val = self.portfolio.total_value(new_price)
        self.risk.update(new_val)
        return self._get_obs(), (new_val - 100000.0)/100000.0

# ==========================================
# 4. GRPO Configuration & Trainer
# ==========================================
MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
SYSTEM_PROMPT = """You are a Quant Trader. Analyze State and Signals.
<thought> Reasoning </thought> <action> {"direction": int, "size": float} </action>"""

model, tokenizer = FastLanguageModel.from_pretrained(model_name=MODEL_NAME, max_seq_length=1024, load_in_4bit=True, device_map="auto")
model = FastLanguageModel.get_peft_model(model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

def extract_action(c):
    m = re.search(r"<action>\s*({.*?})\s*</action>", c, re.DOTALL)
    try: return json.loads(m.group(1)) if m else None
    except: return None

# SETUP REAL DATA (Placeholder or uploaded CSV)
dummy_df = pd.DataFrame({"open": np.random.randn(1000)+100, "high": np.random.randn(1000)+101, "low": np.random.randn(1000)+99, "close": np.random.randn(1000)+100, "volume": np.random.rand(1000)*1000})
env = TradingEnv(dummy_df)

def env_reward_func(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        action = extract_action(completion)
        if not action or "direction" not in action:
            rewards.append(0.0)
            continue
        
        # Extract starting step from prompt (mocking state)
        env.reset(step=np.random.randint(0, 800))
        _, pnl = env.step({"direction": action["direction"], "size": action.get("size", 0.5)})
        
        # Reward is PnL based
        rewards.append(1.0 + pnl * 100) # Scale PnL
    return rewards

def format_reward_func(prompts, completions, **kwargs):
    return [1.0 if extract_action(c) else 0.0 for c in completions]

# Create dataset from real environment observations
def get_dataset():
    data = []
    for _ in range(200):
        obs = env.reset(step=np.random.randint(0, 800))
        prompt = f"{SYSTEM_PROMPT}\nState: {obs.tolist()}\nSignals: {{'ta': 0.5, 'position_limit': 1.0}}\n"
        data.append({"prompt": prompt})
    return Dataset.from_list(data)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward_func, env_reward_func],
    args=GRPOConfig(output_dir="qwen_trading", learning_rate=5e-5, per_device_train_batch_size=2, gradient_accumulation_steps=4, max_steps=100, num_generations=4),
    train_dataset=get_dataset(),
)

trainer.train()
model.save_pretrained("qwen_05b_final")
