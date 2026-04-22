# Multi-Agent RL Trading Environment (OpenEnv)


# ALWAYS REMEMBER
Add a summary of what you did each in "summary.md"
## Overview

This project implements a **multi-agent reinforcement learning trading environment** using OpenEnv.

The system simulates financial markets and trains multiple specialized agents to:

* generate signals
* manage risk
* execute trades
* optimize portfolio allocation

The goal is to enable **learning through reward feedback**, where agents improve by reinforcing profitable and risk-aware behavior over time.

---

## Objectives

* Build a **multi-agent system** with interacting roles
* Design a **realistic trading environment**
* Implement a **reward-driven RL loop**
* Ensure **long-horizon learning**
* Provide **normalized grading (0 to 1)**

---

## System Architecture

### Agents

1. Quant Researcher (TA Agent)

   * Inputs: price data, indicators
   * Outputs: trade signals (bullish/bearish + confidence)

2. Fundamental Analyst (FA Agent)

   * Inputs: news / macro data
   * Outputs: sentiment bias

3. Risk Modeler

   * Inputs: portfolio + volatility
   * Outputs: position size, constraints

4. Quant Trader (Executor)

   * Inputs: signals + constraints
   * Outputs: action (buy/sell/hold)

5. Portfolio Manager (Meta Agent)

   * Inputs: all agent outputs + performance
   * Outputs: capital allocation, overrides

---

## Environment Design

### State

* OHLCV price data
* Technical indicators (RSI, EMA, volatility,MACD,RSI, Bollinger Bands)
* Portfolio:

  * cash
  * positions
  * unrealized PnL
* Risk metrics:

  * drawdown
  * exposure

---

### Action Space

* Discrete:

  * Buy
  * Sell
  * Hold

* Continuous:

  * Position size (0 → 1)

---

### Transition

* Market moves to next timestep
* Portfolio updates
* PnL recalculated

---

## Reinforcement Learning Loop

```
for episode:
    reset environment
    
    for t in time:
        state = get_state()
        
        researcher_signal = researcher(state)
        fa_signal = fa_agent(state)
        risk_constraints = risk_model(state)
        
        action = trader(state, signals, constraints)
        
        next_state, reward = environment.step(action)
        
        update_agents(reward)
```

---

## Reward System

### Core Reward Function

Reward must balance profit and risk:

```
reward_raw =
    + profit
    - drawdown_penalty
    - volatility_penalty
    + sharpe_bonus
    - overtrading_penalty
```

---

### Components

* profit = change in portfolio value
* drawdown_penalty = max_drawdown * weight
* volatility_penalty = return_std * weight
* sharpe_bonus = risk_adjusted_return
* overtrading_penalty = number_of_trades * weight

---

## Normalization (CRITICAL)

All rewards must be normalized between **0 and 1**.

### Normalization Function

```
normalized_reward = (reward_raw - min_reward) / (max_reward - min_reward)
normalized_reward = clip(normalized_reward, 0, 1)
```

---

## Grading System (0 → 1)

Grades are evaluation metrics derived from performance.

### Metrics

* normalized_profit
* normalized_sharpe
* normalized_drawdown_inverse
* normalized_consistency

---

### Final Grade

```
grade =
    0.4 * normalized_profit +
    0.3 * normalized_sharpe +
    0.2 * (1 - normalized_drawdown) +
    0.1 * consistency
```

All components must be scaled between **0 and 1**.

---

## Multi-Agent Reward Strategy

### Option 1 (Baseline)

* All agents share same reward

### Option 2 (Advanced)

* Researcher → rewarded for signal accuracy
* Risk Model → rewarded for low drawdown
* Trader → rewarded for execution + profit
* Portfolio Manager → rewarded for overall grade

---

## Data Sources

* Historical price data:

  * yfinance
  * ccxt (crypto)

---

## Installation

### System Requirements

* Python 3.10+
* Git

---

### Install Dependencies

```
pip install openenv
pip install torch transformers datasets
pip install trl unsloth
pip install pandas numpy
pip install yfinance ccxt
pip install gymnasium
pip install matplotlib plotly
pip install langchain
```

---

## File Structure

```
project/
│── env/
│   ├── trading_env.py
│   ├── reward.py
│   └── state.py
│
│── agents/
│   ├── researcher.py
│   ├── fa_agent.py
│   ├── risk_model.py
│   ├── trader.py
│   └── portfolio_manager.py
│
│── training/
│   ├── train.py
│   └── config.py
│
│── data/
│
│── utils/
│
│── main.py
```

---

## Training

* Use HuggingFace TRL or Unsloth
* Train using episodic simulation
* Track:

  * reward curve
  * grade improvement

---

## Evaluation

Compare:

* Random agent vs trained agent
* Before vs after reward
* Grade progression (0 → 1)

---

## Visualization

* Equity curve
* Drawdown chart
* Trade log

---

## Extensions (Optional)

* Market regimes (bull/bear)
* News-based sentiment
* Multi-strategy competition
* Self-play training

---

## Expected Outcome

* Agents learn to:

  * avoid bad trades
  * optimize risk-adjusted returns
  * adapt to market conditions

* Demonstrable improvement in:

  * reward
  * grade (0 → 1)

---

## Key Constraint

All evaluation outputs, rewards, and grades MUST be normalized between:

```
0 ≤ value ≤ 1
```

---

## End

------

## Submission Artifacts

### 1. Training Script (Colab)

A minimal training script demonstrating the environment using HuggingFace TRL / Unsloth is provided below:

**Colab Notebook Link:**
[ADD_YOUR_COLAB_LINK_HERE]

This notebook includes:

* Environment initialization
* Agent interaction loop
* Reward computation
* Basic training loop
* Reward / grade tracking

---

### 2. Media Submission

A short explanation of the project (under 2 minutes) is provided below:

**Video / Blog Link:**
[ADD_YOUR_VIDEO_OR_HF_BLOG_LINK_HERE]

Content includes:

* Problem statement
* Environment design
* Agent roles
* Reward system
* Demonstration of learning/improvement

---

## Notes for Submission

* Ensure the Colab notebook runs end-to-end without errors
* Include sample outputs (reward curve / grades)
* Keep video/blog concise and focused on:

  * What is novel
  * How agents interact
  * Evidence of learning

---
