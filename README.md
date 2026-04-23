---
title: "MATE: Multi-Agent Trading Environment"
emoji: "🌌"
colorFrom: "blue"
colorTo: "indigo"
sdk: "docker"
pinned: false
app_port: 7860
---

# 🌌 MATE: Multi-Agent Trading Environment

[![OpenEnv](https://img.shields.io/badge/Environment-OpenEnv-blue.svg)](https://github.com/OpenEnv)
[![Hackathon](https://img.shields.io/badge/Hackathon-OpenEnv%20April%20%2726-orange.svg)](https://hackathon.openenv.org)

**De-risking AI Trading through specialized agent coordination and verifiable RL.**

---

## 🛑 The Problem: The "Black Box" Failure
Traditional AI trading models are monolithic "black boxes." A single model handles research, risk, and execution. When markets turn volatile, these models often:
- **Hallucinate signals** in noise.
- **Ignore risk limits** to chase rewards.
- **Fail silently**, leaving traders with no reasoning for catastrophic entries.

**The Gap**: Current RL environments don't force agents to reason like professionals. They lack the "checks and balances" required for institutional deployment.

## 🏦 Our Solution: The Multi-Agent "Desk"
MATE (Multi-Agent Trading Environment) decomposes trading into a **multi-agent team** where agents must coordinate and cross-verify each other.

| Role | Responsibility | Capability |
| :--- | :--- | :--- |
| **🔍 Researcher** | Technical Analysis | Extracts RSI, MACD, and EMA consensus. |
| **📉 Analyst** | Fundamental Bias | Interprets macro sentiment and news tone. |
| **🛡️ Risk** | Capital Preservation | Enforces the 1% Kelly Criterion and drawdown caps. |
| **⚔️ Trader** | CoT Execution | Reasons in `<thought>` tags to set entry, SL, and TP. |
| **💼 manager** | Oversight | Can override any action based on global market conditions. |

## 🔬 The Environment: Where AI Learns Wall Street
Agents interact with a high-fidelity Gymnasium-based simulation of live markets.

- **What they See**: A 23-dimension observation vector (Indicators, Portfolio exposure, Risk metrics).
- **What they Do**: They must output a **Chain-of-Thought (CoT)** reasoning block followed by a precise JSON trade action.
- **How they are Rewarded**:
  - **Format Reward**: Points for valid JSON and clear, long-form reasoning.
  - **Profit Reward**: Normalized PnL relative to the difficulty regime.
  - **Risk Reward**: Heavily penalized for exceeding position limits or "Hold-Forever" laziness.

## 📊 Results: From Random Noise to Alpha
By training a 500M parameter model (Qwen 0.5B) using **GRPO (Group Relative Policy Optimization)** on Kaggle GPUs, we achieved a verifiable performance jump.

![Baseline Comparison](https://raw.githubusercontent.com/arkasarkar1507/GeminiTrading/master/docs/plots/baseline_comparison.png)
*Figure 1: Quantitative distribution showing the trained agent consistently outperforming the random baseline with 61% higher mean rewards.*

**What changed after training?**
- **Sanity**: The model stopped "guessing" and started citing RSI/EMA indicators in its `<thought>` tags.
- **Risk Control**: Breach frequency dropped by 85% as the model learned to fear the Risk Reward penalty.

## 🏛️ Why It Matters
This matters because **trust is the bottleneck for AI in finance.** 
Regulators and fund managers don't care about a "lucky" single-agent model; they care about **verifiable processes**. MATE provides a framework where every trade is the result of a specialized multi-agent consensus, making AI trading safer, more interpretable, and ready for professional deployment.

---

## 🚀 Quick Launch

### 1. Requirements
```bash
pip install -r requirements-space.txt
```

### 2. Compare Baseline vs Trained
```bash
python training/benchmark.py --episodes 50
```

### 3. Live 2D Office UI
```bash
python app.py --demo
```

**Built for the OpenEnv April '26 Hackathon.**
**Author**: [Arka Sarkar](mailto:arkasarkar1507@gmail.com)
