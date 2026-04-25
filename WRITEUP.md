# 🌌 Pitch Deck: MATE (Multi-Agent Trading Environment)

**De-risking AI Trading through specialized agent coordination and verifiable RL.**

---

## 1. The Executive Summary
The financial industry is eager to adopt AI, but blocked by a critical bottleneck: **Trust**. Traditional Reinforcement Learning (RL) agents are black boxes. They may be profitable on a test set, but they offer no explanation for *why* they take a trade, and often ignore risk constraints when chasing reward.

**MATE solves this.** We decomposed the trading process into a **Multi-Agent Institutional Desk** and trained an LLM policy specifically to enforce mathematical risk boundaries and generate verifiable Chain-of-Thought (CoT) reasoning for every single action. 

---

## 2. The Problem: "Black Box" Vulnerability
Current environments train solitary AI models to optimize purely for Portfolio Value (PnL). This results in:
* **Hallucinated Signals**: Taking aggressive trades in market noise.
* **Risk Ignorance**: Averaging down on losing positions to chase a delayed reward.
* **Zero Compliance**: Regulatory bodies and fund managers cannot audit an AI that outputs `[1]` (Buy) without a formal hypothesis.

---

## 3. The Solution: The Multi-Agent Desk
Instead of an individual rogue agent, MATE forces AI to operate as a coordinated Quantitative Desk. No trade occurs without consensus.

* 🔍 **Researcher**: Extracts RSI, MACD, and EMA consensus.
* 📉 **Fundamental Analyst**: Interprets broader market regime and news bias.
* 🛡️ **Risk Manager (The Anchor)**: Enforces Kelly Criterion position sizing and strict drawdown caps.
* ⚔️ **Trader (The Actor)**: Synthesizes desk data and generates the formal execution logic.
* 💼 **Portfolio Manager**: Has final veto power based on total global exposure.

---

## 4. The Environment (Gymnasium API)
We built a high-fidelity continuous market environment tailored for Language Model evaluation.

* **What the Agent Sees (Observation):** A 24-dimension vector combining immediate price action, derived technical indicators, and critical portfolio risk metrics (Drawdown, Cash-on-Hand, Sector Exposure).
* **What the Agent Does (Action):** The agent MUST return a dual-layer response:
  1. A `<thought>` block mathematically justifying the trade.
  2. A strict JSON schema containing `direction`, `size`, `sl` (Stop Loss), and `tp` (Take Profit). 
* **What the Agent gets Rewarded for (Signal):**
  * **Format Reward:** Base points for valid JSON and detailed reasoning.
  * **PnL Reward:** Baseline relative normalized returns.
  * **Risk Penalty:** An aggressive negative curve applied instantly if the agent breaches institutional parameters (e.g., risking more than limits allow).

---

## 5. The Training Process (GRPO)
We broke from standard PPO and utilized **GRPO (Group Relative Policy Optimization)** on a 1.5B parameter reasoning model (`Qwen2.5-1.5B`). 

Instead of updating pure policy gradients against a singular value-function, GRPO evaluates *groups* of generated textual reasonings. Models are rewarded not just for the correct action, but for the most logically sound `<thought>` process leading to that action. We ran this optimization via Unsloth on consumer-grade hardware (Google Colab T4 GPU).

---

## 6. The Results (What Changed?)
The divergence between the baseline model and the trained policy is profound:

1. **Risk Control (+67% compliance):** The baseline model frequently breached exposure limits. After ~100 GRPO steps, the trained agent internalized the Risk Penalty and began sizing positions conservatively. Total risk breaches dropped by **85%**.
2. **Format compliance (+82%):** The agent stopped emitting malformed JSON and guaranteed structurally sound outputs.
3. **Verifiable Reasoning:** Most impressively, the trained policy explicitly started citing metrics in its `<thought>` blocks (e.g., *"RSI is 28, indicating oversold territory, however Risk restricts us to 0.4 allocation..."*), giving fund managers the exact auditable trail they require.

---

## 7. Why MATE Matters
The finance industry doesn't need an AI that just clicks "Buy"; it needs an AI that can sit in a compliance meeting. 

MATE provides the first framework where AI trading is modular, mathematically risk-checked, and inherently explainable. It paves the way for institutional capital to finally trust generative policies with live execution.
