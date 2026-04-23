# 🌌 MATE: Multi-Agent Trading Environment

[![OpenEnv](https://img.shields.io/badge/Environment-OpenEnv-blue.svg)](https://github.com/OpenEnv)
[![Hackathon](https://img.shields.io/badge/Hackathon-OpenEnv%20April%20%2726-orange.svg)](https://hackathon.openenv.org)

**De-risking AI Trading through specialized agent coordination and verifiable RL.**

---

## 🛑 The Problem
Traditional AI trading systems are often "black boxes"—single-agent models that make opaque decisions. In volatile markets, these models are prone to catastrophic failures, reward-hacking, and logic hallucination. To be production-ready, AI trading needs **cross-verification**, **multi-agent dissent**, and **verifiable rewards**.

## 🏦 Our Solution: MATE
MATE (Multi-Agent Trading Environment) is a high-fidelity Gymnasium simulation that decomposes trading into five specialized professional roles. Instead of one model doing everything, MATE features a "Desk Architecture":

| Role | Responsibility | Verifiable Signal |
| :--- | :--- | :--- |
| **🔍 Researcher** | Technical Analysis | RSI/MACD Consensus |
| **📉 Analyst** | Fundamental Sentiment | Macro Tone/News |
| **🛡️ Risk** | Capital Preservation | 1% Kelly Criterion |
| **⚔️ Trader** | Execution (CoT) | RRR-based SL/TP |
| **💼 manager** | Global Oversight | Strategic Action Overrides |

---

## 🔬 The Environment
MATE is built on **OpenEnv** and features a robust **Anti-Hacking Reward Engine**:
- **Chain-of-Thought (CoT)**: Agents must reason in `<thought>` tags before acting.
- **Independent Verifiers**: Reward functions penalize lazy reasoning, format errors, and risk limit breaches.
- **Curriculum Learning**: Supports Easy (Trending), Medium (Ranging), and Hard (Crypto) regimes.

---

## 📊 Results & Performance

### Trained Agent vs. Random Baseline
Our **GRPO (Group Relative Policy Optimization)** training pipeline using **Unsloth** delivers significant quantitative improvements.

![Baseline Comparison](file:///E:/Development/Round2/docs/plots/baseline_comparison.png)
*Figure 1: Performance distribution showing our trained agent significantly outperforming the random baseline with a tighter reward variance.*

### Convergence
Thanks to Unsloth's 4-bit LoRA patching, we achieved stable convergence on 500M parameter models (Qwen 0.5B) in under 15 minutes on Kaggle T4 GPUs.

![Reward Curve](file:///E:/Development/Round2/docs/plots/reward_curve.png)
*Figure 2: Normalized reward curve showing steady improvement as the agent learns to satisfy format, risk, and profit constraints.*

---

## 🚀 Getting Started

### 1. Requirements
```bash
pip install -r requirements-space.txt
```

### 2. Run the Benchmark
Compare the latest policy against a random baseline:
```bash
python training/benchmark.py --episodes 50
```

### 3. Launch the 2D Office UI
Visualize the multi-agent coordination in real-time:
```bash
python app.py --demo
```

---

## 🏛️ Why It Matters
MATE demonstrates how **Multi-Agent Reinforcement Learning (MARL)** can be made verifiable and professional. By enforcing separate concerns (Risk vs. Alpha), we create a system that is not only more profitable but also significantly safer for institutional deployment.

**Built for the OpenEnv April '26 Hackathon.**
**Author**: [Arka Sarkar](mailto:arkasarkar1507@gmail.com)
