---
title: "QuantHive"
emoji: "🏛️"
colorFrom: "blue"
colorTo: "indigo"
sdk: "docker"
pinned: false
app_port: 7860
---

# 🏛️ QuantHive — Decentralized Multi-Agent Trading Governance

[![OpenEnv](https://img.shields.io/badge/Environment-OpenEnv-blue.svg)](https://github.com/meta-pytorch/OpenEnv)
[![PettingZoo](https://img.shields.io/badge/Framework-PettingZoo%20AEC-green.svg)](https://pettingzoo.farama.org/)
[![Hackathon](https://img.shields.io/badge/Hackathon-OpenEnv%20April%20'26-orange.svg)](https://hackathon.openenv.org)

**Can three AI agents with conflicting goals learn to govern each other?**

QuantHive is a PettingZoo AEC (Agent-Environment Cycle) environment where **three independent RL agents** — a Risk Manager, a Portfolio Manager, and a Trader — negotiate via observation message-passing with **adversarial reward structures**. The Risk Manager is rewarded for *restricting* dangerous trades; the Trader is rewarded for *profit*. Their tension creates **emergent self-regulation** — not hardcoded rules, but learned governance.

> Existing "multi-agent" trading envs are single-agent systems with hardcoded rules pretending to be agents. QuantHive puts governance in the hands of independently trainable agents.

---

## 📌 Deliverables

| **Output** | **Link** |
| :--- | :--- |
| 🚀 Live Space | [Hugging Face Space](https://huggingface.co/spaces/ARKAISW/QuantHive) |
| 🧠 Trained Model | [QuantHive GRPO Trader](https://huggingface.co/ARKAISW/QuantHive-GRPO-Trader) |
| 📓 Kaggle Run | [Kaggle Notebook](https://www.kaggle.com/code/arka2930/notebook24ed9f9bff) |
| 📝 **Submission Blog** | [QuantHive: Multi-Agent Governance (HF)](https://huggingface.co/spaces/ARKAISW/QuantHive/blob/main/blog.md) |
| 🐍 Setup Script | [QuantHive Training Script](https://github.com/ARKAISW/multi-agent-trading-env/blob/master/train_hf.py) |

---

## 🛑 The Problem: AI Agents Can't Govern Each Other

Traditional RL trading environments optimize a single agent for PnL. "Governance" is just hardcoded business rules inside `env.step()`. This creates agents that:

- **Ignore risk constraints** — sizing positions recklessly to chase reward
- **Can't adapt to dynamic oversight** — rules are static, never learned
- **Have no inter-agent negotiation** — governance is a monolith, not a dialogue

> Regulators don't want a model that follows static rules. They want AI that can **negotiate, comply, and adapt** to changing oversight — the way human teams do.

---

## 🏦 The Solution: PettingZoo AEC with 3 Adversarial Agents

QuantHive decomposes trading governance into **three independent RL agents** that take turns each market step via PettingZoo's AEC (Agent-Environment Cycle):

```
┌─────────────────────────────────────────────────────────────┐
│                    One Market Cycle                          │
│                                                              │
│  ① Risk Manager ──▶ ② Portfolio Manager ──▶ ③ Trader        │
│    obs: 24 dims       obs: 27 dims           obs: 29 dims   │
│    act: Box(3)        act: Box(2)            act: Dict(4)   │
│                                                              │
│  RM message ──────────▶ PM obs                               │
│  RM + PM messages ────────────────────────▶ Trader obs       │
│                                                              │
│  After Trader acts: market advances one candle               │
└─────────────────────────────────────────────────────────────┘
```

| Agent | Observation | Action | Reward Strategy |
|:---|:---|:---|:---|
| 🛡️ **Risk Manager** | Market + Portfolio + Risk (24) | `[size_limit, allow_new, force_reduce]` | +reward for restricting during drawdown; shares downside pain |
| 💼 **Portfolio Manager** | Base obs + RM message (27) | `[capital_allocation, override_strength]` | Grade-based portfolio performance; penalized for deep drawdown |
| ⚖️ **Trader** | Base obs + RM + PM messages (29) | `{direction, size, sl, tp}` | Pure PnL + compliance bonus; penalized per governance intervention |

### The Key Innovation: Governance is Emergent, Not Hardcoded

Each agent's **output becomes part of the next agent's observation**. The RM sends `[size_limit, allow_new, force_reduce]` — these are learned constraints, not static rules. The Trader must read them and decide whether to comply or risk intervention.

```python
# From a real governance cycle — RM clamped the Trader's size
info["governance"] = {
    "rm_message": [0.35, 1.0, 0.0],      # RM: limit 35%, allow new, don't force reduce
    "pm_message": [0.50, 0.0],             # PM: 50% allocation, no override
    "proposed": {"direction": 1, "size": 0.7},
    "executed": {"direction": 1, "size": 0.35},  # RM clamped size from 0.7 to 0.35
    "interventions": [{"agent": "RiskManager", "type": "size_clamp"}]
}
```

---

## 🔬 The Environment: Observation Spaces

| Agent | Dims | Source | Features |
|:---|:---|:---|:---|
| Risk Manager | 24 | `MarketState` + `PortfolioState` + `RiskState` | OHLCV, RSI, EMA20/50, MACD, BB, ATR, Volatility, Cash ratio, Exposure, Drawdown, Sharpe |
| Portfolio Manager | 27 | Base (24) + RM message (3) | Above + `[size_limit, allow_new_positions, force_reduce]` |
| Trader | 29 | Base (24) + RM (3) + PM (2) | Above + `[capital_allocation, override_strength]` |

**Trader Action Space**: `{direction: 0/1/2, size: [0,1], sl: price, tp: price}`

**What Makes It Hard**: The Trader must reason about *dynamic, learned constraints* from the RM and PM — not static rules. If the RM decides high drawdown warrants a 15% size cap, the Trader must learn to read that signal and comply.

---

## 🧪 Training: Multi-Agent GRPO with Alternating Optimization

We use two training approaches:

### 1. REINFORCE-Style Multi-Agent Training
Alternating optimization: episodes where the Trader is optimized (RM/PM frozen), then episodes where the RM is optimized (Trader/PM frozen). Each agent's policy gradient is computed from its own discounted returns.

### 2. GRPO for the Trader (Qwen 2.5-1.5B)
The Trader agent is trained as a language model via **GRPO** using 5 verifiers with **governance-aware rewards**:

| # | Verifier | What It Checks |
|---|:---|:---|
| 1 | **Format** | Valid `<thought>` + `<action>` tags, reasoning length ≥ 150 chars |
| 2 | **Alignment** | Does the reasoning match the market signals? (Anti-hallucination) |
| 3 | **Risk** | Is the proposed size within the **RM's dynamic size_limit**? |
| 4 | **Profit** | Does the direction match the actual price trend? |
| 5 | **🏛️ Governance** | Would this action pass governance without intervention? Checks compliance against **learned RM constraints**, not hardcoded limits. |

Verifiers #3 and #5 are **the differentiators**: they read the RM's dynamic `size_limit` from the prompt, meaning the Trader must learn to comply with *learned* governance, not static rules.

---

## 📊 Results: From Reckless to Self-Regulated

### Live Training Evidence (Kaggle Qwen 2.5 1.5B)

![Kaggle Training Overview](plots/kaggle_training_loss.png)
*Figure 2: Live GRPO training logs showing loss and reward curves converging over 250 steps.*

![Kaggle Reward Breakdown](plots/kaggle_training_reward.png)
*Figure 3: Detailed reward progression indicating rapid convergence on format, risk compliance, and governance.*

### Training Outcomes

| Metric | Early Training | Late Training | Change |
|:---|:---|:---|:---|
| Governance Interventions | High | Low | Agent learned self-regulation |
| RM Size Restrictions | Reactive | Anticipatory | RM learned preemptive risk mgmt |
| Trader Compliance | Low | High | Trader reads & respects RM signals |
| Reasoning Quality | Random | Cites constraints | Verifiable CoT |

**The trained Trader explicitly cites governance constraints in its reasoning:**
> *"RSI is 28 indicating oversold territory, however the Risk Manager restricts us to 0.35 allocation given current drawdown of 4.2%. The Portfolio Manager has allocated 50% capital. Proposing a conservative 0.25 size..."*

---

## 🎯 Theme Alignment: Multi-Agent Interactions (Theme #1)

QuantHive directly addresses Theme #1 and both sub-themes:

- **Fleet AI — Scalable Oversight**: The Risk Manager and Portfolio Manager are oversight agents that monitor and constrain the Trader in real-time, creating scalable governance. Adding more oversight agents (compliance, ESG, etc.) is trivial within the AEC framework.
- **Halluminate — Multi-Actor Environments**: Three independent actors with adversarial incentives negotiate through observation message-passing, producing emergent strategic behavior. The Trader must model what constraints the Risk Manager will impose based on the current portfolio state — theory-of-mind reasoning.

The PettingZoo AEC architecture enables genuine multi-agent dynamics that cannot be replicated by a single agent with hardcoded rules.

---

## 🏛️ Why It Matters

The finance industry doesn't need AI that clicks "Buy." It needs AI that can **sit in a compliance meeting**.

QuantHive demonstrates that RL agents can learn to:
1. **Govern each other** — independent agents with conflicting rewards create emergent regulation
2. **Negotiate constraints** — governance is a dialogue, not a monolith
3. **Show verifiable reasoning** — generating auditable Chain-of-Thought
4. **Reduce interventions** — learning self-regulation through adversarial training

This generalizes beyond finance to **healthcare, autonomous systems, and any domain where AI must operate under institutional oversight**.

---

## 🚀 Quick Launch

### 1. Install
```bash
pip install -r requirements-space.txt
```

### 2. Run Multi-Agent Training
```bash
python training/train_multi_agent.py --episodes 200 --difficulty easy
```

### 3. Launch Interactive UI
```bash
python app.py --demo
```

### 4. OpenEnv Standard API
```bash
# Reset the multi-agent environment
curl -X POST http://localhost:7860/reset

# Step with a trader action (RM & PM use rule-based policies)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"direction": 1, "size": 0.1, "sl": 0, "tp": 0}'

# Get full environment state (including governance log)
curl http://localhost:7860/state
```

### 5. PettingZoo Compliance Test
```python
from pettingzoo.test import api_test
from env.multi_agent_env import MultiAgentTradingEnv
env = MultiAgentTradingEnv()
api_test(env, num_cycles=50, verbose_progress=True)
```

---

**Built for the OpenEnv April '26 Hackathon | Theme 1: Multi-Agent Interactions (Fleet AI — Scalable Oversight, Halluminate — Multi-Actor Environments)**
**Author**: [Arka Sarkar](mailto:arkasarkar1507@gmail.com)
