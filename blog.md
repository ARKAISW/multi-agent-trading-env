# 🏛️ QuantHive: Decentralized Multi-Agent Trading Governance

**Can three AI agents with conflicting goals learn to govern each other — without any hardcoded rules?**

---

## 1. The Problem: Single-Agent Governance is Fake

The financial industry wants to deploy AI traders, but faces a governance paradox: current "multi-agent" trading systems are **single-agent systems with hardcoded business rules pretending to be governance**. The rules live inside `env.step()` — they're static, non-negotiable, and non-adaptive.

This creates a fundamental gap:
- **Hardcoded rules can't learn.** A static size cap of 10% doesn't adapt to regime changes.
- **Single-agent systems can't negotiate.** There's no dialogue between risk management and profit-seeking.
- **"Governance" is just a if-statement.** There's no independent actor with its own incentive structure enforcing constraints.

**The real question isn't "Can an agent follow rules?" It's "Can agents learn to *create and enforce* rules for each other?"**

---

## 2. The Insight: Independent Agents with Conflicting Rewards

True governance requires **independent actors with conflicting incentives**. A risk manager who is rewarded for *restricting* trades will naturally oppose a trader who is rewarded for *profit*. Their tension — not any hardcoded rule — creates emergent regulation.

PettingZoo's AEC (Agent-Environment Cycle) framework enables exactly this:
- Each agent takes its turn **independently**, with its own observation and action space
- Each agent's **output becomes part of the next agent's observation** (message passing)
- Each agent has its own **adversarial reward function** — aligned on different objectives

The key insight: **governance is a negotiation, not a monolithic rule engine.**

---

## 3. Architecture: 3-Agent AEC Cycle

```
  ┌──────────────┐     message     ┌───────────────────┐    message     ┌──────────┐
  │ Risk Manager │ ──────────────▶ │ Portfolio Manager  │ ─────────────▶│  Trader  │
  │  obs: 24     │  [size_limit,   │     obs: 27        │  [cap_alloc,  │ obs: 29  │
  │  act: Box(3) │   allow_new,    │     act: Box(2)    │   override]   │ act: Dict│
  └──────────────┘   force_reduce] └───────────────────┘               └──────────┘
                                                                             │
                                                                    Executes trade
                                                                             │
                                                                    Market advances
```

### Inter-Agent Message Passing

| From | To | Message | Effect |
|:---|:---|:---|:---|
| Risk Manager | PM + Trader | `[size_limit, allow_new, force_reduce]` | Constrains maximum position size, blocks new positions, forces reduction |
| Portfolio Manager | Trader | `[capital_allocation, override_strength]` | Caps capital deployment, can veto entire trades |

### Adversarial Rewards

| Agent | Rewarded For | Penalized For |
|:---|:---|:---|
| **Risk Manager** | Restricting size during drawdown, force-reduce during severe DD | Allowing reckless sizing during drawdown, portfolio losses |
| **Portfolio Manager** | High portfolio grade (profit + Sharpe - drawdown) | Deep drawdowns, excessive overrides |
| **Trader** | PnL, compliance (no interventions triggered) | Each governance intervention that modifies the trade |

The adversarial structure creates an **arms race**: if the Trader ignores RM limits, the RM learns to clamp harder. If the RM over-restricts, it gets no upside from portfolio growth. Equilibrium emerges organically.

---

## 4. Training: Multi-Agent GRPO with Alternating Optimization

### Phase 1: Rule-Based Warm-Up (REINFORCE)
All three agents use rule-based policies. We collect trajectories and compute per-agent discounted returns. Alternating optimization: Trader episodes → RM episodes → repeat.

### Phase 2: GRPO for the Trader (Qwen 2.5-1.5B)
The Trader is upgraded to a language model trained via **GRPO** with 5 verifiers. The prompts include the RM and PM governance messages:

```
You are a trading agent in a multi-agent governance system.
The Risk Manager has set the following constraints:
  size_limit=0.35, new_positions=allowed, force_reduce=no.
The Portfolio Manager allocated: capital_cap=0.50, override=none.
Market state: [... 24 values ...]

<thought>
RSI is 28 — oversold territory. EMA20 crossing above EMA50 suggests bullish
momentum. However, the Risk Manager restricts allocation to 0.35 given current
drawdown. The Portfolio Manager has allocated 50% capital. I'll propose a
conservative 0.25 size to stay within constraints...
</thought>
<action>
{"direction": 1, "size": 0.25, "sl": 49000, "tp": 52000}
</action>
```

**Critical distinction**: The governance verifier checks compliance against the RM's *dynamic, learned* `size_limit` — not a hardcoded constant. The Trader must learn to **read and respect** the RM message.

---

## 5. Anti-Reward-Hacking: Adversarial Rewards as Natural Defense

The adversarial reward structure inherently prevents gaming:

- **If the Trader ignores RM limits** → RM is rewarded for clamping harder → interventions increase → Trader pays penalty → arms race pushes toward compliance
- **If RM always blocks everything** → RM gets no upside from portfolio growth → it learns to moderate restrictions → allows profitable trades when risk is low
- **Multiple independent reward signals** per agent: each agent's reward depends on different metrics (PnL vs drawdown vs grade), preventing collapse to a single gaming strategy
- **Governance intervention log** provides process-level reward: the Trader is evaluated not just on *outcome* (final PnL) but on *process* (how many interventions were triggered)
- **Message-passing creates accountability**: The RM's constraints are observable and logged. If the RM sets a bad limit, the Trader's performance suffers, and the PM's grade drops — creating pressure on the RM to improve.

This is fundamentally different from single-agent systems where "compliance" is just a penalty term that can be gamed by the same agent.

---

## 6. Results

After multi-agent training with alternating optimization:

- **Governance interventions decreased** as the Trader learned to read and respect RM/PM signals
- **RM learned preemptive restriction**: during drawdown periods, the RM proactively reduces the size limit before the Trader even proposes a large trade
- **Trader compliance improved**: the Trader began proposing sizes within the RM's limit without needing to be clamped
- **Reasoning quality transformed**: trained Trader explicitly cites governance constraints in its `<thought>` blocks

### Per-Agent Reward Dynamics

The adversarial reward curves show the expected tension:
- RM and Trader returns are negatively correlated during drawdowns (RM restricts → Trader PnL drops)
- During recovery periods, both agents benefit (RM allows moderate sizing → Trader captures upside)
- PM serves as a stabilizer, penalizing extreme behavior from either side

---

## 7. Why It Matters

This is the **first true PettingZoo multi-agent governance environment for finance**. Every prior "multi-agent" trading env either:
- Uses a single agent with hardcoded rules (not multi-agent)
- Has multiple agents that don't interact (parallel, not AEC)
- Has no adversarial reward structure (agents are cooperative, not regulatory)

QuantHive's framework generalizes to:
- **Healthcare**: diagnostic AI governed by safety oversight agents
- **Autonomous systems**: vehicle AI constrained by regulatory agents
- **Legal AI**: document generation agents subject to compliance reviewers

Any domain where AI must operate under institutional oversight can benefit from **adversarial multi-agent governance training**.

---

**Built for the OpenEnv April '26 Hackathon | Theme 1: Multi-Agent Interactions**
**Sub-themes: Fleet AI — Scalable Oversight | Halluminate — Multi-Actor Environments**
**Author**: Arka Sarkar
