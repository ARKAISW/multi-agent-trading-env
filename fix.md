# QuantHive Round2-Copy — Complete Fix List

> **Context**: This project at `E:\Development\Round2 - Copy` is a PettingZoo AEC multi-agent trading environment for the OpenEnv Hackathon. It was forked from a working Gymnasium single-agent version (`Round2`). The core PettingZoo env (`env/multi_agent_env.py`) and a basic training script (`training/train_multi_agent.py`) have already been created, but several files still reference the old Gym version and critical deliverables are missing.
>
> **Goal**: Make this a complete, submission-ready hackathon entry. All edits happen in `E:\Development\Round2 - Copy`.

---

## PROJECT ARCHITECTURE (What Already Exists)

- `env/multi_agent_env.py` — **NEW, DONE** — PettingZoo AECEnv with 3 agents:
  - `risk_manager_0`: obs=Box(24), action=Box(3) [size_limit, allow_new, force_reduce]
  - `portfolio_manager_0`: obs=Box(27), action=Box(2) [cap_alloc, override_strength]
  - `trader_0`: obs=Box(29), action=Dict{direction, size, sl, tp}
  - Turn order: RM → PM → Trader per market step
  - Inter-agent message passing: RM output → PM obs, RM+PM output → Trader obs
  - Adversarial rewards: RM rewarded for restricting during drawdown, Trader rewarded for PnL
- `env/trading_env.py` — OLD Gymnasium env (keep for backward compat, used for data generation)
- `env/state.py` — MarketState, PortfolioState, RiskState (shared by both envs)
- `env/reward.py` — Reward functions + 5 GRPO verifiers (format, alignment, risk, profit, governance)
- `training/train_multi_agent.py` — **NEW, DONE** — REINFORCE-style multi-agent training with rule-based policies
- `training/train_grpo.py` — OLD GRPO training script for the Gym env
- `api/server.py` — **PARTIALLY REWRITTEN** — imports updated to PettingZoo, `make_initial_state()` updated, but SimulationRunner still uses old Gym logic
- `app.py` — Gradio/FastAPI launcher
- `ui/` — React frontend (functional, shows agent messages + chart)
- `openenv.yaml` — **STALE** — still points to `env.trading_env:TradingEnv`
- `README.md` — **STALE** — describes the old Gym governance-in-env design
- `WRITEUP.md` — **STALE** — describes single-agent architecture
- `mate_training.ipynb` — **STALE** — Colab notebook for old Gym env
- `Dockerfile` — Functional but missing `pettingzoo` dependency
- `plots/` — Has old training plots from Gym version

---

## CHANGES NEEDED (In Priority Order)

---

### 🔴 1. Fix `openenv.yaml` — Points to Wrong Environment

**File**: `openenv.yaml`

Change `entry_point` from the old Gym env to the new PettingZoo env. Update observation space to reflect multi-agent structure.

```yaml
# OpenEnv Manifesto
version: "1.0"
name: "QuantHive"
description: "Decentralized multi-agent trading governance — three independent RL agents (Risk Manager, Portfolio Manager, Trader) with adversarial rewards negotiate via PettingZoo AEC turns."
author: "Arka Sarkar"

# Environment Specification
environment:
  entry_point: "env.multi_agent_env:MultiAgentTradingEnv"
  type: "pettingzoo_aec"
  agents:
    - risk_manager_0
    - portfolio_manager_0
    - trader_0
  observation_space:
    risk_manager_0: { shape: [24], dtype: "float32", description: "Market + portfolio + risk state" }
    portfolio_manager_0: { shape: [27], dtype: "float32", description: "Base obs + RM constraints [size_limit, allow_new, force_reduce]" }
    trader_0: { shape: [29], dtype: "float32", description: "Base obs + RM constraints + PM allocation [cap_alloc, override_strength]" }
  action_space:
    risk_manager_0:
      type: "box"
      shape: [3]
      description: "[size_limit (0-1), allow_new_positions (0-1), force_reduce (0-1)]"
    portfolio_manager_0:
      type: "box"
      shape: [2]
      description: "[capital_allocation (0-1), override_strength (0-1)]"
    trader_0:
      type: "dict"
      items:
        direction: { type: "int", low: 0, high: 2, description: "0=Hold, 1=Buy, 2=Sell" }
        size: { type: "float", low: 0.0, high: 1.0 }
        sl: { type: "float", description: "Stop Loss price" }
        tp: { type: "float", description: "Take Profit price" }

server:
  port: 7860
  endpoints:
    reset: "/reset"
    step: "/step"
    state: "/state"

tags:
  - "PettingZoo AEC"
  - "Multi-Agent"
  - "Adversarial Rewards"
  - "Financial Governance"
  - "Inter-Agent Negotiation"
  - "Self-Regulation"
```

---

### 🔴 2. Add `pettingzoo` to Dependencies

**File**: `requirements.txt` — add `pettingzoo>=1.24.0`
**File**: `requirements-space.txt` — add `pettingzoo>=1.24.0`

---

### 🔴 3. Finish `api/server.py` — Complete SimulationRunner Rewrite

**File**: `api/server.py`

The imports and `make_initial_state()` have been updated. The `SimulationRunner` class and the API endpoints (`/reset`, `/step`, `/state`) still use the old `TradingEnv.step()` loop. They must be rewritten to:

1. **SimulationRunner** must instantiate `MultiAgentTradingEnv` instead of `TradingEnv`
2. **Each simulation step** must run a full AEC cycle (RM → PM → Trader) using `env.agent_iter()`
3. Use the rule-based policies from `training/train_multi_agent.py` (`RuleRiskManagerPolicy`, `RulePortfolioManagerPolicy`, `RuleTraderPolicy`) as the default agent policies for the demo
4. After each AEC cycle, broadcast per-agent messages and negotiation state to the UI via `sim_state`
5. The `negotiation` field in `sim_state` must be populated with RM and PM messages each cycle
6. The `flow` field must log the per-agent turn messages (e.g., "RM: Size limit set to 0.35", "PM: Allocation capped at 0.5", "Trader: BUY 0.3 @ 50123.45")

The OpenEnv facade endpoints must still work:
- `POST /reset` → calls `env.reset()`, returns initial trader observation
- `POST /step` → accepts a trader action dict, runs full AEC cycle (RM and PM use rule policies), returns trader's obs/reward/done/info
- `GET /state` → calls `env.state()`, returns full shared state

This is the most complex single change. The existing `SimulationRunner` class structure can be adapted — replace the inner loop body.

---

### 🔴 4. Generate Training Evidence (Plots)

After the GRPO training pipeline (change #8) is working:

- Run training for ≥100 GRPO steps
- Save to `plots/`:
  - `reward_curve.png` — per-agent reward over training steps (RM, PM, Trader on same axes)
  - `loss_curve.png` — policy loss convergence
  - `baseline_comparison.png` — random vs trained agent performance per metric
- Each plot must have labeled axes, a title, and a one-line caption
- Commit these `.png` files to the repo

---

### 🔴 5. Deploy to HF Space

- Update `Dockerfile` to install `pettingzoo` (add to `requirements-space.txt`)
- Push to HF Space at `https://huggingface.co/spaces/ARKAISW/QuantHive`
- Verify from a logged-out browser that `/reset`, `/step`, `/state` all return valid JSON
- The Space must be public and cloneable

---

### 🟠 6. Rewrite `README.md`

**File**: `README.md`

The current README describes the old Gym-based governance-in-env design. Rewrite it to describe the PettingZoo architecture. Keep the same general structure but update all technical content:

Key sections to change:
- **Title/Tagline**: "Can three AI agents with conflicting goals learn to govern each other?" or similar
- **The Problem**: Same framing (AI can't self-govern), but add: "Existing 'multi-agent' trading envs are single-agent with hardcoded rules pretending to be agents"
- **The Solution**: Describe PettingZoo AEC with 3 independent agents, adversarial rewards, and inter-agent message passing. Remove all references to "governance lives in env.step()" — that was the old design. Now governance is emergent from agent interaction
- **Environment section**: Update observation dimensions (RM=24, PM=27, Trader=29), explain message passing, show the AEC turn diagram
- **Training section**: Update to reflect multi-agent GRPO, show per-agent reward curves
- **Results section**: Update with new plot embeds and new metrics
- **Theme alignment**: Explicitly cite Theme #1 (Multi-Agent Interactions) and sub-themes (Fleet AI Scalable Oversight, Halluminate Multi-Actor)
- **Quick Launch**: Keep the same curl examples but verify they work with the new server

Include a code example showing the multi-agent negotiation:
```python
info["governance"] = {
    "rm_message": [0.35, 1.0, 0.0],      # RM: limit 35%, allow new, don't force reduce
    "pm_message": [0.50, 0.0],             # PM: 50% allocation, no override
    "proposed": {"direction": 1, "size": 0.7},
    "executed": {"direction": 1, "size": 0.35},  # RM clamped size from 0.7 to 0.35
    "interventions": [{"agent": "RiskManager", "type": "size_clamp"}]
}
```

---

### 🟠 7. Rewrite `WRITEUP.md`

**File**: `WRITEUP.md`

Rewrite the narrative:
1. **Problem**: Single-agent governance is fake — it's just business rules. True governance requires independent actors with conflicting incentives
2. **Insight**: PettingZoo AEC enables actual decentralized decision-making. RM is rewarded for restricting risk, Trader for profit, PM for balanced growth. Their tension creates emergent regulatory behavior
3. **Architecture**: 3-agent AEC cycle, inter-agent messages in observation space, adversarial reward structure
4. **Training**: Multi-agent GRPO with alternating optimization
5. **Results**: Per-agent reward curves, compliance rate improvement, RM learned to restrict, Trader learned to comply
6. **Why it matters**: First true PettingZoo multi-agent governance env for finance. Generalizes to healthcare/autonomous systems oversight

---

### 🟠 8. Build PettingZoo-Compatible GRPO Pipeline for Qwen 2.5

**New File**: `training/train_grpo_multiagent.py`

This is the most important training change. Create a GRPO trainer that:

1. Uses `MultiAgentTradingEnv` as the environment
2. Trains the Trader agent as a Qwen 2.5-1.5B model using Unsloth + TRL `GRPOTrainer`
3. RM and PM can use rule-based policies during Trader training (alternating optimization)
4. The Trader's prompt must include the RM/PM messages (constraints, allocation) as part of the state description so the LLM can reason about them
5. Adapt the 5 existing GRPO verifiers from `reward.py`:
   - `format_reward_func` — same (check `<thought>` + `<action>` tags)
   - `alignment_reward_func` — same (anti-hallucination)
   - `risk_reward_func` — update to use RM's `size_limit` from the message instead of hardcoded limit
   - `profit_reward_func` — same (direction vs price trend)
   - `governance_reward_func` — update to check if Trader's proposed size ≤ RM's size_limit (dynamic, not static)
6. The key differentiator: the governance verifier now checks compliance against *learned* RM constraints, not hardcoded ones. This means the Trader must learn to read and respect the RM message in its observation

Example prompt format for Qwen:
```
You are a trading agent in a multi-agent governance system. 
The Risk Manager has set the following constraints: size_limit=0.35, new_positions=allowed, force_reduce=no.
The Portfolio Manager allocated: capital_cap=0.50, override=none.
Market state: [... 24 values ...]
Your task: Propose a trade action that maximizes profit while respecting the governance constraints.
<thought>Your reasoning here</thought>
<action>{"direction": 1, "size": 0.30, "sl": 49000, "tp": 52000}</action>
```

---

### 🟠 9. Rewrite `mate_training.ipynb`

**File**: `mate_training.ipynb`

Rewrite the Colab notebook to:
1. Install pettingzoo, openenv, trl, unsloth
2. Import `MultiAgentTradingEnv`
3. Run GRPO training via the new `train_grpo_multiagent.py` pipeline
4. Generate and display loss/reward plots inline
5. Save plots as `.png` in the `plots/` directory
6. Must be fully re-runnable on Google Colab T4 GPU

---

### 🟡 10. Multi-Agent Reward Visualization Script

**New File**: `training/plot_multiagent.py`

Create a script that:
- Loads training logs from the GRPO run
- Plots per-agent rewards (RM, PM, Trader) on same axes
- Plots governance intervention rate over training
- Plots compliance rate (% of Trader actions passing without RM/PM override)
- Saves all to `plots/` as `.png` with labeled axes and titles

---

### 🟡 11. Strengthen Theme #1 Alignment in README

Add a dedicated section in README:
```markdown
## 🎯 Theme Alignment: Multi-Agent Interactions (Theme #1)

QuantHive directly addresses Theme #1 and both sub-themes:

- **Fleet AI — Scalable Oversight**: The Risk Manager and Portfolio Manager are oversight agents that monitor and constrain the Trader in real-time, creating scalable governance.
- **Halluminate — Multi-Actor Environments**: Three independent actors with adversarial incentives negotiate through observation message-passing, producing emergent strategic behavior.

The PettingZoo AEC architecture enables theory-of-mind reasoning: the Trader must model what constraints the Risk Manager will impose based on the current portfolio state.
```

---

### 🟡 12. Document Anti-Reward-Hacking in WRITEUP

Add a section explaining how the adversarial reward structure inherently prevents gaming:
- If the Trader learns to ignore RM limits → RM is rewarded for clamping → arms race
- If RM always blocks → RM gets no upside from portfolio growth → it learns moderation
- Multiple independent reward signals per agent (not one monolithic score)
- Governance intervention log provides process-level reward, not just final outcome

---

### 🟡 13. Verify Curriculum Learning Works with PettingZoo Env

Test that `MultiAgentTradingEnv(difficulty="easy")`, `"medium"`, `"hard"` all work correctly:
- Run 10 episodes at each difficulty
- Confirm the Trader gets non-zero reward at "easy" difficulty
- Mention curriculum design in WRITEUP

---

### 🟢 14. Update UI to Show Agent Negotiation

Update the React UI (`ui/src/`) to:
- Show RM → PM → Trader turn order visually
- Display RM message [size_limit, allow_new, force_reduce] and PM message [cap_alloc, override] each cycle
- Flash when an intervention occurs (RM clamped size, PM vetoed trade)
- Show per-agent reward bars

---

### 🟢 15. Prepare Slide Deck for 3-Min Pitch

Create a 6-slide deck:
1. Problem: "AI agents can't govern each other"
2. Solution: PettingZoo AEC with 3 adversarial agents
3. Architecture: RM → PM → Trader cycle + message passing diagram
4. Key innovation: Adversarial rewards = emergent self-regulation
5. Results: Per-agent reward curves + compliance improvement
6. Demo: Live UI showing negotiation

---

### 🟢 16. Upload Trained Model to HF Hub

After training completes:
- Save the LoRA adapter for Qwen 2.5-1.5B
- Upload to HF Hub (e.g., `ARKAISW/quanthive-trader-lora`)
- Link from README

---

### 🟢 17. Record <2 Min Video Demo

- Screen record the UI showing multi-agent negotiation
- Show before/after: random Trader vs trained Trader
- Upload to YouTube (URL only, no video files in repo)
- Link from README

---

### 🟢 18. Run PettingZoo API Test

Run PettingZoo's built-in compliance test to verify the env is properly implemented:
```python
from pettingzoo.test import api_test
from env.multi_agent_env import MultiAgentTradingEnv
env = MultiAgentTradingEnv()
api_test(env, num_cycles=50, verbose_progress=True)
```
Fix any issues that arise. Mention passing this test in README as quality evidence.
