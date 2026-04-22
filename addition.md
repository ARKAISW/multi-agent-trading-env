# Build Instruction: Extend Multi-Agent RL Trading System with Hybrid Policy + 2D Simulation UI

## Context

We already have a fully working **OpenEnv-based multi-agent RL trading environment** with:

* 5 agents (Researcher, FA, Risk, Trader, Portfolio Manager)
* Reward + grading system normalized to [0, 1]
* Training pipeline (`app.py`)
* LLM integrations (Groq + Gemini)
* Fast mode for testing
* Plotting + evaluation tools

This system is stable and should NOT be rewritten.

---

## Objective

Extend the existing system by adding:

1. **Hybrid Policy Architecture (Local Model + API Models)**
2. **Optional Colab Training Pipeline for a Single Local Model**
3. **2D Indie-Style Simulation UI (React-based)**
4. **Clear separation between environment, policy, and reasoning layers**

---

## Part 1: Hybrid Policy System

### Goal

Introduce a **single local trained model** as the main decision policy, while keeping API models (Groq/Gemini) for reasoning.

---

### Requirements

* Local model = **final decision maker**
* API models = **signal + explanation generators**

---

### Data Flow

```text
State →
    FA Agent (Groq) → sentiment
    Researcher → indicators
    ↓
    Local Model → final action (buy/sell/hold + size)
    ↓
Environment → reward → next state
```

---

### Implementation Tasks

* Create `policy/local_model.py`

  * load trained model (placeholder for now)
  * input: structured state + signals
  * output: action JSON

* Modify `quant_trader.py`

  * replace decision logic with local model call

* Keep:

  * FA agent (Groq)
  * Portfolio manager (Gemini)

---

## Part 2: Colab Training Pipeline (Single Model)

### Goal

Provide a minimal, reproducible training pipeline.

---

### Constraints

* Train **ONE model only**
* Use lightweight model:

  * TinyLlama OR Phi-2 OR similar
* Use:

  * Unsloth (preferred) or TRL

---

### Tasks

* Create `training/colab_train.ipynb`

Notebook must:

1. Load environment
2. Generate trajectories
3. Format dataset:

```text
State → Instruction → Action
```

4. Train using LoRA
5. Save adapter weights

---

### Output

* `models/local_policy/` (saved weights)

---

## Part 3: 2D Simulation UI (React)

### Goal

Build a **2D indie-style office simulation UI** that visualizes agents as characters.

---

### Core Design

* Office layout (grid or absolute positioning)
* Each agent = character card

---

### Required Components

#### 1. Agent Cards

Each agent must display:

* name
* role
* current message
* confidence (0–1)
* status

---

#### 2. Communication Layer

* speech bubbles
* message flow between agents

---

#### 3. Balance Panel (Top Right)

* animated portfolio value
* green ↑ / red ↓ transitions

---

#### 4. Chart Panel

* price chart
* trade markers

---

#### 5. Metrics Panel

All normalized [0,1]:

* reward
* grade
* drawdown
* sharpe

---

### Tech Stack

* React (Vite)
* Tailwind
* Framer Motion
* Recharts

---

### Tasks

* Create `/ui` directory
* Build:

  * `AgentCard.jsx`
  * `OfficeScene.jsx`
  * `BalancePanel.jsx`
  * `MetricsPanel.jsx`
  * `ChartPanel.jsx`

---

## Part 4: Backend → UI Bridge

### Goal

Stream environment outputs to UI.

---

### Implementation

* Create simple API (`FastAPI` or Flask)
* Endpoint:

```json
GET /state
{
  "agents": [...],
  "messages": [...],
  "portfolio": {...},
  "metrics": {...}
}
```

---

## Part 5: Modes

### Add 2 execution modes

#### 1. Fast Mode

* already exists
* keep for debugging

#### 2. Demo Mode

* full agent interaction
* UI enabled
* API models active

---

## Constraints

* DO NOT break existing environment
* Maintain normalization:

```text
0 ≤ reward ≤ 1
0 ≤ grade ≤ 1
0 ≤ confidence ≤ 1
```

* Keep modular structure
* Avoid training multiple models

---

## Deliverables

* Working hybrid agent system
* Colab notebook for training
* React UI (office simulation)
* API bridge for live updates

---

## Success Criteria

* Agents interact coherently
* Reward/grade improves over time
* UI clearly shows coordination
* System runs end-to-end in demo mode

---

## End
