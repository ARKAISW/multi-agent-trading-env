# 🎮 UI Design Specification — Cutesy Quant Firm Simulation

## Overview

This module defines a **2D indie-style visualization layer** for the Multi-Agent RL Trading Environment.

The goal is to transform abstract agent interactions, trading decisions, and reward signals into a **visually intuitive, engaging simulation** resembling a small quant firm office.

This directly supports:

* Multi-agent interaction clarity
* Reward and learning visualization
* Storytelling for demo and judging

---

## 🧠 Core Concept

A **“living office” simulation** where:

* Each AI agent is represented as a character
* Agents communicate via visible messages
* Decisions affect a shared portfolio
* Learning is visualized over time

---

## 🎨 Art Style Specification

### Style Choice

* 2D pixel-art / stylized indie aesthetic
* Soft pastel color palette
* Minimal but expressive character design

### Rationale

* Pixel art is widely used for clarity and simplicity in 2D systems ([gamemaker.io][1])
* It provides a **cozy, interpretable visual layer** rather than overwhelming realism
* Low-resolution sprites enhance readability and system understanding

### Style Rules

Define consistently:

* Resolution (e.g., 32x32 or 64x64 sprites)
* Color palette (role-based colors)
* Outline thickness
* Animation frame count
* Character proportions

A consistent style guide improves visual coherence and scalability ([Sprite-AI][2])

---

## 🏢 Office Layout

### Structure

```
┌────────────────────────────┐
│        📈 Balance Panel     │
│                            │
│  🧠 Researcher     💻 Trader │
│                            │
│  📊 Risk Modeler   👑 PM    │
│                            │
│        📉 Chart Panel       │
└────────────────────────────┘
```

---

### Zones

1. **Top Panel**

   * Portfolio balance
   * Live PnL indicator

2. **Agent Floor**

   * Each agent at a fixed workstation
   * Communication visible between agents

3. **Bottom Panel**

   * Market chart
   * Trade markers

---

## 🤖 Agent Representation

Each agent is visualized as:

* A small animated character (sprite)
* A workstation (desk + monitor)
* A role-specific color theme

---

### Agent Roles

#### Quant Researcher

* Visual cues: charts, floating indicators
* Behavior: signal generation

#### Trader

* Visual cues: multiple monitors
* Behavior: executes trades

#### Risk Modeler

* Visual cues: warning icons
* Behavior: restricts exposure

#### Portfolio Manager

* Visual cues: elevated seat / calm posture
* Behavior: override authority

---

## 💬 Communication System

### Objective

To visually demonstrate **multi-agent reasoning and coordination**, as required by the theme.

---

### Implementation

* Speech bubbles above agents
* Message transitions between agents
* Short-term visible history

---

### Example Flow

```
Researcher → "RSI oversold, bullish bias"
Risk → "Volatility high, reduce size"
Trader → "Executing reduced position"
PM → "Approved"
```

---

### Design Notes

* Messages should be concise
* Fade after a short duration
* Color-coded by agent

---

## 📈 Trading Visualization

### Balance Panel (Top Right)

Displays:

* Portfolio value (live)
* PnL change

Animations:

* Green pulse → profit
* Red flash → loss

---

### Chart Panel

Displays:

* Price time series
* Trade markers:

  * Buy → green marker
  * Sell → red marker

---

### Metrics Panel

All values normalized to [0, 1]:

* Reward
* Grade
* Drawdown
* Sharpe proxy

---

## 🧠 Learning Visualization

### Objective

Clearly demonstrate **agent improvement over time**

---

### Features

#### 1. Before vs After Toggle

* Pre-training behavior
* Post-training behavior

---

#### 2. Performance Graphs

* Reward vs episode
* Grade vs episode
* Drawdown trend

---

#### 3. Feedback Animation

* Good trade → green highlight
* Bad trade → red highlight

---

## ⚙️ System Modes

### Fast Mode

* No animations
* No API calls
* Used for debugging

---

### Demo Mode

* Full UI enabled
* All agents active
* Communication visible

---

## 🔌 Backend → UI Interface

### API Contract

```json
{
  "agents": [
    {
      "name": "Trader",
      "message": "Executing buy",
      "confidence": 0.78
    }
  ],
  "portfolio": {
    "value": 102000,
    "pnl": 2000
  },
  "metrics": {
    "reward": 0.72,
    "grade": 0.68
  },
  "trades": []
}
```

---

## 🧠 Design Principles

1. **Clarity over realism**
   Visuals must explain behavior, not just look good

2. **State visibility**
   Every important decision should be observable

3. **Agent identity**
   Each agent must feel distinct

4. **Learning visibility**
   Improvement must be obvious without explanation

---

## 🎯 Success Criteria

The UI is successful if:

* A viewer can understand agent interaction without reading code
* Decisions and conflicts are visually clear
* Learning progression is observable
* The system feels alive and coordinated

---

## 🚀 Final Note

The UI is not just decoration — it is a **core storytelling layer**.

It should communicate:

> “This is a system of agents learning, collaborating, and improving under constraints.”

---

[1]: https://gamemaker.io/en/blog/2d-game-art-styles?utm_source=chatgpt.com "The Ultimate Guide To 2D Video Game Art Styles"
[2]: https://www.sprite-ai.art/blog/2d-pixel-art-style-guide?utm_source=chatgpt.com "2D pixel art style guide for games [with examples]"
