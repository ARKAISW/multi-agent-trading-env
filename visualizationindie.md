# 2D Indie-Style Multi-Agent Trading Office UI

## Overview

This UI represents the multi-agent trading system as a **2D indie-style office simulation**, where each agent is visualized as a character performing their role in real time.

The interface combines:

* **visual storytelling (agents as humans)**
* **real-time decision flow**
* **actual trading metrics (PnL, grade, reward)**

The goal is to make agent coordination, conflict, and learning **visually intuitive and engaging**.

---

## Core Concept

The UI is a **live simulation scene**, not just a dashboard.

* Each agent sits in a different part of a small office
* Agents communicate via speech bubbles
* Actions trigger animations
* System performance is shown via live metrics

---

## Scene Layout

```id="scene_layout"
+------------------------------------------------------+
|                    BALANCE PANEL                     |
|               $ Value (Animated)                     |
+------------------------------------------------------+

|  Researcher   |   Risk Manager   |   Portfolio Mgr   |
|     👨‍💻        |       🛡️         |        🧭         |

|  FA Analyst   |     Trader       |    Chart Panel    |
|     🌍        |       ⚡         |       📈          |

+------------------------------------------------------+
|         METRICS PANEL (PnL, Grade, Reward)           |
+------------------------------------------------------+
```

---

## Agent Visualization

Each agent is a **2D character (sprite or card)** with:

* Name
* Role
* Current action
* Speech bubble
* Confidence score (0 → 1)
* Status indicator

---

### Agents

#### Quant Researcher 👨‍💻

* Multiple screens with charts
* Generates signals

Example:

* “Bullish breakout (0.82)”

---

#### Fundamental Analyst 🌍

* News panel / scrolling feed
* Provides sentiment

Example:

* “Macro looks positive”

---

#### Risk Manager 🛡️

* Risk gauges / warning signals
* Controls exposure

Example:

* “Reduce size to 2% ⚠️”

---

#### Trader ⚡

* Main execution terminal
* Executes trades

Example:

* “BUY executed @ 42000”

---

#### Portfolio Manager 🧭

* Observes all agents
* Approves / overrides

Example:

* “Approved” / “Reject trade”

---

## Communication System

Agents communicate via **speech bubbles + arrows**

### Format

```id="comm_format"
Sender → Receiver: Message (confidence)
```

### Behavior

* Messages appear near agents
* Fade out after a few seconds
* Color-coded:

  * Green → positive signal
  * Red → warning
  * Blue → neutral

---

## Balance Panel (Top Right)

### Features

* Displays total portfolio value
* Updates every timestep
* Animated changes:

  * Green ↑ for profit
  * Red ↓ for loss

---

## Chart Panel (Right Side)

### Displays

* Price chart (line or candlestick)
* Current position markers
* Trade entry/exit points

---

## Metrics Panel (Bottom)

### Metrics (ALL NORMALIZED 0 → 1)

```id="metrics"
- Reward
- Grade
- Sharpe (normalized)
- Drawdown (inverse normalized)
```

---

### Grade Calculation

```id="grade_formula"
grade =
    0.4 * normalized_profit +
    0.3 * normalized_sharpe +
    0.2 * (1 - normalized_drawdown) +
    0.1 * consistency
```

---

## Animation System

### Required Animations

* Speech bubbles appear/disappear
* Balance number transitions
* Agent state changes:

  * idle → active → decision
* Trade execution flash

---

## Interaction Flow

```id="flow_ui"
1. New state arrives
2. Researcher + FA generate signals
3. Risk evaluates constraints
4. Messages appear
5. Trader executes decision
6. Portfolio manager confirms
7. Balance updates
8. Metrics update
9. Repeat
```

---

## Training Mode Toggle

### Modes

* **Untrained Mode**

  * random decisions
  * unstable balance

* **Trained Mode**

  * coordinated behavior
  * smoother performance

---

## Data Requirements

* Agent outputs (signals, decisions)
* Market data (price)
* Portfolio state
* Reward values (0 → 1)
* Grade values (0 → 1)

---

## Tech Stack

### Core

* React (frontend)
* Tailwind CSS (styling)

---

### Animation

* Framer Motion

---

### Charts

* Recharts OR Chart.js

---

### State Management

* React Context OR Zustand

---

## Installation

```id="install_ui"
# create app
npm create vite@latest trading-ui
cd trading-ui

# install dependencies
npm install

# styling
npm install tailwindcss postcss autoprefixer
npx tailwindcss init -p

# animations
npm install framer-motion

# charts
npm install recharts

# state management
npm install zustand

# optional icons / assets
npm install lucide-react
```

---

## Assets (Optional but Recommended)

* 2D character sprites OR emoji placeholders
* Desk / office background
* Monitor / chart icons

---

## Key Constraints

All dynamic values must be normalized:

```id="constraints"
0 ≤ reward ≤ 1
0 ≤ grade ≤ 1
0 ≤ confidence ≤ 1
```

---

## Design Principles

* Fun but informative
* Visual clarity > realism
* Show decisions, not just results
* Emphasize agent interaction

---

## Expected Outcome

The UI should allow viewers to:

* visually understand agent roles
* observe decision-making in real time
* see coordination and conflict
* track performance improvements

---

## End
