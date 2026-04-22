# 🌌 GeminiTrading: Multi-Agent RL Ecosystem

[![OpenEnv](https://img.shields.io/badge/Environment-OpenEnv-blue.svg)](https://github.com/OpenEnv)
[![Hackathon](https://img.shields.io/badge/Hackathon-OpenEnv%20April%20%2726-orange.svg)](https://hackathon.openenv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance **Multi-Agent Reinforcement Learning (MARL)** trading environment built for the **OpenEnv April '26 Hackathon**. This project features a collaborative ecosystem of five specialized agents working together to navigate market volatility and maximize risk-adjusted returns.

---

## 🏛️ Project Architecture

Our system is divided into five specialized neural/rule-based agents:

| Agent | Responsibility | Key Signals |
| :--- | :--- | :--- |
| **🔍 Quant Researcher** | Signal Discovery | RSI, EMA, Bollinger, MACD |
| **📉 Fundamental Analyst** | Sentiment Bias | News Momentum, Market Mood |
| **🛡️ Risk Modeler** | Capital Preservation | Drawdown limits, Position Sizing |
| **⚔️ Quant Trader** | Execution | Final Trade Direction & Confidence |
| **💼 Portfolio Manager** | Global Oversight | Performance Monitoring & Overrides |

---

## 🛠️ Tech Stack

- **Frameworks**: [OpenEnv](https://github.com/OpenEnv) (Gymnasium-based), PyTorch
- **Data**: `yfinance`, `ccxt`
- **Fine-tuning**: Unsloth, HuggingFace TRL (for 135M CoT fine-tuning)
- **UI**: React (Vite), Tailwind CSS, Framer Motion
- **API**: FastAPI (Uvicorn)

---

## 🚀 Quick Start

### 1. Prerequisites
```bash
# Clone the repository
git clone https://github.com/arkasarkar1507/GeminiTrading.git
cd GeminiTrading

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Training Pipeline
```bash
# Default train (dummy data)
python app.py

# Train with Real Data
python app.py --fetch-data --ticker NVDA --start 2023-01-01 --end 2024-12-31
```

### 3. Evaluate Performance
```bash
python app.py --evaluate
```

---

## 🖥️ Indie Quant Office (UI)

The UI provides a live, indie-style simulation of the trading agents working in an office environment.

```bash
cd ui
npm install
npm run dev
```

---

## 📊 Normalization & Rewards

In compliance with `OpenEnv` standards:
- **Rewards**: Normalized `[0, 1]` based on risk-adjusted profitability.
- **Confidence**: All agent decisions include a normalized confidence score.
- **Grades**: Agent performance is graded continuously on a scale of `0` to `1`.

---

## 📜 Roadmap & Future Work
- [x] Core Environment Compliance
- [x] Multi-Agent Interaction Loop
- [x] 135M Local Policy Integration
- [ ] Live Streaming API (FastAPI Bridge)
- [ ] Advanced Social Trading Features

---

## 🤝 Contributing
This is an entry for the **OpenEnv April '26 Hackathon**. Contributions, stars, and feedback are welcome!

**Author**: [Arka Sarkar](mailto:arkasarkar1507@gmail.com)
