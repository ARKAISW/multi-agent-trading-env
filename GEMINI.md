# GEMINI.md - Multi-Agent RL Trading Environment

## Project Overview
This project is a **Multi-Agent Reinforcement Learning (RL) Trading Environment** built for the **OpenEnv April '26 Hackathon** (Theme #1: Multi-Agent Interactions). It features five specialized agents cooperating to trade in a simulated market environment compliant with the `OpenEnv` (Gymnasium-based) standard.

### Core Architecture
The system is divided into five specialized agents that interact to make trading decisions:
1.  **Quant Researcher (`agents/researcher.py`)**: Generates technical signals (RSI, EMA, Bollinger Bands, MACD).
2.  **Fundamental Analyst (`agents/fa_agent.py`)**: Provides momentum-driven sentiment bias.
3.  **Risk Modeler (`agents/risk_model.py`)**: Enforces drawdown limits and dynamic position sizing based on volatility.
4.  **Quant Trader (`agents/trader.py`)**: The executor agent that aggregates signals and risk constraints to determine final trade direction and size.
5.  **Portfolio Manager (`agents/portfolio_manager.py`)**: Oversees cumulative performance and can override decisions during severe drawdowns.

### Key Components
-   **`app.py`**: The main entry point for training and evaluation.
-   **`env/`**: Contains the `trading_env.py` (Gymnasium environment), `state.py` (observation space), and `reward.py` (normalized reward logic).
-   **`training/`**: Manages the training loop and configuration.
-   **`data/`**: Handles market data ingestion from `yfinance` and `ccxt`.
-   **`utils/`**: Provides technical indicators, visualization tools, and evaluation benchmarks.
-   **`plots/`**: Automatically generated visualizations of training progress and agent performance.

---

## Tech Stack
-   **Language**: Python 3.x
-   **RL Framework**: OpenEnv, Gymnasium, PyTorch
-   **Data Science**: Pandas, NumPy, Matplotlib, Plotly
-   **Financial Data**: `yfinance`, `ccxt`
-   **LLM/Agents**: Transformers, Datasets, TRL, Unsloth, LangChain
-   **UI (Planned)**: React, Tailwind CSS, Framer Motion (Indie-style office simulation)

---

## Building and Running

### Prerequisites
Install dependencies using pip:
```bash
pip install -r requirements.txt
```

### Execution Commands
-   **Train (Default)**: Run with 100 episodes on dummy/default data.
    ```bash
    python app.py
    ```
-   **Train with Real Data**: Fetch data for a specific ticker and time range.
    ```bash
    python app.py --fetch-data --ticker AAPL --start 2023-01-01 --end 2024-12-31
    ```
-   **Evaluate**: Run evaluation and compare against a random baseline.
    ```bash
    python app.py --evaluate
    ```
-   **Custom Configuration**:
    ```bash
    python app.py --episodes 50 --cash 50000 --seed 123
    ```

---

## Development Conventions

### CRITICAL: Normalization
**All rewards, evaluation outputs, and grades MUST be normalized between 0 and 1.**
-   `0 ≤ reward ≤ 1`
-   `0 ≤ grade ≤ 1`
-   `0 ≤ confidence ≤ 1`

### ALWAYS REMEMBER: Documentation
-   **Summary Update**: After every significant change or task completion, you **MUST** add a summary of what you did to `summary.md`.

### Coding Style
-   **Modular Design**: Logic is separated into specialized directories (`agents`, `env`, `training`, `utils`).
-   **Docstrings**: All classes and methods should include descriptive docstrings explaining parameters and return values.
-   **Type Hinting**: Use Python type hints (e.g., `Dict`, `Tuple`, `np.ndarray`) for clarity and maintainability.
-   **Agent Pattern**: Agents are implemented as classes with a `__call__` method for decision-making.

### Testing & Validation
-   **Evaluation Mode**: Use `python app.py --evaluate` to benchmark changes against the random baseline.
-   **Visual Verification**: Check the `plots/` directory after training runs to verify reward curves and grading progression.

### Data Management
-   Real-world data should be fetched using `data/fetch_data.py`.
-   Training metrics are persisted in `checkpoints/training_metrics.csv`.

