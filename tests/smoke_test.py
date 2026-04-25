"""
Lightweight smoke test for the hackathon demo environment.

Run:
    python tests/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.server import SimulationRunner
from env.trading_env import TradingEnv
from training.config import TrainingConfig
from training.train import train

OBS_SIZE = 24  # 14 market + 5 portfolio + 5 risk


def check_env_step() -> None:
    env = TradingEnv()
    obs, info = env.reset()
    assert len(obs) == OBS_SIZE, f"unexpected observation size: {len(obs)}"
    assert 0.0 <= info["grade"] <= 1.0, f"grade out of range: {info['grade']}"

    obs, reward, terminated, truncated, info = env.step(
        {"direction": 1, "size": [0.25], "sl": [0.0], "tp": [0.0]}
    )
    assert len(obs) == OBS_SIZE, "step observation size changed"
    assert -1.0 <= reward <= 1.0, f"reward out of range: {reward}"
    assert not terminated, "environment terminated too early"
    assert not truncated, "environment truncated unexpectedly"
    assert info["trade_count"] >= 1, "buy action did not register a trade"


def check_short_selling() -> None:
    """Verify that short selling works end-to-end via direct env actions."""
    env = TradingEnv(max_steps=50)
    obs, info = env.reset()

    # Open a short position (direction=2 from flat = open short)
    obs, reward, terminated, truncated, info = env.step(
        {"direction": 2, "size": [0.3], "sl": [0.0], "tp": [0.0]}
    )
    position_qty = env.portfolio.positions.get("default", 0.0)
    assert position_qty < 0, f"Short position should be negative, got {position_qty}"
    assert info["trade_count"] >= 1, "short action did not register a trade"

    # Cover the short (buy to close, direction=1)
    obs, reward, terminated, truncated, info = env.step(
        {"direction": 1, "size": [1.0], "sl": [0.0], "tp": [0.0]}
    )
    position_qty = env.portfolio.positions.get("default", 0.0)
    assert abs(position_qty) < 1e-8, f"Short should be fully covered, got {position_qty}"


def check_short_sl_tp() -> None:
    """Verify SL/TP execution for short positions."""
    env = TradingEnv(difficulty="easy", max_steps=200)
    obs, info = env.reset()

    current_price = env.market.current_price()
    # Set SL above entry (will get hit if price rises)
    sl_price = current_price * 1.05
    # Set TP below entry (will get hit if price falls)
    tp_price = current_price * 0.95

    obs, reward, terminated, truncated, info = env.step(
        {"direction": 2, "size": [0.2], "sl": [sl_price], "tp": [tp_price]}
    )
    position_qty = info["positions"].get("default", 0.0)
    assert position_qty < 0, f"Expected short position, got {position_qty}"

    # Verify SL/TP are stored
    assert env.portfolio.stop_losses.get("default") == sl_price, "SL not set for short"
    assert env.portfolio.take_profits.get("default") == tp_price, "TP not set for short"

    # Run until SL or TP hits or max steps
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(
            {"direction": 0, "size": [0.0], "sl": [0.0], "tp": [0.0]}
        )
        if terminated or truncated:
            break
        # Check if sl/tp was triggered (position closed)
        position_qty = info["positions"].get("default", 0.0)
        if abs(position_qty) < 1e-8:
            # SL or TP was hit, check trade history
            last_trade = env.portfolio.trade_history[-1]
            assert last_trade["action"] == "cover", f"Expected cover action, got {last_trade['action']}"
            assert last_trade["reason"] in ("stop_loss", "take_profit"), f"Unexpected reason: {last_trade['reason']}"
            break


def check_training_loop() -> None:
    config = TrainingConfig(
        num_episodes=1,
        fast_mode=True,
        tickers=["AAPL"],
        max_steps=10,
    )
    metrics = train(config)
    assert len(metrics) == 1, f"expected one episode, got {len(metrics)}"
    required_keys = {
        "episode",
        "total_reward",
        "mean_reward",
        "final_grade",
        "final_value",
        "pnl_pct",
        "max_drawdown",
        "sharpe_ratio",
        "trade_count",
    }
    missing = required_keys.difference(metrics[0])
    assert not missing, f"training metrics missing keys: {sorted(missing)}"

    trajectory_path = Path(config.save_dir) / config.trajectories_file
    assert trajectory_path.exists(), f"trajectory file missing: {trajectory_path}"
    assert trajectory_path.stat().st_size > 0, "trajectory file is empty"


def check_demo_runner() -> None:
    runner = SimulationRunner()
    runner.step()
    info = runner.info
    assert "positions" in info, "runner info missing positions"
    assert "normalized_sharpe" in info, "runner info missing normalized_sharpe"
    assert 0.0 <= info["grade"] <= 1.0, f"runner grade out of range: {info['grade']}"


def main() -> None:
    check_env_step()
    print("env_step_ok")

    check_short_selling()
    print("short_selling_ok")

    check_short_sl_tp()
    print("short_sl_tp_ok")

    check_training_loop()
    print("training_loop_ok")

    check_demo_runner()
    print("demo_runner_ok")

    print("smoke_test_passed")


if __name__ == "__main__":
    main()
