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


def check_env_step() -> None:
    env = TradingEnv()
    obs, info = env.reset()
    assert len(obs) == 23, f"unexpected observation size: {len(obs)}"
    assert 0.0 <= info["grade"] <= 1.0, f"grade out of range: {info['grade']}"

    obs, reward, terminated, truncated, info = env.step(
        {"direction": 1, "size": [0.25], "sl": [0.0], "tp": [0.0]}
    )
    assert len(obs) == 23, "step observation size changed"
    assert 0.0 <= reward <= 1.0, f"reward out of range: {reward}"
    assert not terminated, "environment terminated too early"
    assert not truncated, "environment truncated unexpectedly"
    assert info["trade_count"] >= 1, "buy action did not register a trade"


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

    check_training_loop()
    print("training_loop_ok")

    check_demo_runner()
    print("demo_runner_ok")

    print("smoke_test_passed")


if __name__ == "__main__":
    main()
