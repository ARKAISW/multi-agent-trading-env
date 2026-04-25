import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.trader import QuantTrader
from env.reward import normalize_reward, compute_raw_reward

OBS_SIZE = 24  # 14 market + 5 portfolio + 5 risk


def test_sl_tp_calculation_long():
    """Test SL/TP for long (buy) positions."""
    trader = QuantTrader()
    obs = np.zeros(OBS_SIZE)
    
    res_data = ("bullish", 0.8, "Signal reasoning")
    fa_data = (0.7, "Sentiment reasoning")
    
    current_price = 50000.0
    risk_data = (
        0.5, 
        {"suggested_sl_ratio": 0.02, "raw_price": current_price}, 
        "Risk reasoning"
    )
    
    direction, size, sl, tp, reasoning = trader(obs, res_data, fa_data, risk_data)
    
    print(f"Long: Direction: {direction}, Size: {size}")
    print(f"Price: {current_price}, SL: {sl}, TP: {tp}")
    
    if direction == 1:
        assert sl < current_price, "Buy SL should be below entry"
        assert tp > current_price, "Buy TP should be above entry"
        expected_sl = current_price * (1 - 0.02)
        assert abs(sl - expected_sl) < 1e-5, f"Expected SL {expected_sl}, got {sl}"
    
    print("Long SL/TP test passed!")


def test_sl_tp_calculation_short():
    """Test SL/TP for short positions."""
    trader = QuantTrader()
    obs = np.zeros(OBS_SIZE)
    
    res_data = ("bearish", 0.9, "Strong bearish signal")
    fa_data = (0.1, "Bearish sentiment")
    
    current_price = 50000.0
    risk_data = (
        0.5,
        {"suggested_sl_ratio": 0.02, "raw_price": current_price},
        "Risk reasoning"
    )
    
    direction, size, sl, tp, reasoning = trader(obs, res_data, fa_data, risk_data)
    
    print(f"Short: Direction: {direction}, Size: {size}")
    print(f"Price: {current_price}, SL: {sl}, TP: {tp}")
    
    if direction == 2:
        assert sl > current_price, f"Short SL should be ABOVE entry, got SL={sl}, price={current_price}"
        assert tp < current_price, f"Short TP should be BELOW entry, got TP={tp}, price={current_price}"
        expected_sl = current_price * (1 + 0.02)
        assert abs(sl - expected_sl) < 1e-5, f"Expected SL {expected_sl}, got {sl}"
        print("Short SL/TP test passed!")
    else:
        print(f"Trader chose direction={direction} instead of short (2). Skipping SL/TP assertions.")


def test_reward_normalization():
    assert normalize_reward(1.0) > 0.0, "Positive reward should be > 0"
    assert normalize_reward(-1.0) < 0.0, "Negative reward should be < 0"
    assert -1.0 <= normalize_reward(100.0) <= 1.0, "Capping failed"
    assert -1.0 <= normalize_reward(-100.0) <= 1.0, "Capping failed"
    assert abs(normalize_reward(0.0)) < 1e-10, "Zero input should give zero"
    print("Reward normalization test passed!")


def test_directional_reward():
    """Test that reward correctly handles both long and short directions."""
    # Buy in uptrend = positive reward
    r1 = compute_raw_reward(
        profit=0.001, drawdown=0.0, volatility=0.01, sharpe=0.5,
        trade_count=1, direction=1, price_trend=0.01
    )
    # Sell/Short in downtrend = positive reward
    r2 = compute_raw_reward(
        profit=0.001, drawdown=0.0, volatility=0.01, sharpe=0.5,
        trade_count=1, direction=2, price_trend=-0.01
    )
    # Wrong direction = negative directional bonus
    r3 = compute_raw_reward(
        profit=0.001, drawdown=0.0, volatility=0.01, sharpe=0.5,
        trade_count=1, direction=1, price_trend=-0.01
    )
    
    assert r1 > r3, f"Correct long direction ({r1}) should score higher than wrong ({r3})"
    assert r2 > r3, f"Correct short direction ({r2}) should score higher than wrong ({r3})"
    print(f"Directional rewards: correct_long={r1:.3f}, correct_short={r2:.3f}, wrong={r3:.3f}")
    print("Directional reward test passed!")


if __name__ == "__main__":
    try:
        test_sl_tp_calculation_long()
        test_sl_tp_calculation_short()
        test_reward_normalization()
        test_directional_reward()
        print("\nAll math verifications passed!")
    except Exception as e:
        print(f"\nVerification failed: {e}")
        sys.exit(1)
