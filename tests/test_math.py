import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.trader import QuantTrader
from env.reward import normalize_reward, compute_raw_reward

def test_sl_tp_calculation():
    trader = QuantTrader()
    # Dummy observation (23 features)
    obs = np.zeros(23)
    
    # Mock data
    res_data = ("bullish", 0.8, "Signal reasoning")
    fa_data = (0.7, "Sentiment reasoning")
    
    # Risk data with raw_price
    current_price = 50000.0
    risk_data = (
        0.5, 
        {"suggested_sl_ratio": 0.02, "raw_price": current_price}, 
        "Risk reasoning"
    )
    
    direction, size, sl, tp, reasoning = trader(obs, res_data, fa_data, risk_data)
    
    print(f"Direction: {direction}, Size: {size}")
    print(f"Price: {current_price}, SL: {sl}, TP: {tp}")
    
    assert direction in [1, 2], "Trader should issue a buy or sell signal"
    if direction == 1:
        assert sl < current_price, "Buy SL should be below entry"
        assert tp > current_price, "Buy TP should be above entry"
        # Check specific ratios
        expected_sl = current_price * (1 - 0.02)
        assert abs(sl - expected_sl) < 1e-5, f"Expected SL {expected_sl}, got {sl}"
    
    print("SL/TP test passed!")

def test_reward_normalization():
    # Test normalization bounds
    assert normalize_reward(1.0) >= 0.5, "Positive reward should be >= 0.5"
    assert normalize_reward(-1.0) <= 0.5, "Negative reward should be <= 0.5"
    assert 0.0 <= normalize_reward(5.0) <= 1.0, "Capping failed"
    assert 0.0 <= normalize_reward(-5.0) <= 1.0, "Capping failed"
    print("Reward normalization test passed!")

if __name__ == "__main__":
    try:
        test_sl_tp_calculation()
        test_reward_normalization()
        print("\nAll math verifications passed!")
    except Exception as e:
        print(f"\nVerification failed: {e}")
        sys.exit(1)
