"""
Technical indicators computation for OHLCV data.
"""

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_ema(close: pd.Series, period: int = 20) -> pd.Series:
    """Compute Exponential Moving Average."""
    return close.ewm(span=period, adjust=False).mean()


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26,
                 signal: int = 9) -> tuple:
    """Compute MACD, Signal, and Histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(close: pd.Series, period: int = 20,
                            std_dev: float = 2.0) -> tuple:
    """Compute Bollinger Bands (upper, middle, lower)."""
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def compute_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Compute rolling volatility (std of returns)."""
    returns = close.pct_change()
    return returns.rolling(window=period).std()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR)."""
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators and attach to the dataframe.
    Expects columns: open, high, low, close, volume.
    Returns a copy with indicator columns added.
    """
    df = df.copy()
    close = df["close"]

    # RSI
    df["rsi"] = compute_rsi(close)

    # EMA
    df["ema_20"] = compute_ema(close, 20)
    df["ema_50"] = compute_ema(close, 50)

    # MACD
    macd, macd_signal, macd_hist = compute_macd(close)
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(close)
    df["bb_upper"] = bb_upper
    df["bb_middle"] = bb_middle
    df["bb_lower"] = bb_lower

    # Volatility & ATR
    df["volatility"] = compute_volatility(close)
    df["atr"] = compute_atr(df)

    # Fill NaN from rolling windows
    df = df.bfill()
    df = df.fillna(0)

    return df
