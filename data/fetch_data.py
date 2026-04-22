"""
Data fetching utilities for historical market data.
Supports yfinance (equities) and ccxt (crypto).
"""

import os
import argparse
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False)
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df.dropna(inplace=True)
    return df


def fetch_ccxt(exchange_id: str, symbol: str, timeframe: str = "1d",
               limit: int = 500) -> pd.DataFrame:
    """Fetch OHLCV data from a crypto exchange via ccxt."""
    import ccxt

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("date", inplace=True)
    df.drop(columns=["timestamp"], inplace=True)
    return df


def save_data(df: pd.DataFrame, filename: str) -> str:
    """Save dataframe to CSV in the data directory."""
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath)
    print(f"Saved {len(df)} rows to {filepath}")
    return filepath


def load_data(filename: str) -> pd.DataFrame:
    """Load a previously saved CSV data file."""
    filepath = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch market data")
    parser.add_argument("--source", choices=["yfinance", "ccxt"], default="yfinance")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--exchange", default="binance", help="ccxt exchange id")
    parser.add_argument("--symbol", default="BTC/USDT", help="ccxt symbol")
    args = parser.parse_args()

    if args.source == "yfinance":
        data = fetch_yfinance(args.ticker, args.start, args.end)
        save_data(data, f"{args.ticker}_{args.start}_{args.end}.csv")
    else:
        data = fetch_ccxt(args.exchange, args.symbol)
        safe_name = args.symbol.replace("/", "_")
        save_data(data, f"{safe_name}_{args.exchange}.csv")
