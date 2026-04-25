"""
Multi-Agent RL Trading Environment — Entry Point

Usage:
    python app.py                              # Default: 100 episodes, dummy data
    python app.py --episodes 50 --ticker AAPL  # Custom config
    python app.py --evaluate                   # Run evaluation with baseline comparison
"""

import argparse
import sys

from training.config import TrainingConfig
from training.train import train, run_random_baseline
from utils.evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Agent RL Trading Environment"
    )
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of training episodes")
    parser.add_argument("--ticker", type=str, default="AAPL",
                        help="Stock ticker symbol")
    parser.add_argument("--start", type=str, default="2023-01-01",
                        help="Data start date")
    parser.add_argument("--end", type=str, default="2024-12-31",
                        help="Data end date")
    parser.add_argument("--cash", type=float, default=100_000.0,
                        help="Initial cash")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation with random baseline comparison")
    parser.add_argument("--fetch-data", action="store_true",
                        help="Fetch real market data before training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log metrics every N episodes")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps per episode")
    parser.add_argument("--gbm", action="store_true",
                        help="Use synthetic GBM data instead of real data")
    parser.add_argument("--fast", action="store_true",
                        help="Run a quick test with fewer episodes/steps and no LLM")
    parser.add_argument("--demo", action="store_true",
                        help="Launch the Demo UI backend API server")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.demo:
        print("Starting Demo API Server on port 7860...")
        import uvicorn
        from api.server import app
        uvicorn.run(app, host="0.0.0.0", port=7860)
        return

    config = TrainingConfig(
        tickers=[args.ticker],
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.cash,
        num_episodes=2 if args.fast else args.episodes,
        seed=args.seed,
        log_every=args.log_every,
        max_steps=50 if args.fast else args.max_steps,
        fast_mode=args.fast,
    )

    # Optionally fetch real data or generate GBM
    df = None
    if args.gbm:
        from env.trading_env import TradingEnv
        print("Generating synthetic GBM data (mu=0.1, sigma=0.2)...")
        env_gen = TradingEnv()
        df = env_gen._make_dummy_data_from_profile(n=500, mu=0.1, sigma=0.2)
        print(f"Generated {len(df)} rows of GBM data.\n")
    elif args.fetch_data:
        from data.fetch_data import fetch_yfinance
        print(f"Fetching data for {args.ticker} from {args.start} to {args.end}...")
        df = fetch_yfinance(args.ticker, args.start, args.end)
        print(f"Loaded {len(df)} rows of market data.\n")

    if args.evaluate:
        results = evaluate(config, df=df)
        print(f"\nGrade improvement: {results['grade_improvement']:+.4f}")
    else:
        metrics = train(config, df=df)
        print(f"\nDone! {len(metrics)} episodes completed.")


if __name__ == "__main__":
    main()
