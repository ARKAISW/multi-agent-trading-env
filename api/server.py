from pathlib import Path
import asyncio

import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from agents.fa_agent import FundamentalAnalyst
from agents.portfolio_manager import PortfolioManager
from agents.researcher import QuantResearcher
from agents.risk_model import RiskModeler
from agents.trader import QuantTrader
from env.trading_env import TradingEnv
from training.config import TrainingConfig


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIST = ROOT_DIR / "ui" / "dist"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def make_initial_state():
    return {
        "is_running": False,
        "current_step": 0,
        "agents": {
            "Researcher": {"message": "Scanning the tape.", "confidence": 0.0, "status": "idle"},
            "Fundamental Analyst": {"message": "Watching macro tone.", "confidence": 0.0, "status": "idle"},
            "Risk Manager": {"message": "Limits standing by.", "confidence": 0.0, "status": "idle"},
            "Trader": {"message": "Desk is flat.", "confidence": 0.0, "status": "idle"},
            "Portfolio Manager": {"message": "Waiting for conviction.", "confidence": 0.0, "status": "idle"},
        },
        "portfolio": {"value": 100000.0, "cash": 100000.0, "positions": {}},
        "metrics": {"reward": 0.0, "grade": 0.0, "drawdown": 0.0, "sharpe": 0.0},
        "chart": {"price": 50000.0, "trade": None, "price_change": 0.0},
        "trade": {
            "pulse": 0,
            "side": "HOLD",
            "size": 0.0,
            "price": 50000.0,
            "sl": 0.0,
            "tp": 0.0,
            "portfolio_delta": 0.0,
            "notional": 0.0,
            "reason": "Waiting for the first coordinated decision.",
            "override": False,
        },
        "flow": [],
        "engine": {
            "name": "Desk Policy",
            "mode": "Rule Fallback",
            "policy_active": False,
            "note": "Local policy is disabled by default for demo builds. Enable USE_LOCAL_POLICY=true after mounting a trained model.",
        },
    }


sim_state = make_initial_state()


class SimulationRunner:
    def __init__(self):
        self.config = TrainingConfig(tickers=["AAPL"], fast_mode=False, max_steps=100)
        self.env = TradingEnv(
            df=None,
            initial_cash=self.config.initial_cash,
            ticker=self.config.tickers[0],
            commission=self.config.commission,
            reward_weights=self.config.reward_weights,
            max_steps=self.config.max_steps,
        )

        self.researcher = QuantResearcher()
        self.fa_agent = FundamentalAnalyst(fast_mode=self.config.fast_mode)
        self.risk_model = RiskModeler(
            max_drawdown_limit=self.config.risk_max_drawdown,
            max_exposure=self.config.risk_max_exposure,
            vol_threshold=self.config.risk_vol_threshold,
        )
        self.trader = QuantTrader()
        self.portfolio_manager = PortfolioManager(fast_mode=self.config.fast_mode)

        self.obs, self.info = self.env.reset()
        self.done = False

        sim_state["engine"] = {
            "name": "Qwen2.5 Trading Desk",
            "mode": "Local SLM Live" if self.trader.policy.is_active else "Rule Fallback",
            "policy_active": self.trader.policy.is_active,
            "note": (
                "Trader is following the locally loaded policy model."
                if self.trader.policy.is_active
                else "Trader is using the fallback rule policy until a local SLM is enabled."
            ),
        }

    def step(self):
        if self.done:
            self.obs, self.info = self.env.reset()
            self.done = False
            self.fa_agent.reset()
            self.portfolio_manager.reset()

        global sim_state

        previous_value = sim_state["portfolio"]["value"]
        previous_price = sim_state["chart"]["price"]
        current_price = self.env.market.current_price()

        r_sig, r_conf, r_reasoning = self.researcher(self.obs)
        researcher_message = f"{r_sig.title()} bias. {r_reasoning}"
        sim_state["agents"]["Researcher"] = {
            "message": researcher_message,
            "confidence": r_conf,
            "status": "active",
        }

        fa_sent, fa_reasoning = self.fa_agent(self.obs)
        sim_state["agents"]["Fundamental Analyst"] = {
            "message": fa_reasoning,
            "confidence": abs((fa_sent * 2.0) - 1.0),
            "status": "active",
        }

        r_lim, r_cons, risk_reasoning = self.risk_model(self.obs)
        r_cons["raw_price"] = current_price
        sim_state["agents"]["Risk Manager"] = {
            "message": f"Limit {r_lim:.2f}. {risk_reasoning}",
            "confidence": 1.0,
            "status": "active" if r_lim < 0.6 else "idle",
        }

        direction, size, sl, tp, trader_reasoning = self.trader(
            self.obs,
            (r_sig, r_conf, r_reasoning),
            (fa_sent, fa_reasoning),
            (r_lim, r_cons, risk_reasoning),
        )
        dir_str = ["HOLD", "BUY", "SELL"][direction]
        sim_state["agents"]["Trader"] = {
            "message": f"{dir_str} {size:.2f}. {trader_reasoning}",
            "confidence": size,
            "status": "active" if direction != 0 else "idle",
        }

        cap_alloc, override = self.portfolio_manager(self.obs, (direction, size), self.info)
        pm_message = "Approved. Keep the lane clear."
        if override:
            direction, size = override
            dir_str = ["HOLD", "BUY", "SELL"][direction]
            pm_message = f"Override to {dir_str} {size:.2f}"

        sim_state["agents"]["Portfolio Manager"] = {
            "message": pm_message,
            "confidence": cap_alloc,
            "status": "active" if override or cap_alloc < 0.6 else "idle",
        }

        action = {
            "direction": direction,
            "size": np.array([size], dtype=np.float32),
            "sl": np.array([sl], dtype=np.float32),
            "tp": np.array([tp], dtype=np.float32),
        }
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.done = terminated or truncated

        new_price = self.env.market.current_price()
        portfolio_value = self.info.get("portfolio_value", 0.0)
        portfolio_delta = portfolio_value - previous_value
        price_change = new_price - previous_price
        sim_state["current_step"] = self.env.current_step

        sim_state["portfolio"] = {
            "value": portfolio_value,
            "cash": self.info.get("cash", 0.0),
            "positions": self.info.get("positions", {}),
        }
        sim_state["metrics"] = {
            "reward": reward,
            "grade": self.info.get("grade", 0.0),
            "drawdown": self.info.get("max_drawdown", 0.0),
            "sharpe": self.info.get("normalized_sharpe", 0.0),
        }
        sim_state["chart"] = {
            "price": new_price,
            "trade": dir_str if direction != 0 else None,
            "price_change": price_change,
        }
        sim_state["trade"] = {
            "pulse": sim_state["trade"]["pulse"] + 1,
            "side": dir_str,
            "size": float(size),
            "price": float(current_price),
            "sl": float(sl),
            "tp": float(tp),
            "portfolio_delta": float(portfolio_delta),
            "notional": float(previous_value * size if direction != 0 else 0.0),
            "reason": trader_reasoning,
            "override": bool(override),
        }
        sim_state["flow"] = [
            {"from": "Researcher", "to": "Risk Manager", "strength": float(r_conf), "active": True, "tone": "signal"},
            {"from": "Researcher", "to": "Portfolio Manager", "strength": float(r_conf), "active": r_sig != "neutral", "tone": "research"},
            {
                "from": "Fundamental Analyst",
                "to": "Portfolio Manager",
                "strength": float(abs((fa_sent * 2.0) - 1.0)),
                "active": True,
                "tone": "macro",
            },
            {"from": "Risk Manager", "to": "Trader", "strength": float(1.0 - r_lim), "active": True, "tone": "risk"},
            {"from": "Portfolio Manager", "to": "Trader", "strength": float(cap_alloc), "active": True, "tone": "approval"},
            {"from": "Trader", "to": "Market", "strength": float(size), "active": direction != 0, "tone": dir_str.lower()},
        ]


runner = None


async def simulation_loop():
    global sim_state, runner
    if runner is None:
        runner = SimulationRunner()

    while sim_state["is_running"]:
        runner.step()
        await asyncio.sleep(1.0)


@app.get("/state")
@app.get("/api/state")
def get_state():
    return sim_state


@app.post("/start")
@app.post("/api/start")
async def start_sim(background_tasks: BackgroundTasks):
    global sim_state
    if not sim_state["is_running"]:
        sim_state["is_running"] = True
        background_tasks.add_task(simulation_loop)
    return {"status": "started"}


@app.post("/stop")
@app.post("/api/stop")
def stop_sim():
    global sim_state
    sim_state["is_running"] = False
    return {"status": "stopped"}


@app.post("/api/step")
def step_sim():
    global runner
    if runner is None:
        runner = SimulationRunner()
    runner.step()
    return {"status": "stepped"}


# --- OpenEnv Standard Endpoints for Judges ---

@app.post("/openenv/reset")
async def openenv_reset():
    """Standard OpenEnv reset endpoint for remote evaluators."""
    global runner
    if runner is None:
        runner = SimulationRunner()
    obs, info = runner.env.reset()
    # Ensure obs is a list for JSON serialization
    return {"observation": obs.tolist(), "info": info}


@app.post("/openenv/step")
async def openenv_step(action: dict):
    """Standard OpenEnv step endpoint for remote evaluators."""
    global runner
    if runner is None:
        runner = SimulationRunner()
    
    # The action coming from a remote OpenEnv client is usually a dict
    # but we need to ensure numpy arrays for our TradingEnv
    formatted_action = {
        "direction": int(action.get("direction", 0)),
        "size": np.array([float(action.get("size", 0.0))], dtype=np.float32),
        "sl": np.array([float(action.get("sl", 0.0))], dtype=np.float32),
        "tp": np.array([float(action.get("tp", 0.0))], dtype=np.float32),
    }
    
    obs, reward, terminated, truncated, info = runner.env.step(formatted_action)
    
    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info
    }


if FRONTEND_DIST.exists():
    @app.get("/")
    def serve_index():
        return FileResponse(FRONTEND_DIST / "index.html")


    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        asset_path = FRONTEND_DIST / full_path
        if full_path and asset_path.exists() and asset_path.is_file():
            return FileResponse(asset_path)
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/")
    def demo_not_built():
        return JSONResponse(
            {
                "message": "Frontend bundle not found. Run `npm install && npm run build` inside `ui/`.",
                "frontend_dist": str(FRONTEND_DIST),
            }
        )


def run_server():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    run_server()
