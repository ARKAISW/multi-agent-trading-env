"""
Multi-Agent Trading API Server.

Uses the PettingZoo AEC MultiAgentTradingEnv with three RL agents
(RiskManager → PortfolioManager → Trader) that negotiate each cycle.

Advisory agents (QuantResearcher, FundamentalAnalyst) run in parallel
to enrich the UI with signal context but do NOT participate in the AEC loop.
"""

from pathlib import Path
import asyncio

import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from agents.fa_agent import FundamentalAnalyst
from agents.researcher import QuantResearcher
from env.multi_agent_env import (
    MultiAgentTradingEnv,
    RISK_MANAGER,
    PORTFOLIO_MGR,
    TRADER,
    ALL_AGENTS,
)
# TradingEnv kept for backward compat data generation only (not used in endpoints)
from training.config import TrainingConfig
from training.train_multi_agent import (
    RuleRiskManagerPolicy,
    RulePortfolioManagerPolicy,
    RuleTraderPolicy,
)


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
        # Five logical agents for the UI (maps to the 3 PZ agents + 2 advisory)
        "agents": {
            "Researcher":          {"message": "Scanning the tape.",         "confidence": 0.0, "status": "idle"},
            "Fundamental Analyst": {"message": "Watching macro tone.",        "confidence": 0.0, "status": "idle"},
            "Risk Manager":        {"message": "Limits standing by.",         "confidence": 0.0, "status": "idle"},
            "Trader":              {"message": "Desk is flat.",               "confidence": 0.0, "status": "idle"},
            "Portfolio Manager":   {"message": "Waiting for conviction.",     "confidence": 0.0, "status": "idle"},
        },
        "portfolio": {"value": 100000.0, "cash": 100000.0, "positions": {}},
        "metrics":   {"reward": 0.0, "grade": 0.0, "drawdown": 0.0, "sharpe": 0.0},
        "chart":     {"price": 50000.0, "trade": None, "price_change": 0.0},
        "trade": {
            "pulse": 0, "side": "HOLD", "size": 0.0, "price": 50000.0,
            "sl": 0.0, "tp": 0.0, "portfolio_delta": 0.0, "notional": 0.0,
            "reason": "Waiting for the first coordinated decision.",
            "override": False,
        },
        "flow":   [],
        "engine": {
            "name":          "Multi-Agent Governance (PettingZoo AEC)",
            "mode":          "Rule Fallback",
            "policy_active": False,
            "note":          "Three independent RL agents negotiating via AEC turns: RiskManager → PortfolioManager → Trader.",
        },
        "negotiation": {          # Exposes per-agent negotiation each cycle
            "rm_size_limit":   0.5,
            "rm_allow_new":    True,
            "rm_force_reduce": False,
            "pm_cap_alloc":    0.5,
            "pm_override":     0.0,
            "governance_log":  [],
        },
    }


sim_state = make_initial_state()


class SimulationRunner:
    """
    Orchestrates the PettingZoo AEC loop.

    Each call to step() runs one full AEC cycle:
        RiskManager → PortfolioManager → Trader  →  market advance

    Advisory agents (Researcher, FA) provide contextual signals
    for the UI but do NOT affect the AEC action pipeline.
    """

    def __init__(self):
        self.config = TrainingConfig(tickers=["AAPL"], fast_mode=True, max_steps=100)

        # ── PettingZoo multi-agent environment ──────────────────────────────
        self.env = MultiAgentTradingEnv(
            df=None,
            initial_cash=self.config.initial_cash,
            ticker=self.config.tickers[0],
            commission=self.config.commission,
            max_steps=self.config.max_steps,
        )

        # ── Rule-based AEC policies ─────────────────────────────────────────
        self.policies = {
            RISK_MANAGER:  RuleRiskManagerPolicy(),
            PORTFOLIO_MGR: RulePortfolioManagerPolicy(),
            TRADER:        RuleTraderPolicy(),
        }

        # ── Advisory agents (UI flavor only) ────────────────────────────────
        self.researcher = QuantResearcher()
        self.fa_agent   = FundamentalAnalyst(fast_mode=self.config.fast_mode)

        # ── OpenEnv PZ env (separate instance for judge endpoints) ─────────
        self._openenv_env = MultiAgentTradingEnv(
            df=None,
            initial_cash=self.config.initial_cash,
            ticker=self.config.tickers[0],
            commission=self.config.commission,
            max_steps=self.config.max_steps,
        )
        self._openenv_policies = {
            RISK_MANAGER:  RuleRiskManagerPolicy(),
            PORTFOLIO_MGR: RulePortfolioManagerPolicy(),
        }
        self._openenv_env.reset()

        # ── Initialize demo PZ env ──────────────────────────────────────────
        self.env.reset()
        self.done = False

        sim_state["engine"] = {
            "name":          "Multi-Agent Governance (PettingZoo AEC)",
            "mode":          "Rule Fallback",
            "policy_active": False,
            "note":          "Three independent RL agents negotiating via AEC turns: RiskManager → PortfolioManager → Trader.",
        }

    def step(self):
        """Run one full AEC cycle (RM → PM → Trader → market advance)."""
        if self.done:
            self.env.reset()
            self.fa_agent.reset()
            self.done = False

        global sim_state

        previous_value = sim_state["portfolio"]["value"]
        previous_price = sim_state["chart"]["price"]

        # ── Get a base observation for advisory agents ──────────────────────
        base_obs = self.env.observe(RISK_MANAGER)

        # ── Advisory: Researcher ────────────────────────────────────────────
        r_sig, r_conf, r_reasoning = self.researcher(base_obs)
        researcher_message = f"{r_sig.title()} bias. {r_reasoning}"
        sim_state["agents"]["Researcher"] = {
            "message": researcher_message,
            "confidence": r_conf,
            "status": "active",
        }

        # ── Advisory: Fundamental Analyst ───────────────────────────────────
        fa_sent, fa_reasoning = self.fa_agent(base_obs)
        sim_state["agents"]["Fundamental Analyst"] = {
            "message": fa_reasoning,
            "confidence": abs((fa_sent * 2.0) - 1.0),
            "status": "active",
        }

        # ── AEC Cycle: Step through all 3 agents ───────────────────────────
        rm_action = None
        pm_action = None
        trader_action = None
        cycle_rewards = {}

        for agent in [RISK_MANAGER, PORTFOLIO_MGR, TRADER]:
            if not self.env.agents:
                self.done = True
                break

            obs = self.env.observe(agent)
            action = self.policies[agent].act(obs)

            if agent == RISK_MANAGER:
                rm_action = action
            elif agent == PORTFOLIO_MGR:
                pm_action = action
            elif agent == TRADER:
                trader_action = action

            self.env.step(action)
            cycle_rewards[agent] = self.env.rewards.get(agent, 0.0)

        # ── Check termination ───────────────────────────────────────────────
        if not self.env.agents or all(self.env.terminations.get(ag, False) for ag in ALL_AGENTS):
            self.done = True

        # ── Extract state from the env ──────────────────────────────────────
        env_state = self.env.state()
        trader_info = self.env.infos.get(TRADER, {})

        current_price = env_state["price"]
        portfolio_value = env_state["portfolio_value"]
        portfolio_delta = portfolio_value - previous_value
        price_change = current_price - previous_price

        # ── Parse negotiation messages ──────────────────────────────────────
        rm_msg = env_state.get("rm_message", [0.5, 1.0, 0.0])
        pm_msg = env_state.get("pm_message", [0.5, 0.0])

        rm_size_limit  = float(rm_msg[0]) if len(rm_msg) > 0 else 0.5
        rm_allow_new   = bool(rm_msg[1] > 0.5) if len(rm_msg) > 1 else True
        rm_force_reduce = bool(rm_msg[2] > 0.5) if len(rm_msg) > 2 else False
        pm_cap_alloc   = float(pm_msg[0]) if len(pm_msg) > 0 else 0.5
        pm_override_s  = float(pm_msg[1]) if len(pm_msg) > 1 else 0.0

        # ── Update UI state: Risk Manager ───────────────────────────────────
        rm_reasoning = f"Limit {rm_size_limit:.2f}"
        if rm_force_reduce:
            rm_reasoning += " | FORCE REDUCE active"
        if not rm_allow_new:
            rm_reasoning += " | New positions BLOCKED"
        sim_state["agents"]["Risk Manager"] = {
            "message": rm_reasoning,
            "confidence": 1.0 - rm_size_limit,
            "status": "active" if rm_size_limit < 0.4 or rm_force_reduce else "idle",
        }

        # ── Update UI state: Portfolio Manager ──────────────────────────────
        pm_message = f"Capital allocation: {pm_cap_alloc:.0%}"
        if pm_override_s > 0.7:
            pm_message += " | VETO signal active"
        sim_state["agents"]["Portfolio Manager"] = {
            "message": pm_message,
            "confidence": pm_cap_alloc,
            "status": "active" if pm_override_s > 0.5 or pm_cap_alloc < 0.3 else "idle",
        }

        # ── Update UI state: Trader ─────────────────────────────────────────
        gov = trader_info.get("governance", {})
        executed = gov.get("executed", {}) if gov else {}
        direction = executed.get("direction", 0) if executed else 0
        size      = executed.get("size", 0.0) if executed else 0.0
        sl        = executed.get("sl", 0.0) if executed else 0.0
        tp        = executed.get("tp", 0.0) if executed else 0.0
        interventions = gov.get("interventions", []) if gov else []
        was_compliant = gov.get("was_compliant", True) if gov else True

        dir_str = ["HOLD", "BUY", "SELL"][direction]
        trader_reasoning = f"{dir_str} {size:.2f}"
        if not was_compliant:
            intervention_types = [i.get("type", "?") for i in interventions]
            trader_reasoning += f" (overridden: {', '.join(intervention_types)})"
        else:
            trader_reasoning += " (compliant — no governance intervention)"

        sim_state["agents"]["Trader"] = {
            "message": trader_reasoning,
            "confidence": size,
            "status": "active" if direction != 0 else "idle",
        }

        # ── Sim state update ────────────────────────────────────────────────
        sim_state["current_step"] = env_state["step"]

        sim_state["portfolio"] = {
            "value":     portfolio_value,
            "cash":      env_state["cash"],
            "positions": env_state["positions"],
        }
        sim_state["metrics"] = {
            "reward":   float(cycle_rewards.get(TRADER, 0.0)),
            "grade":    trader_info.get("grade", 0.0),
            "drawdown": env_state["max_drawdown"],
            "sharpe":   env_state["sharpe_ratio"],
        }
        sim_state["chart"] = {
            "price":        current_price,
            "trade":        dir_str if direction != 0 else None,
            "price_change": price_change,
        }
        sim_state["trade"] = {
            "pulse":           sim_state["trade"]["pulse"] + 1,
            "side":            dir_str,
            "size":            float(size),
            "price":           float(current_price),
            "sl":              float(sl),
            "tp":              float(tp),
            "portfolio_delta": float(portfolio_delta),
            "notional":        float(portfolio_value * size if direction != 0 else 0.0),
            "reason":          trader_reasoning,
            "override":        not was_compliant,
        }

        # ── Flow graph for UI ───────────────────────────────────────────────
        sim_state["flow"] = [
            {"from": "Researcher",          "to": "Risk Manager",      "strength": float(r_conf),          "active": True,               "tone": "signal"},
            {"from": "Researcher",          "to": "Portfolio Manager", "strength": float(r_conf),          "active": r_sig != "neutral", "tone": "research"},
            {"from": "Fundamental Analyst", "to": "Portfolio Manager", "strength": float(abs((fa_sent * 2.0) - 1.0)), "active": True,   "tone": "macro"},
            {"from": "Risk Manager",        "to": "Trader",            "strength": float(1.0 - rm_size_limit),        "active": True,   "tone": "risk"},
            {"from": "Portfolio Manager",   "to": "Trader",            "strength": float(pm_cap_alloc),               "active": True,   "tone": "approval"},
            {"from": "Trader",              "to": "Market",            "strength": float(size),                       "active": direction != 0, "tone": dir_str.lower()},
        ]

        # ── Negotiation state (multi-agent-specific) ───────────────────────
        sim_state["negotiation"] = {
            "rm_size_limit":   rm_size_limit,
            "rm_allow_new":    rm_allow_new,
            "rm_force_reduce": rm_force_reduce,
            "pm_cap_alloc":    pm_cap_alloc,
            "pm_override":     pm_override_s,
            "governance_log":  env_state.get("governance_log", []),
        }


runner = None


async def simulation_loop():
    global sim_state, runner
    if runner is None:
        runner = SimulationRunner()

    while sim_state["is_running"]:
        runner.step()
        await asyncio.sleep(0.4)


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
# These use the PettingZoo MultiAgentTradingEnv directly.
# RM and PM run rule-based policies; the Trader action comes from the external caller.

@app.post("/openenv/reset")
@app.post("/reset")
async def openenv_reset():
    """Standard OpenEnv reset — resets the multi-agent PZ env.
    Returns the Trader's initial observation."""
    global runner
    if runner is None:
        runner = SimulationRunner()
    runner._openenv_env.reset()
    trader_obs = runner._openenv_env.observe(TRADER)
    return {"observation": trader_obs.tolist(), "info": {}}


@app.post("/openenv/step")
@app.post("/step")
async def openenv_step(action: dict):
    """Standard OpenEnv step — runs a full AEC cycle.
    RM and PM use rule-based policies. The submitted action is used for the Trader.
    Returns trader's obs/reward/terminated/truncated/info."""
    global runner
    if runner is None:
        runner = SimulationRunner()

    env = runner._openenv_env
    policies = runner._openenv_policies

    # If the episode is over, auto-reset
    if not env.agents:
        env.reset()

    # Run full AEC cycle: RM → PM → Trader
    for agent in [RISK_MANAGER, PORTFOLIO_MGR, TRADER]:
        if not env.agents:
            break
        if agent == TRADER:
            # Use the externally-provided trader action
            trader_action = {
                "direction": int(action.get("direction", 0)),
                "size": np.array([float(action.get("size", 0.0))], dtype=np.float32),
                "sl": np.array([float(action.get("sl", 0.0))], dtype=np.float32),
                "tp": np.array([float(action.get("tp", 0.0))], dtype=np.float32),
            }
            env.step(trader_action)
        else:
            obs = env.observe(agent)
            agent_action = policies[agent].act(obs)
            env.step(agent_action)

    # Collect results from the Trader's perspective
    trader_obs = env.observe(TRADER)
    trader_reward = float(env.rewards.get(TRADER, 0.0))
    terminated = bool(env.terminations.get(TRADER, False))
    truncated = bool(env.truncations.get(TRADER, False))
    trader_info = env.infos.get(TRADER, {})

    return {
        "observation": trader_obs.tolist(),
        "reward": trader_reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": trader_info,
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
