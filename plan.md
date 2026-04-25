# Goal Description

To elevate QuantHive into a definitive Top 15 Hackathon submission, we need to transition from a "single-agent Gym environment with pipeline functions" to a **"decentralized society of interacting agents"** using true multi-agent RL principles. 

This requires rewriting the core environment to use the **PettingZoo** AEC (Agent Environment Cycle) API, giving each agent independent observation/action spaces, conflicting reward functions (emergent behavior), and communication channels.

## User Review Required

> [!WARNING]
> **Massive Architectural Rewrite**
> This change will rip out the foundation of your current, working project. 
> 
> 1. We will replace [trading_env.py](file:///e:/Development/Round2/env/trading_env.py) (Gym) with `multi_agent_env.py` (PettingZoo).
> 2. The API server ([server.py](file:///e:/Development/Round2/api/server.py)) and UI will break and need to be rewritten to support asynchronous agent steps.
> 3. The current GRPO training script ([train_grpo.py](file:///e:/Development/Round2/training/train_grpo.py)) trains a single policy on JSON. In a true multi-agent setup, we need an *online* RL loop. We will build a multi-agent rollout collector connecting to Unsloth/TRL, but it is experimental and computationally heavy.
> 
> If you are close to the submission deadline, doing this is extremely risky. If you proceed, the repository will be in a broken state until all components are rewired.

## Proposed Changes

### Core Environment (PettingZoo)
Replace the single-agent Gym environment with a multi-agent PettingZoo environment.
#### [NEW] `env/multi_agent_env.py` 
- Inherits from `pettingzoo.utils.env.AECEnv`.
- Agents: `["risk_manager_0", "portfolio_manager_0", "trader_0"]`.
- [step()](file:///e:/Development/Round2/api/server.py#108-227) and `observe()` functions that alternate execution between agents.
- **Agent Negotiation:** The observation space of the Trader includes the output messages/constraints from the Risk Manager and PM.
- **Adversarial Rewards:** 
  - Trader: Rewarded for PnL.
  - Risk Manager: Rewarded for capping size when volatility/drawdown is high, penalized when Trader loses money.
#### [MODIFY] [env/trading_env.py](file:///e:/Development/Round2/env/trading_env.py)
- Deprecate or refactor to wrap the PettingZoo environment for legacy compatibility.

### Agents & Governance
Modify the agent definitions to act as independent RL policies within the PettingZoo loop.
#### [MODIFY] [agents/risk_model.py](file:///e:/Development/Round2/agents/risk_model.py)
#### [MODIFY] [agents/portfolio_manager.py](file:///e:/Development/Round2/agents/portfolio_manager.py)
#### [MODIFY] [agents/trader.py](file:///e:/Development/Round2/agents/trader.py)
- Refactor agents to accept PettingZoo observations (which include multi-agent messages) and output PettingZoo actions.

### Training Loop (Online Multi-Agent RL)
Connect the LLMs to the PettingZoo environment for online rollout collection.
#### [NEW] `training/train_multi_agent.py`
- An online RL loop that steps the `multi_agent_env`.
- Collects trajectories (Observation, Action, Reward) for multiple agents.
- Feeds collected rollout buffers into the GRPO/PPO trainer. Note: Full multi-agent online LLM training is extremely heavy; we may implement it as alternating optimization (freeze RM, train Trader, freeze Trader, train RM).

### API Server and UI
Update the server to orchestrate a PettingZoo AEC loop.
#### [MODIFY] [api/server.py](file:///e:/Development/Round2/api/server.py)
- Rewrite [SimulationRunner](file:///e:/Development/Round2/api/server.py#72-227) to step through the PettingZoo `agent_iter()`.
- Broadcast state updates to the UI, showing the negotiation and adversarial interactions.

## Verification Plan

### Automated Tests
1. Initialize `MultiAgentEnv` and run the `pettingzoo.test.api_test()`.
2. Verify that taking actions with `risk_manager_0` updates the observation space of `trader_0`.
3. Verify that the adversarial reward functions independently return conflicting scores (e.g., RM gets +1 for restricting, Trader gets -1 for missing a trade).

### Manual Verification
1. Run the new API server and step through the UI to see the multi-agent negotiation in real-time.
2. Run `train_multi_agent.py` for 50 steps to ensure trajectories build correctly and gradients update.
