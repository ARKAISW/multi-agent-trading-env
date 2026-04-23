---
title: "MATE: Multi-Agent Trading Environment"
emoji: "🌌"
colorFrom: "blue"
colorTo: "indigo"
sdk: "docker"
pinned: false
app_port: 7860
---

# MATE: Multi-Agent Trading Environment

[![OpenEnv](https://img.shields.io/badge/Environment-OpenEnv-blue.svg)](https://github.com/OpenEnv)
[![Hackathon](https://img.shields.io/badge/Hackathon-OpenEnv%20April%20%2726-orange.svg)](https://hackathon.openenv.org)

MATE is a multi-agent trading environment built for the OpenEnv April '26 Hackathon. It is optimized for Theme #1, multi-agent interactions: researcher, analyst, risk, trader, and portfolio manager coordinate inside a verifiable market environment with normalized rewards and explicit safety checks.

## What Judges Can Test

The Hugging Face Space is expected to expose OpenEnv-compatible endpoints:

- `POST /openenv/reset`
- `POST /openenv/step`

Reset returns:

```json
{
  "observation": [0.0],
  "info": {
    "grade": 0.0
  }
}
```

Step accepts:

```json
{
  "direction": 1,
  "size": 0.25,
  "sl": 0.0,
  "tp": 0.0
}
```

Step returns:

```json
{
  "observation": [0.0],
  "reward": 0.5,
  "terminated": false,
  "truncated": false,
  "info": {
    "grade": 0.62,
    "trade_count": 1
  }
}
```

## Local Checks

Install deps:

```bash
pip install -r requirements-space.txt
```

Run the smoke test:

```bash
python -u tests/smoke_test.py
```

Run the UI backend:

```bash
python app.py --demo
```

Run a quick environment/training pass:

```bash
python app.py --gbm --fast
```

## Hugging Face Deployment

This repo is set up for a Docker Space.

- The container listens on port `7860`.
- The default runtime is secret-free.
- `USE_LOCAL_POLICY=false` by default.
- `ENABLE_REMOTE_PM=false` by default.
- `ENABLE_REMOTE_JUDGE=false` by default.

That means the Space should boot and respond even without mounted model artifacts or API keys.

After deployment, test the judge-facing contract directly:

```bash
curl -X POST https://<your-space>.hf.space/openenv/reset
```

```bash
curl -X POST https://<your-space>.hf.space/openenv/step \
  -H "Content-Type: application/json" \
  -d "{\"direction\":1,\"size\":0.25,\"sl\":0.0,\"tp\":0.0}"
```

## Training Path

For submission, keep environment deployment and model training separate.

- CPU warm start from trajectories:

```bash
python training/train_cpu.py
```

- GPU-only GRPO refinement:

```bash
python training/train_grpo.py --regime hard
```

`training/train_grpo.py` is intended for CUDA-backed machines such as Hugging Face GPU runners. On CPU-only machines it exits early with a clear error.

## Evaluation Notes

The environment uses multiple verifier-style reward components rather than one opaque reward:

- format compliance
- action and signal alignment
- position-limit compliance
- trend/profit consistency

This is the core submission claim: the agent is evaluated not only on reward, but on whether its behavior is structurally valid and risk-aware.

## Files That Matter Most

- [openenv.yaml](/e:/Development/Round2/openenv.yaml)
- [api/server.py](/e:/Development/Round2/api/server.py)
- [env/trading_env.py](/e:/Development/Round2/env/trading_env.py)
- [env/reward.py](/e:/Development/Round2/env/reward.py)
- [training/train_grpo.py](/e:/Development/Round2/training/train_grpo.py)
- [tests/smoke_test.py](/e:/Development/Round2/tests/smoke_test.py)

Built for the OpenEnv April '26 Hackathon.
