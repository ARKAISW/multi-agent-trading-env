# OpenEnv Hackathon: Build at the Bleeding Edge of AI

**Event:** India's Biggest Mega AI Hackathon
**Built on:** Meta's OpenEnv (the foundation for next-gen RL environments used by leading AI labs)
**Sponsored by:** Hugging Face, PyTorch

**Grand Prize:** Winners get an interview opportunity at Meta & Hugging Face AI teams

**Important Dates:**
- Round 1 Begins: March 25th
- Grand Finale (48-hour sprint in Bangalore): April 25th - 26th


## Results Announcement

- Top 100 Finalists Announced: Friday, May 1st
- Winners Livestream: Friday, May 8th


## Credits & Resources

Get your credits for Cursor AI and Hugging Face as early as possible.

**Cursor AI Credit:** Each participant is eligible. Visit the Scaler Hackathon dashboard to avail credits:
https://tinyurl.com/sclr-openenv-dashboard

**Hugging Face Credits:** $30 credit per person. Avail credits at:
https://huggingface.co/coupons/claim/hf-openenv-community

The same links will be shared in the on-campus Discord channels.


## Meet Your Mentors

**Onsite / Available:**
- Sanyam Bhutani - Partner Engineer, META
- Yash Khare - Partner Engineer, META
- Nilesh Pandey - Partner Engineer, META
- Adithya S Kolavi - Engineer, Hugging Face
- Adarsh Shirawalmath - ML Engineer, Hugging Face
- Arkadip Maitra - ML Engineer, Red Hat
- Aashay Sachdeva - Founding Team, Sarvam
- Deepa Dhevannan - Gen AI Solution Architect
- Soumik Rakshit - ML Engineer, Zomato
- Ayush Satyam - ML Engineer, Red Hat
- Parshant Sharma - ML Engineer, Red Hat

**Remotely Available:**
- Ben Burtenshaw - Community Education AI, Hugging Face
- Alireza Shamsoshoara - PyTorch, Meta


## Discord Guidelines

Important: Since global tech leaders and executives are present, a high level of professionalism and decorum must be maintained. Failure to follow the guidelines will lead to strict action and may impact your participation in the hackathon.


## Technical Session Agenda

- PyTorch Foundation Introduction
- Hackathon Themes
- Submission and Judging Rules
- RL 101 + OpenEnv Recap
- Best Practices
- Q&A


## About PyTorch Foundation

**Mission:** Democratizing and accelerating the adoption of accessible, high-impact AI technologies by cultivating a robust ecosystem of open-source, vendor-neutral projects spanning the entire AI lifecycle.

**Hosted Projects:** Multiple open-source projects under the foundation


## Hackathon Goals

- Learn reinforcement learning (RL)
- Now is a great time to learn RL
- Hack and create cool environments you can use to add skills to models
- Showcase your work on the Hugging Face Hub
- Have fun

**GitHub Repository:** https://github.com/meta-pytorch/OpenEnv


## Guidelines for Problem Statement

- It is NOT mandatory to choose the same problem statement as Round 1. Only choose it if it aligns with the provided hackathon themes.
- Before the onsite event (April 25-26): Work on building the environment, agent behaviors, and reward model.
- Onsite (April 25-26): Post-training will be done when you receive compute credits for Hugging Face.


## What Judges Look For (TL;DR)

Build an environment that an LLM could actually be trained on to get measurably better at something interesting. Then show that training. Then tell the story.

A messy but ambitious environment with real training evidence beats a polished but boring one. Pick a problem that excites you (that energy comes through in the pitch).

**Note:** Only one submission per team. The URL link of your environment must be submitted as judges will pull the environment from the URL to evaluate it. Changes after the deadline will not be considered.


## Judging Criteria

| Criterion | Weight | What It Means |
|-----------|--------|----------------|
| Environment Innovation | 40% | Is the environment novel, creative, or genuinely challenging? Does it meaningfully test agent behavior in a new way? |
| Showing Improvement in Rewards | 20% | Is there observable evidence of training progress? Reward curves, before/after behavior, baseline comparison. |
| Storytelling & Presentation | 30% | Can you clearly explain the problem, the environment, and what the agent learned? Is the demo engaging for a non-technical audience? |
| Reward & Training Pipeline | 10% | Is the reward logic coherent? Does the pipeline produce meaningful improvement? |


## Minimum Submission Requirements (Non-Negotiable)

Submissions missing any of these are at a serious disadvantage:

1. Use OpenEnv (latest release). Build on top of the framework; don't reinvent the wheel.

2. A working training script using Unsloth or Hugging Face TRL, ideally as a Colab notebook so judges can re-run it.

3. Evidence that you actually trained: at minimum, loss and reward plots from a real run.

4. A short writeup: a mini-blog on Hugging Face, a less than 2 minute video on YouTube explaining what your environment does and what you trained, or a short slide deck. All materials must be linked from your README.

5. Push your environment to a Hugging Face Space so it's discoverable and runnable.

6. A README that motivates the problem, explains how the environment works, and shows results.

7. README must have a link to the environment in the Hugging Face Space and all additional references to other materials (videos, blog posts, slides, presentations, etc.).

8. Do not include large video files in your HF Hub submission. Use URL references instead.


## What Makes a Submission Stand Out

### 1. Pick an Ambitious, Original Problem

Ask yourself:
- Does this environment teach an LLM something it currently can't do well?
- Is the domain underexplored in RL/LLM training?
- Could a researcher write a paper about training on this?

Avoid clones of chess, snake, tic-tac-toe, and grid-world.

### 2. Design a Reward Signal That Actually Teaches

A great environment has a reward function that:
- Provides a rich, informative signal (not just 0/1 at the end)
- Captures something hard to measure in a clever way
- Uses OpenEnv's Rubric system thoughtfully (composable rubrics are better than monolithic scoring)
- Is hard to game (an agent that exploits the reward without solving the task should not get high scores)

### 3. Show Real Training, End to End

The bar is not "training script exists." The bar is "training script runs against the environment, the agent learns, and you can show it."

- Your training loop must connect to your environment (not a static dataset)
- Train long enough that the curves mean something
- Compare a trained agent vs. a random/untrained baseline (quantitative and/or qualitative)
- Include plots and numbers in your README and writeup

### 4. Make Your Plots Readable

Reviewers spend seconds, not minutes, on each plot.

- Label both axes ("training step" or "episode" on x, "reward" or "loss" on y) and include units
- Save plots as .png or .jpg and commit them to the repo (don't leave them only in a Colab cell or a deleted Wandb run)
- If you used Wandb, include the link to that specific run
- Embed key plots in your README with a one-line caption explaining what each one shows
- If you have multiple runs (baseline vs. trained, ablations), put them on the same axes so comparison is obvious

### 5. Tell a Story, Not an API Doc

Your README, blog, and pitch should answer:

1. **Problem:** What capability gap or interesting domain are you targeting?
2. **Environment:** What does the agent see, do, and get rewarded for?
3. **Results:** What changed after training? Show it.
4. **Why does it matter:** Who would care, and why?

A reviewer should be able to read your README in 3-5 minutes and want to try your environment.

### 6. Engineer It Cleanly (Table Stakes)

Engineering quality matters less than ambition, but sloppy work hurts.

- Use OpenEnv's Environment or MCPEnvironment base classes properly
- Respect client/server separation (clients should never import server internals)
- Follow the standard Gym-style API (reset, step, state)
- Have a valid openenv.yaml manifest
- Don't use reserved tool names (reset, step, state, close) for MCP tools


## OpenEnv Technical Recap

### The RL Loop (Conceptual Example: Teaching a Dog to Sit)

```
observation = environment.reset()  # Start a new episode
while not done:
    observation = environment.observe()  # What does the agent see?
    action = agent.choose(observation)   # What does the agent do?
    result = environment.step(action)    # Environment responds
    reward = result.reward               # Get feedback
    agent.learn(reward)                  # Agent learns
```

### The Four Key Concepts

- **reset()** - Start a new episode. Begin a fresh training session.
- **observation** - What the agent sees. The current state of the world.
- **action** - What the agent does. Sit, spin, move left, etc.
- **step(action)** - Execute the action. Returns three things: new observation, reward, and done flag (episode over).

### Building Your Environment in 5 Simple Steps

1. **Define Types (models.py)** - Action, Observation, State dataclasses
2. **Implement Environment (server/environment.py)** - reset(), step(), state() methods
3. **Create Client (client.py)** - HTTPEnvClient subclass
4. **Create Server (server/app.py)** - app = create_fastapi_app(env)
5. **Dockerize (Dockerfile)** - Standard container setup

**Or use the CLI:** `openenv init my_env` - scaffolding ready in seconds.

### The Universal Interface

Every OpenEnv environment implements these 3 methods:

```python
class Environment:
    def reset(self) -> Observation:
        """Start a new episode"""
    
    def step(self, action: Action) -> Observation:
        """Execute action, return observation"""
    
    def state(self) -> State:
        """Get episode metadata"""
```

### Type-Safe by Design

Define your data structures with Python dataclasses:

- **Action:** What the agent does (move, jump, click, type, etc.)
- **Observation:** What the agent sees (board state, pixels, text, etc.)
- **State:** Episode metadata (ID, step count, timestamp, etc.)

### Connecting to Any Environment

This pattern works for Chess, Atari, Trading, Android - everything:

```python
# Connect to environment (runs in Docker container)
env = SomeEnv.from_docker_image("some-env:latest")

# Start new episode
result = env.reset()

# Take action
action = SomeAction(...)
result = env.step(action)

# Get episode metadata
state = env.state()

# Clean up
env.close()  # Container stops automatically
```


## Model Context Protocol (MCP) - Adding Tools to Your Environment

**The Challenge:** Modern AI agents need access to external systems like web search APIs, file operations, database queries, Git operations, and custom integrations.

**The Solution:** MCP (Model Context Protocol) - a standard protocol for AI agents to discover and call tools. It features a REST-like API (JSON-RPC), works with any AI framework, and has plug-and-play tool servers.


## Deployment Commands

```bash
# Initialize a new environment
openenv init my_env
cd my_env

# Deploy to your namespace
openenv push

# Deploy to specific repo
openenv push --repo-id username/my-env

# Deploy as private
openenv push --repo-id username/my-env --private
```


## Hugging Face Spaces - Three Components

Every HF Space provides three components:

### 1. Server: A Running Environment Endpoint

Connect directly to the running Space (WebSocket under the hood).

**Async (recommended):**
```python
async with EchoEnv(base_url="https://openenv-echo-env.hf.space") as client:
    result = await client.reset()
    result = await client.step(EchoAction(message="Hello"))
```

**Sync (using .sync() wrapper):**
```python
with EchoEnv(base_url="https://openenv-echo-env.hf.space").sync() as client:
    result = client.reset()
    result = client.step(EchoAction(message="Hello"))
```

**Available Endpoints:**
- /ws - WebSocket persistent session (used by client)
- /health - HTTP GET health check
- /reset - HTTP POST reset environment (stateless)
- /step - HTTP POST execute action (stateless)
- /state - HTTP GET current state
- /docs - HTTP GET OpenAPI documentation
- /web - HTTP GET interactive web UI

**Check if space is running:**
```bash
curl https://openenv-echo-env.hf.space/health
# Returns: {"status":"healthy"}
```

### 2. Repository: Installable Python Package

Every Space is a Git repository. OpenEnv environments include a pyproject.toml, making them pip-installable directly from the Space URL.

```bash
# Install client package from Space
pip install git+https://huggingface.co/spaces/openenv/echo-env
```

This installs: Client class (EchoEnv), Models (EchoAction, EchoObservation), and Utilities.

After installation:
```python
from envs.echo_env import EchoEnv, EchoAction, EchoObservation
action = EchoAction(message="Hello")
```

### 3. Registry: Docker Container Image

```bash
# Pull the image
docker pull registry.hf.space/openenv-echo-env:latest

# Run locally on port 8001
docker run -d -p 8001:8000 registry.hf.space/openenv-echo-env:latest
```


## Client Usage Examples

```python
import asyncio
from echo_env import EchoEnv, EchoAction

async def main():
    # Development: connect to remote Space
    async with EchoEnv(base_url="https://openenv-echo-env.hf.space") as client:
        result = await client.reset()

    # Production: run locally for speed
    # docker run -d -p 8001:8000 registry.hf.space/openenv-echo-env:latest
    async with EchoEnv(base_url="http://localhost:8001") as client:
        result = await client.reset()

    # Or let the client manage Docker for you
    client = await EchoEnv.from_env("openenv/echo-env")  # Auto-pulls and runs
    async with client:
        result = await client.reset()

asyncio.run(main())

# For sync usage, use the .sync() wrapper:
with EchoEnv(base_url="http://localhost:8001").sync() as client:
    result = client.reset()
```


## Clone and Run Environment Locally

```bash
# Clone from HF Space
git clone https://huggingface.co/spaces/burtenshaw/openenv-benchmark
cd openenv-benchmark

# Install in editable mode
uv sync

# Start server
uv run server

# Run isolated from remote Space
uv run --isolated --project https://huggingface.co/spaces/burtenshaw/openenv-benchmark server
```


## Local Development with Uvicorn

```bash
# Full control over uvicorn options
uvicorn benchmark.server.app:app --host "$HOST" --port "$PORT" --workers "$WORKERS"

# With reload for development
uvicorn benchmark.server.app:app --host 0.0.0.0 --port 8000 --reload

# Multi-worker mode for better concurrency
uvicorn benchmark.server.app:app --host 0.0.0.0 --port 8000 --workers 4
```


## Run Container Locally from Space

```bash
# Clone from HF Space
git clone https://huggingface.co/spaces/burtenshaw/openenv-benchmark
cd openenv-benchmark

# Using OpenEnv CLI (recommended)
openenv build -t openenv-benchmark:latest

# Or with Docker directly
docker build -t openenv-benchmark:latest -f server/Dockerfile .
```


## Environment Setup

### Using uv venv:
```bash
uv venv
source .venv/bin/activate
uv pip install openenv-core
```

### Using conda:
```bash
conda create -n openenv_hackathon python=3.12
conda activate openenv_hackathon
uv pip install openenv-core
```

### Initialize a New Environment:
```bash
openenv init HackEnv101_AlirezaShamsoshoara
```

This creates 11 files and generates uv.lock. Next steps:
```bash
cd /path/to/HackEnv101_AlirezaShamsoshoara
# Edit environment implementation in server/..._environment.py
# Edit models in models.py
# Install dependencies: uv sync
```


## Training Resources

### Training with TRL (GRPO)

Hugging Face TRL integrates natively with OpenEnv environments for GRPO training.

**Resources:**
- TRL OpenEnv Documentation: https://huggingface.co/docs/trl/en/openenv
- Sudoku Example: https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_sudoku_qrpo.ipynb
- Wordle Example: https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_worldle_qrpo.ipynb
- More TRL Examples: https://github.com/huggingface/trl/tree/main/examples/scripts/openenv

**General Training Examples:**
- Main examples directory: https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial/examples
- Unsloth 2048 example: https://github.com/meta-pytorch/OpenEnv/blob/main/tutorial/examples/unsloht_2048.ipynb
- Wordle example (TRL): https://github.com/meta-pytorch/OpenEnv/blob/main/tutorial/examples/worldle.py

### Training with Unsloth

Unsloth provides 2x faster training and 70% less memory through custom CUDA kernels. Works as a drop-in replacement - same TRL API, just faster.

**The Pattern:**
1. Load model via FastLanguageModel (with 4-bit quantization)
2. Apply LoRA adapters for parameter-efficient training
3. Use OpenEnv as the reward function
4. Train with standard GRPOTrainer

**Google Colab Ready:** Run on a free T4 GPU. Unsloth + OpenEnv Colab notebook available for the 2048 game environment with 20B parameter models.

**Also Compatible With:** TRL, torchforge, SkyRL, ART, Oumi, veRL


## Accessing Hugging Face Infrastructure

Use HF infrastructure to run your training. Hugging Face Jobs provide compute for AI and data workflows.

**Important Notes:**
- Depends on your model size, choose your GPU model wisely
- Choose wisely so you can run training/inference for a reasonable time with your credits
- A T4 GPU (small/medium) is a good choice

**Methods to Run Jobs:**
- hf CLI
- huggingface_hub Python client
- Jobs HTTP API

**Pricing and Billing Resources:**
- Billing settings: https://huggingface.co/settings/billing
- Jobs settings: https://huggingface.co/settings/jobs
- Jobs documentation: https://huggingface.co/docs/hub/jobs
- Job CLI documentation: https://huggingface.co/docs/huggingface_hub/guides/cli#hf-jobs
- Jobs guide: https://huggingface.co/docs/huggingface_hub/guides/jobs
- Jobs pricing: https://huggingface.co/docs/hub/jobs-pricing
- Jobs examples: https://huggingface.co/docs/hub/jobs-examples

**Check available hardware:**
```bash
hf jobs hardware
```

### Example Hardware Options

| Name | Pretty Name | CPU | RAM | Accelerator | Cost/Hour |
|------|-------------|-----|-----|-------------|-----------|
| cpu-basic | CPU Basic | 2 vCPU | 16 GB | N/A | $0.01 |
| cpu-upgrade | CPU Upgrade | 8 vCPU | 32 GB | N/A | $0.03 |
| t4-small | Nvidia T4 - small | 4 vCPU | 15 GB | 1x T4 (16 GB) | $0.40 |
| t4-medium | Nvidia T4 - medium | 8 vCPU | 30 GB | 1x T4 (16 GB) | $0.60 |
| a10g-small | Nvidia A10G - small | 4 vCPU | 15 GB | 1x A10G (24 GB) | $1.00 |
| a10g-large | Nvidia A10G - large | 12 vCPU | 46 GB | 1x A10G (24 GB) | $1.50 |
| a100-large | Nvidia A100 - large | 12 vCPU | 142 GB | 1x A100 (80 GB) | $2.50 |
| a100x4 | 4x Nvidia A100 | 48 vCPU | 568 GB | 4x A100 (320 GB) | $10.00 |
| a100x8 | 8x Nvidia A100 | 96 vCPU | 1136 GB | 8x A100 (640 GB) | $20.00 |
| h200 | Nvidia H200 | 23 vCPU | 256 GB | 1x H200 (141 GB) | $5.00 |
| h200x2 | Nvidia H200 (2x) | 46 vCPU | 512 GB | 2x H200 (282 GB) | $10.00 |
| h200x4 | Nvidia H200 (4x) | 92 vCPU | 1024 GB | 4x H200 (564 GB) | $20.00 |
| h200x8 | Nvidia H200 (8x) | 184 vCPU | 2048 GB | 8x H200 (1128 GB) | $40.00 |


## Still Have Questions?

Please mention them in the Discord India OpenEnv Hackathon channels, and the team will do their best to answer.

---

## Example Model Reference

- model_name_or_path: Qwen/Qwen2-0.5B (and similar models)
```

This is plain markdown text. Just copy everything between the triple backticks and paste it into any markdown editor or document.