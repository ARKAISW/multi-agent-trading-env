

Theme #1 - Multi-Agent Interactions
Environments for this theme involve cooperation, competition, negotiation, and
coalition formation. Learning from these environments will enable agents to model the
beliefs and incentives of others in partially observable settings. This drives
theory-of-mind reasoning and emergent strategic behavior.
Expected Outcome: an environment that can be used to train multi-agent task
handling in a LLM
Example environments: Market simulations, compute-allocation negotiations,
collaborative puzzle worlds, mixed cooperative/competitive strategy games.
Sub-themes with bonus prizes.
- Fleet AI. Scalable Oversight: Environments that train oversight agents to
monitor, analyze, and explain the behavior of other AI agents operating in
complex, multi-agent settings.
- Halluminate. Multi-Actor Environments: Build a realistic environment where an
agent interacts with and manages multiple actors (agents) to discover and
achieve the task
Theme #2 - (Super) Long-Horizon Planning & Instruction
## Following
You will build environments that require deep, multi-step reasoning with sparse or
delayed rewards. After using these environments, the goal is to enable agents to
decompose goals, track state over extended trajectories, and recover from early
mistakes. The aim is to push beyond shallow next-token reasoning toward structured
planning and durable internal representations.
Expected Outcome: an environment that can capture and improve LLM behaviour on
challenging long horizon tasks that need long running sessions beyond context
memory limits.

Example environments: Research-planning simulators, large-scale codebase
refactoring tasks, strategic resource management worlds, long-horizon logistics
optimization, extremely complicated long-horizon instruction following (e.g., 300
instructions scattered around).
Sub-themes with bonus prizes.
- Scale AI. Environments for long horizon workflows for non-code use cases
within a business setting: focusing on either Sales, Project management, or HR
## & IT.
- Mercor. Make an environment with capped/uncapped rewards where frontier
model rewards scale with token output.
## Theme #3 - World Modeling
## #3.1 Professional Tasks
Here you will develop environments that require real interaction with tools, APIs, or dynamic
systems where the model is expected to do real hard work instead of exploiting short-cuts to
arrive at the desired outcome. Learning from these environments will enable agents to
maintain consistent internal state, update beliefs based on outcomes, and orchestrate
multi-step workflows. The goal is to strengthen causal reasoning and persistent world models.
Expected Outcome: an environment capturing nuances of a defined partially observable world
and improve LLM interaction with it
Example environments: Dynamic browser/API ecosystems, enterprise applications, scientific
workflow loops (papers → code → experiments), economic simulations with feedback,
tool-discovery benchmarks.
Sub-themes with bonus prizes.
- Scaler AI Labs. Multi-App RL Environment for Enterprise Workfl ows: Create RL
environments to demonstrate complex workflows, business rule nuances etc in
a large enterprise



## #3.2 Personalized Tasks
Here we will develop an environment that offers real personalized task handling,
imagine replying to personal messages or handling dinner conflicts due to work
conflicts, replying to tough emails. Think any personal assistant tasks

Expected Outcome: An environment that gives the model a realistic simulation of
handling personal tasks, conflicts and managing them as delegations

Example environments: Executive Assistant Meeting Planner, Dinner and drive
planning, email and message replying, shopping, etc

Sub-themes with bonus prizes.
- Patronus AI. Consumer Workflows with Schema Drift: Multi-step consumer
workflow environments where the underlying data schemas, API contracts, and
t&cs/policies/rules change.

Theme #4 - Self-Improvement
The focus here is to create environments where agents can learn to generate new
challenges, escalate difficulty, and improve through self-play or adaptive curricula.
Rather than optimizing fixed tasks, the goal is for agents to learn to drive their own
capability growth. The objective is recursive skill amplific ation.
Expected Outcome: an environment for improving self-play of a LLM over a defined
set of tasks
Example environments: Self-play negotiation arenas, auto-generated math/proof
tasks, evolving coding competitions, adaptive RL curricula.
Sub-themes with bonus prizes.
- Snorkel AI. Simulated Experts-in-the-Loop: Environment that simulates
interactions with real subject-matter experts, with changing requirements /
preferences.


## Theme #5: Wild Card - Impress Us!
We do not want to limit your focus if your idea doesn’t fit the boxes above, we want
and WILL reward out of box tasks, please be creative but remember to add
submissions that meaningfully add value to LLM training on a certain task.

Guidelines for Problem Statement
● It is NOT mandatory to choose the same problem statement as Round 1. Only
choose the same problem statement if it aligns with the above provided
Hackathon themes.
● You can start working on your problem statement once you have finalized it.
Post-training can be done onsite on 25th & 26th when you receive compute
credits for HuggingFace.
● Before the onsite, we suggest you work on building the environment, agent
behaviours, reward model and evaluate if your work aligns with the judging
criteria given below.


## Judging Criteria
Minimum requirements:
● Usage of OpenEnv (latest release)
● Show a minimal training script for your environment using Unsloth or HF TRL in
## Colab
● Write a mini-blog on HuggingFace or mini-video on YouTube talking about your
submission, <2 minutes

## First Round Judging Overview

● Pitch Format: Each team has 3 minutes to pitch, followed by 2 minutes for
Q&A (5 minutes total).
● Evaluation: Teams will be scored based on the following criteria:
- Environment Innovation (40%): Is the environment novel, creative, or
challenging? Does it meaningfully test the agent’s behavior?
- Storytelling (30%): Does the team clearly explain the problem,
environment, and agent behavior? Is the demo engaging and easy to
follow?
- Showing Improvement in Rewards (20%): Does the demo provide
observable evidence of training progress (reward curves, metrics, or
before/after behavior)?
- Reward and Training Script/Pipeline Setup (10%): Is the reward logic
coherent, and does the pipeline produce meaningful improvement in
the agent’s inference (how it acts in the environment)?
Each evaluator will judge about 10-15 teams during the judging process,
submitting scores individually for each team. Once scores are submitted, the
Cerebral Valley team will aggregate your scores with the other judge's scores to
determine the top 15 finalist projects.
