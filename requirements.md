

**What the automated round checks**
These are the items the validation pass looks for. If any is missing or broken at the deadline, the submission won't make it to a human judge; regardless of how strong the underlying idea is. Verify each one explicitly before you submit.

- Public, cloneable Hugging Face Space at the submitted URL. Test from a logged-out browser. Private spaces, dead links, or 404s are an automatic out.
- Valid OpenEnv structure: proper Environment / MCPEnvironment base class, Gym-style reset / step / state, and a parseable openenv.yaml.
- Training evidence committed to the repo as image files (.png / .jpg): At minimum a loss curve and a reward curve. Wandb-only links and plots that live only in a Colab cell don't count: they may not be reachable when validation runs.
- A runnable training script (Unsloth, HF TRL, or other frameworks), preferably linked as a Colab notebook so it can be re-executed end to end (Python script is acceptable as well).
- A README that links every deliverable: HF Space, training notebook, and your writeup (blog / video / slides), with the key plots embedded inline. If validation can't reach a deliverable from the README, it counts as missing.


**TL;DR**

Build an environment that an LLM could actually be trained on to get measurably better at

something interesting. Then show that training. Then tell the story.

A messy but ambitious environment with real training evidence beats a polished but boring one.

Pick a problem that excites you (that energy comes through in the pitch).

**Judging Criteria**

**Criterion: Environment Innovation**Weight: 40%What it means:Is the environment novel, creative, or genuinely challenging?Does it meaningfully test agent behavior **in** a way that hasn't been done before?

**Criterion: Storytelling & Presentation**Weight: 30%What it means:Can you clearly explain the problem, the environment, and what the agent learned?Is the demo engaging and easy to follow **for** a non-technical audience?

**Criterion: Showing Improvement in Rewards**Weight: 20%What it means:Is there observable evidence of training progress? Reward curves, before/after behavior,comparison against a baseline -- anything that proves the agent learned something.

**Criterion: Reward & Training Pipeline**Weight: 10%What it means:Is the reward logic coherent? Does the pipeline produce meaningful improvement **in** the trainedagent's behavior?

**Minimum Submission Requirements**

**NOTE:** These are **non-negotiable**. Submissions missing any of these are at a serious disadvantage.

*   **Use OpenEnv** (latest release). Build on top of the framework; don’t reinvent the wheel.
    
*   **A working training script** using **Unsloth or Hugging Face TRL**, ideally as a Colab notebook so judges can re-run it.
    
*   **Evidence that you actually trained**; at minimum, loss and reward plots from a real run.
    
*   **A short writeup**: a mini-blog on Hugging Face or a < 2 minute video on YouTube explaining what your environment does and what you trained, or a short slide deck of presentation. Please make sure that all materials are linked from your README file so that judges can access them easily.
    
*   **Push your environment to a Hugging Face Space** so it’s discoverable and runnable.
    
*   **A README** that motivates the problem, explains how the env works, and shows results.
    
    *   README should have a link to the environment in the Hugging Face Space. It should also have all additional references to other materials (e.g. videos, blog posts, slides, presentations, etc.) that you want to include.
        
*   Please do not include big video files in your Env submission on HF Hub as we would like to have a small size for each env (Please use url as reference link to additional materials).
    

**What Makes a Submission Stand Out**

_**Pick an ambitious, original problem**_

The themes (problems) are deliberately open. Use them as launching pads, not boxes. Judges have seen a lot of chess, snake, tic-tac-toe, and grid-world clones. To score well on innovation,

you need a genuinely fresh angle. Some questions to ask yourself:

*   Does this environment exist to teach an LLM something it currently can’t do well?
    
*   Is the domain underexplored in RL/LLM training?
    
*   Could a researcher write a paper about training on this?
    

_**Design a reward signal that actually teaches**_

A great environment has a reward function that:

*   Provides a **rich, informative signal** (not just 0/1 at the end)
    
*   Captures something **hard to measure** in a clever way
    
*   Uses OpenEnv’s **Rubric system** thoughtfully (composable rubrics > monolithic scoring)
    
*   Is **hard to game**; an agent that exploits the reward without solving the task should not get high scores
    

_**Show real training, end to end**_

The bar isn’t “training script exists.” The bar is “training script runs against the environment, the

agent learns, and you can show it.” Concretely:

*   Your training loop should connect to **your** environment (not a static dataset)
    
*   Train long enough that the curves mean something
    
*   Compare a **trained agent vs. a random/untrained baseline**; quantitative and/or qualitative
    
*   Include the plots and numbers in your README and writeup
    

_**Make your plots readable**_

Reviewers spend seconds, not minutes, on each plot. Help them out:

*   **Label both axes** (e.g. “training step” / “episode” on x, “reward” / “loss” on y) and include units where they apply
    
*   Save plots as _.png_ or _.jpg_ and **commit them to the repo** (don’t leave them only in a Colab cell or a deleted Wandb run) (if you ran via Wandb, please include the link to that specific run of your plots)
    
*   **Embed the key plots in your README** with a one-line caption explaining what each one shows If you have multiple runs (baseline vs. trained, ablations, etc.), put them on the same axes so the comparison is obvious
    

_**Tell a story, not an API doc**_

Your README, blog, and pitch should answer:

1.  **Problem)** what capability gap or interesting domain are you targeting?
    
2.  **Environment)** what does the agent see, do, and get rewarded for?
    
3.  **Results)** what changed after training? Show it.
    
4.  **Why does it matter)** who would care, and why?
    

_A reviewer should be able to read your README in 3~5 minutes and want to try your_

_environment._

**NOTE:** If you have a video, HF post, or anything else interesting, please make sure that it’s linked

  from your README as a link.

_**Engineer it cleanly (table stakes)**_

Engineering quality matters less than ambition, but sloppy work hurts. Make sure you:

*   Use OpenEnv’s Environment / MCPEnvironment base classes properly
    
*   Respect the **client / server separation** (clients should never import server internals)
    
*   Follow the standard Gym-style API (reset, step, state)
    
*   Have a valid openenv.yaml manifest
    
*   Don’t use reserved tool names (reset, step, state, close) for MCP tools
    

**Final Note**

Judges are looking for environments that push the frontier of what we can train LLMs to do. Be

ambitious. Pick a problem you find genuinely interesting; that almost always produces better

work than chasing what you think judges want. Good luck.
