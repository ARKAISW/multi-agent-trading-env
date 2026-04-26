---
title: "QuantHive: Teaching AI to Survive Being Wrong"
emoji: "🏛️"
colorFrom: "blue"
colorTo: "indigo"
sdk: "docker"
pinned: false
---

# QuantHive: Teaching AI to Survive Being Wrong

Most people think trading is about predicting the next price movement.

The first lesson I learned from observing a real risk quant was that professional trading isn't primarily about prediction. It's mostly about surviving being wrong.

### The Origin

I’m a Grade 12 student in India, and my older cousin is a risk quant. Early on, I got to see what real institutional finance looks like. It wasn’t about chaotic chart reading or betting on the next big breakout. It was a strict, highly disciplined system of constraints and balances. I learned early that real trading is not prediction; it's about controlled risk.

When I started experimenting with AI and Reinforcement Learning, I became fascinated by disciplined decision systems. Most AI trading environments in the open-source world are simple single-agent setups. They provide a model with price history and reward it solely for maximizing profit and loss.

But that's not how a hedge fund operates. If a human trader goes rogue, the risk desk intervenes forcefully. I wondered how AI would handle that if it were trained properly.

### The Insight

That changed my perspective. The intriguing question was not whether AI could predict the next price movement.

**It was whether AI could learn institutional discipline.**

Could we train an AI not only to pursue profits but also to negotiate, comply, and adjust to shifting oversight? Could we create a system where governance isn’t a rigid rule but a conversation?

### Entering the QuantHive

To address this, I built **QuantHive**—a governance-first trading environment that incorporates a multi-agent setup centered around PettingZoo’s AEC model. Instead of one reckless AI, I divided institutional trading into three opposing roles:

1. **The Trader:** Aims to maximize profit and find alpha.
2. **The Portfolio Manager:** Controls capital allocation and seeks steady growth without significant drawdowns.
3. **The Risk Manager:** Has the authority to limit position sizes and reduce exposure forcefully if risks arise.

They interact through structured message passing and governance limits within the environment loop. The environment rewards survival, not recklessness. The Risk Manager is rewarded for limiting trades during risky drawdowns, while the Trader must figure out how to make money within the changing limits set by the others.

### From Floats to Thoughts: Semantic Reasoning

The most valuable change came when training the Qwen 2.5 1.5B model with GRPO (Group Relative Policy Optimization).

At first, the agents received raw float arrays (e.g., `0.284`). But to truly achieve "Auditable AI," I shifted the environment to use **Semantic Reasoning**. Instead of a vector of 24 numbers, the AI "reads" the market state in human terms: *"RSI is 28.4 (oversold).”*

This simple change made the most of the LLM's pre-trained world knowledge. I trained the model against five reward verifiers, enforcing not only profit but also *Format, Alignment, Risk, and Governance.*

### The Smoking Gun

After 250 steps of GRPO training, the most interesting result was how the Trader adapted. The Trader began anticipating interventions and made adjustments before being forced to.

Governance compliance rose from a random 7% to **88%**, and Risk Limit Adherence reached **93%** across held-out evaluation episodes in the governed environment.

But the best part is how it complies. Because I required the model to explain its actions in natural language, the trained agent now outputs statements like:

> *"...I also see that the portfolio's allocation of capital is nearing its limit (0.5). Given the Risk Manager's constraint on the size limit, I need to be cautious..."*

It doesn’t just follow the rules; it understands and explicitly references them before taking action.

### The Broader Implication

Finance serves as a high-pressure test case. The larger question is whether autonomous systems can learn to operate under institutional oversight, justify their actions, and adapt to governance without hurting performance.

I set out to determine if AI could be taught institutional discipline. The surprising outcome was not that the model became more profitable first. It became more disciplined first.

---
*Check out the full project and see the live multi-agent choreography on our Hugging Face Space! All links and documents are available in the Space's [README.md](https://huggingface.co/spaces/ARKAISW/QuantHive/blob/main/README.md).*
