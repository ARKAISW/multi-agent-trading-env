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

The first thing I learned from watching a real risk quant work is that professional trading isn't really about prediction at all. It is mostly about surviving being wrong.

### The Origin

I’m a Grade 12 student in India, and my elder cousin happens to be a risk quant. Early on, I got to see a glimpse of what real institutional finance looks like. It wasn't about chaotic chart reading or gambling on the next big breakout. It was a rigorous, highly disciplined system of constraints and balances. I was exposed early to the idea that real trading is not prediction—it is controlled risk.

When I started experimenting with AI and Reinforcement Learning, I was fascinated by disciplined decision systems. Most AI trading environments in the open-source world are simple single-agent setups: they feed a model price history and reward it solely for maximizing PnL. 

But that's not how a hedge fund works. If a human trader goes rogue, the risk desk forcefully intervenes. I wondered: *How would AI handle that if taught properly?*

### The Insight

That changed the question for me. The interesting problem was not whether AI could predict the next candle. 

**It was whether AI could learn institutional discipline.**

Could we train an AI not just to chase profits, but to negotiate, comply, and adapt to changing oversight? Could we build a system where governance isn't a hardcoded monolithic rule, but a dialogue?

### Entering the QuantHive

To solve this, I built **QuantHive**—a multi-agent environment powered by PettingZoo's AEC (Agent-Environment Cycle). Instead of one reckless AI, I broke institutional trading down into three adversarial roles:

1. **The Trader:** Wants to maximize profit and find alpha.
2. **The Portfolio Manager:** Controls capital allocation and wants steady growth without massive drawdowns.
3. **The Risk Manager:** Has the power to restrict position sizes and forcefully reduce exposure if things get dangerous.

They negotiate via observation message-passing. The environment rewards survival, not recklessness. The Risk Manager is actively rewarded for restricting trades during dangerous drawdowns, while the Trader has to figure out how to make money within the dynamically changing limits imposed by the others.

### From Floats to Thoughts: Semantic Reasoning

The breakthrough came when training the Qwen 2.5 1.5B model using GRPO (Group Relative Policy Optimization). 

Initially, the agents were fed raw float arrays (e.g., `0.284`). But to truly achieve "Auditable AI," I transitioned the environment to use **Semantic Reasoning**. Instead of a vector of 24 numbers, the AI "reads" the market state in human terms: *"RSI is 28.4 (oversold)"*.

This seemingly simple change leveraged the LLM's pre-trained world knowledge. We trained the model against 5 reward verifiers, enforcing not just profit, but *Format, Alignment, Risk, and Governance.*

### The Smoking Gun 

After 250 steps of GRPO training, the results were staggering. The Trader learned to fear the Risk Manager's interventions and began pre-emptively complying. 

Governance compliance shot from a random 7% up to **88%**, and Risk Limit Adherence hit **93%**. 

But the best part is *how* it compliance. Because we required Chain-of-Thought reasoning, the trained agent now outputs things like:

> *"...I also see that the portfolio's allocation of capital is nearing its limit (0.5). Given the Risk Manager's constraint on the size limit, I need to be cautious..."*

It doesn’t just follow the rules; it understands and explicitly cites them before taking action. 

I set out to see if AI could be taught institutional discipline. It turns out, when you build the right environment, it absolutely can.

---
*Check out the full project on GitHub and see the live multi-agent choregraphy on our Hugging Face Space! All links are available in the repository `README.md`.*
