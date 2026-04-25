"""
Lightweight verifier helpers for the multi-agent GRPO notebook and trainer.

These functions intentionally avoid importing the training stack so notebooks can
preview prompts and reward functions without loading model or trainer deps.
"""

from __future__ import annotations

import json
import re

import numpy as np


def _extract_json_action(completion: str):
    match = re.search(r"<action>\s*({.*?})\s*</action>", completion, re.DOTALL)
    if not match:
        return None
    return json.loads(match.group(1))


def _extract_signal_value(prompt: str, key: str):
    json_match = re.search(rf'"{key}"\s*:\s*(-?[\d\.]+)', prompt)
    if json_match:
        return float(json_match.group(1))

    plain_match = re.search(rf"{key}\s*[:=]\s*(-?[\d\.]+)", prompt)
    if plain_match:
        return float(plain_match.group(1))

    return None


def risk_reward_func_multiagent(prompts, completions, **kwargs) -> list[float]:
    """Read the Risk Manager limit from the prompt and reward compliant sizing."""

    rewards = []
    for prompt, completion in zip(prompts, completions):
        try:
            limit = _extract_signal_value(prompt, "rm_size_limit")
            if limit is None:
                limit = _extract_signal_value(prompt, "position_limit")
            if limit is None:
                limit = 1.0

            data = _extract_json_action(completion)
            if data is None:
                rewards.append(0.0)
                continue

            size = float(data.get("size", 0.0))
            score = 0.7 if size <= limit else 0.0

            try:
                thought = completion.split("<thought>")[1].split("</thought>")[0].lower()
                if any(kw in thought for kw in ["risk", "limit", "constraint", "size_limit"]):
                    score += 0.3
            except (IndexError, AttributeError):
                pass

            rewards.append(score)
        except Exception:
            rewards.append(0.0)

    return rewards


def governance_reward_func_multiagent(prompts, completions, **kwargs) -> list[float]:
    """Score compliance against both Risk Manager and Portfolio Manager limits."""

    rewards = []
    for prompt, completion in zip(prompts, completions):
        try:
            data = _extract_json_action(completion)
            if data is None:
                rewards.append(0.0)
                continue

            size = float(data.get("size", 0.0))
            direction = int(data.get("direction", 0))

            limit = _extract_signal_value(prompt, "rm_size_limit")
            if limit is None:
                limit = _extract_signal_value(prompt, "position_limit")
            if limit is None:
                limit = 1.0

            pm_cap = _extract_signal_value(prompt, "pm_cap_alloc")
            effective_limit = min(limit, pm_cap) if pm_cap is not None else limit

            score = 0.0
            if size <= effective_limit:
                score += 0.40
                if 0 < size <= effective_limit * 0.8:
                    score += 0.20
            else:
                score -= 0.50

            try:
                thought = completion.split("<thought>")[1].split("</thought>")[0].lower()
                governance_keywords = [
                    "risk",
                    "limit",
                    "constraint",
                    "compliance",
                    "conservative",
                    "governance",
                    "restrict",
                    "drawdown",
                    "cap",
                    "position limit",
                    "size_limit",
                    "risk manager",
                    "portfolio manager",
                    "allocation",
                ]
                if any(kw in thought for kw in governance_keywords):
                    score += 0.20
            except (IndexError, AttributeError):
                pass

            if direction != 0:
                score += 0.20

            rewards.append(float(np.clip(score, 0.0, 1.0)))
        except Exception:
            rewards.append(0.0)

    return rewards


__all__ = [
    "governance_reward_func_multiagent",
    "risk_reward_func_multiagent",
]
