"""Worked examples reproducing the paper's domains.

- ``fintech``: synthetic customer-comms domain (Tier 3 training).
- ``gsm8k``: math reasoning (Tier 2, Theorem 3).
"""
from .fintech import (
    FintechRewardConfig,
    FintechRewardFunction,
    make_fintech_prompts,
    make_scenarios,
    score_response,
)
from .gsm8k import gsm8k_prompts_with_gold, load_gsm8k_prompts

__all__ = [
    "FintechRewardFunction",
    "FintechRewardConfig",
    "score_response",
    "make_scenarios",
    "make_fintech_prompts",
    "load_gsm8k_prompts",
    "gsm8k_prompts_with_gold",
]
