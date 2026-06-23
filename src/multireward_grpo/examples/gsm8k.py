"""
Worked example: GSM8K math reasoning (the paper's Tier-2 Theorem 3 domain).

Loads GSM8K via the ``datasets`` library and returns trainer-ready prompt items
(dicts with the question as the user message and the gold answer under ``gold``),
pairable with :class:`multireward_grpo.rewards.MathRewardFunction`.
"""
from __future__ import annotations

import re

_GSM_GOLD_RE = re.compile(r"####\s*(-?[\d,]+\.?\d*)")


def _extract_gold(answer_field: str) -> str:
    m = _GSM_GOLD_RE.search(answer_field)
    return (m.group(1).replace(",", "") if m else answer_field).strip()


def load_gsm8k_prompts(n: int = 100, split: str = "test", seed: int = 0) -> list[dict]:
    """Return ``n`` GSM8K prompt items: ``{"prompt": question, "gold": answer}``.

    Requires the ``datasets`` package (a base dependency). Pair with
    :class:`multireward_grpo.rewards.MathRewardFunction`, whose channels are
    (correctness, length, format).
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [
        {"prompt": row["question"], "gold": _extract_gold(row["answer"])}
        for row in ds
    ]


def gsm8k_prompts_with_gold(n: int = 100, split: str = "test", seed: int = 0) -> list[tuple[str, str]]:
    """Return ``[(question, gold), ...]`` pairs for the generation pipeline
    (:func:`multireward_grpo.generation.run_corpus`)."""
    return [(p["prompt"], p["gold"]) for p in load_gsm8k_prompts(n, split, seed)]


__all__ = ["load_gsm8k_prompts", "gsm8k_prompts_with_gold"]
