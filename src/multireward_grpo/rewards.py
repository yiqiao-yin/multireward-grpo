"""
Reward-channel framework for multi-reward GRPO.

A *reward function* maps one generated completion to a vector of ``R`` channel
scores. The training loop (:mod:`multireward_grpo.train`) only needs an object
that, given a completion string and the prompt it came from, returns ``R``
floats — see :class:`RewardFunction`. Channel 0 is treated as the "gate" channel
by the conditioning utilities (e.g. correctness / compliance).

This module also ships the paper's GSM8K math reward channels
(correctness / length / format) both as standalone functions and packaged as a
:class:`MathRewardFunction` you can drop straight into the trainer.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

Tokenizer = Callable[[str], int]


@runtime_checkable
class RewardFunction(Protocol):
    """The contract the trainer expects from a reward function.

    Implement either as a callable object with ``channel_names`` /
    ``__call__`` attributes, or as any object satisfying this protocol.
    """

    channel_names: tuple[str, ...]

    def __call__(self, completion: str, prompt: object) -> Sequence[float]:
        """Return ``R == len(channel_names)`` channel scores for one completion.

        ``prompt`` is whatever item the caller passed in the prompt list — a
        raw string, a chat-message list, or a dict carrying metadata such as
        the gold answer. Reward functions pull what they need from it.
        """
        ...


# ---------------------------------------------------------------------------
# answer extraction (GSM8K / MATH flavored)
# ---------------------------------------------------------------------------
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_GSM_FINAL_RE = re.compile(r"####\s*(-?[\d,]+\.?\d*)")
_LAST_NUMBER_RE = re.compile(r"(-?\d[\d,]*\.?\d*)")


def extract_answer(text: str) -> Optional[str]:
    """Try ``\\boxed{x}``, then ``#### x``, then the last number in the text."""
    if m := _BOXED_RE.search(text):
        return _normalize_number(m.group(1))
    if m := _GSM_FINAL_RE.search(text):
        return _normalize_number(m.group(1))
    nums = _LAST_NUMBER_RE.findall(text)
    if nums:
        return _normalize_number(nums[-1])
    return None


def _normalize_number(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else repr(f)
    except ValueError:
        return s


# ---------------------------------------------------------------------------
# individual reward channels
# ---------------------------------------------------------------------------
def reward_correctness(generation: str, gold_answer: str) -> float:
    """Binary 0/1: extracted answer matches the gold answer after normalization."""
    extracted = extract_answer(generation)
    if extracted is None:
        return 0.0
    return 1.0 if extracted == _normalize_number(gold_answer) else 0.0


def reward_length(
    generation: str,
    target_tokens: int = 200,
    tokenize: Optional[Tokenizer] = None,
) -> float:
    """Bounded continuous in ~[-1, 1]: prefers completions near ``target_tokens``.

    ``tokenize`` returns a token count; if ``None`` we fall back to whitespace
    word count. Both overshoot and undershoot are penalized via ``tanh`` of the
    absolute log-ratio.
    """
    n = tokenize(generation) if tokenize else len(generation.split())
    if n <= 0:
        return -1.0
    return math.tanh(-abs(math.log(n / target_tokens)))


def reward_format(generation: str) -> float:
    """Binary 0/1: completion contains a ``\\boxed{...}`` or ``#### N`` answer."""
    return 1.0 if (_BOXED_RE.search(generation) or _GSM_FINAL_RE.search(generation)) else 0.0


# ---------------------------------------------------------------------------
# math reward bundle (correctness, length, format)
# ---------------------------------------------------------------------------
@dataclass
class RewardConfig:
    """Configuration for the GSM8K math reward channels.

    ``contamination_gamma`` is Proposition 4's contamination knob: the length
    reward is scaled by ``(correct + gamma * (1 - correct))`` so that at
    ``gamma=0`` length counts only when correct (the gated target), at
    ``gamma=1`` it counts unconditionally, and at ``gamma>1`` it is over-rewarded
    on wrong answers (the pathological regime Prop 4 analyses).
    """

    target_tokens: int = 200
    weights: tuple[float, ...] = (1.0, 1.0, 0.5)  # (correctness, length, format)
    contamination_gamma: float = 0.0


def score_generation(
    generation: str,
    gold_answer: str,
    config: RewardConfig = RewardConfig(),
    tokenize: Optional[Tokenizer] = None,
) -> tuple[float, float, float, float]:
    """Return ``(correctness, processed_length, format, raw_length)``.

    ``processed_length = (c + gamma*(1-c)) * raw_length``. The ungated
    ``raw_length`` is returned separately so a Proposition-4 gamma sweep can be
    done as cheap post-processing on saved rollouts.
    """
    c = reward_correctness(generation, gold_answer)
    raw_length = reward_length(generation, config.target_tokens, tokenize)
    gate = c + config.contamination_gamma * (1.0 - c)
    length = gate * raw_length
    fmt = reward_format(generation)
    return c, length, fmt, raw_length


@dataclass
class MathRewardFunction:
    """A :class:`RewardFunction` for GSM8K-style math, ready for the trainer.

    Expects each prompt item to be a mapping carrying the gold answer under one
    of ``gold`` / ``gold_answer`` / ``answer``. Returns the
    ``(correctness, length, format)`` channel vector.
    """

    config: RewardConfig = field(default_factory=RewardConfig)
    tokenize: Optional[Tokenizer] = None
    channel_names: tuple[str, ...] = ("correctness", "length", "format")

    def __call__(self, completion: str, prompt: object) -> Sequence[float]:
        gold = _extract_gold(prompt)
        c, length, fmt, _raw = score_generation(
            completion, gold, self.config, self.tokenize
        )
        return (c, length, fmt)


def _extract_gold(prompt: object) -> str:
    if isinstance(prompt, dict):
        for key in ("gold", "gold_answer", "answer"):
            if key in prompt:
                return str(prompt[key])
    raise ValueError(
        "MathRewardFunction needs the gold answer; pass prompt items as dicts "
        "with a 'gold' (or 'gold_answer'/'answer') key."
    )


__all__ = [
    "RewardFunction",
    "RewardConfig",
    "MathRewardFunction",
    "extract_answer",
    "reward_correctness",
    "reward_length",
    "reward_format",
    "score_generation",
]
