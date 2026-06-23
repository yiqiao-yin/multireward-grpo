"""
LLM reward functions for the Thm 3 validation pipeline.

Three verifiable reward channels for math-reasoning generations:
  - correctness: extract final numeric answer, exact-match against gold (Bernoulli)
  - length:      log-ratio of token count to a target length (continuous, bounded)
  - format:      presence of \\boxed{...} or "#### N" closing pattern (Bernoulli)

All functions are pure-Python, no GPU/transformer dependency. They operate on
already-generated text. The reward matrix shape produced is (m, R=3) per prompt.

Each function maps back to a claim in problem-statement.md:
  - correctness drives Prop 1's "high-variance harder objective" story (it's
    the high-variance channel when the model is weak)
  - length is the "easier" channel for Prop 4's gating story (corretness gates
    length: length should count only when correct)
  - format is the secondary verifiable signal used in GDPO's anchor
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# answer extraction (GSM8K-flavored)
# ---------------------------------------------------------------------------
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_GSM_FINAL_RE = re.compile(r"####\s*(-?[\d,]+\.?\d*)")
_LAST_NUMBER_RE = re.compile(r"(-?\d[\d,]*\.?\d*)")


def extract_answer(text: str) -> Optional[str]:
    """Try `\\boxed{x}`, then `#### x`, then the last number in the text.

    Returns the normalized numeric string with commas stripped, or None if no
    number was found at all. Matches the GSM8K / MATH community convention.
    """
    if m := _BOXED_RE.search(text):
        return _normalize_number(m.group(1))
    if m := _GSM_FINAL_RE.search(text):
        return _normalize_number(m.group(1))
    nums = _LAST_NUMBER_RE.findall(text)
    if nums:
        return _normalize_number(nums[-1])
    return None


def _normalize_number(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "")
    # strip trailing punctuation
    s = s.rstrip(".")
    # canonicalize integer-valued floats (e.g. "42.0" -> "42")
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return repr(f)
    except ValueError:
        return s


# ---------------------------------------------------------------------------
# reward channels
# ---------------------------------------------------------------------------
def reward_correctness(generation: str, gold_answer: str) -> float:
    """Binary 0/1: extracted answer matches gold (after normalization)."""
    extracted = extract_answer(generation)
    if extracted is None:
        return 0.0
    return 1.0 if extracted == _normalize_number(gold_answer) else 0.0


def reward_length(generation: str, target_tokens: int = 200,
                  tokenize: Optional[callable] = None) -> float:
    """Bounded continuous in roughly [-1, 1]: prefers responses close to a
    target token budget. Overshooting and undershooting are both penalized
    via tanh of the log ratio.

    `tokenize` is a callable returning the number of tokens; if None we fall
    back to whitespace word count, which is a coarse proxy. For real
    experiments pass a tokenizer-backed counter.
    """
    import math
    n = tokenize(generation) if tokenize else len(generation.split())
    if n <= 0:
        return -1.0
    log_ratio = math.log(n / target_tokens)
    return math.tanh(-abs(log_ratio))


def reward_format(generation: str) -> float:
    """Binary 0/1: response ends with the expected delimiter pattern."""
    return 1.0 if (_BOXED_RE.search(generation) or _GSM_FINAL_RE.search(generation)) else 0.0


# ---------------------------------------------------------------------------
# convenience: full reward vector + contamination knob for Prop 4
# ---------------------------------------------------------------------------
@dataclass
class RewardConfig:
    """Names and weights for the R reward channels.

    `contamination_gamma` controls Prop 4's contamination knob: the length
    reward is multiplied by (correct + gamma*(1-correct)) so that, at
    gamma=0, length counts only when correct (the desired gated behavior);
    at gamma=1, length counts unconditionally; at gamma>1, length is *over*-
    rewarded on incorrect answers (the pathological case Prop 4 analyses).
    """
    target_tokens: int = 200
    weights: tuple[float, ...] = (1.0, 1.0, 0.5)  # (correctness, length, format)
    contamination_gamma: float = 0.0


def score_generation(generation: str, gold_answer: str,
                     config: RewardConfig = RewardConfig(),
                     tokenize: Optional[callable] = None) -> tuple[float, float, float, float]:
    """Return (correctness, processed_length, format, raw_length).

    The fourth value is the ungated length score, returned separately so the
    Prop 4 γ sweep can be done as cheap post-processing on saved rollouts.
    The processed_length is `(c + γ(1-c)) * raw_length` with γ from
    `config.contamination_gamma` — at γ=0 the length is fully gated by
    correctness (Prop 4 "zero-fill" target); at γ=1 it counts equally on
    right and wrong; at γ>1 it is over-rewarded on wrong (the pathological
    regime Prop 4 analyses).
    """
    c = reward_correctness(generation, gold_answer)
    raw_length = reward_length(generation, config.target_tokens, tokenize)
    gate = c + config.contamination_gamma * (1.0 - c)
    length = gate * raw_length
    fmt = reward_format(generation)
    return c, length, fmt, raw_length


__all__ = [
    "extract_answer",
    "reward_correctness",
    "reward_length",
    "reward_format",
    "score_generation",
    "RewardConfig",
]
