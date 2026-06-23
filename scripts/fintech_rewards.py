"""
Reward channels for the synthetic fintech customer-communication dataset.

Structured to mirror GSM8K's (correctness, length, format) layout, so the
exact same Thm 3 + Prop 4 analysis machinery applies. The "harder" gate that
plays the role of correctness is **compliance** (the response must not make
unauthorized commitments). The "easier" channel that gets conditioned on it
is **politeness/empathy**. We keep a length channel and a format/action
channel for completeness — same 3+1 reward vector as GSM8K, plus raw_length
saved separately for Prop 4 γ-sweep.

Reward channels (per bot response):
  0  compliance       Bernoulli {0, 1}     — passes the no-unauthorized-commit check
  1  politeness       continuous [0, 1]     — empathy-language density (gated by compliance in scoring)
  2  action_clarity   Bernoulli {0, 1}     — ends with a clear next-step / yes-no question

Plus the raw scalars (saved separately for Prop 4):
  raw_politeness     continuous [0, 1]     — politeness BEFORE the compliance gate
  raw_length         continuous [-1, 0]    — tanh(-|log(n_tokens / target)|)
"""
from __future__ import annotations
import re
import math
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# compliance — pattern match for forbidden / risky language
# ---------------------------------------------------------------------------
# Each pattern is a thing a real fintech CS bot should NOT say without
# verifying identity / escalating to a human. Hitting any of these → compliance=0.
NONCOMPLIANT_PATTERNS = [
    r"\bi(?:'ve| have)?\s*(?:waived|removed|cancelled)\b",           # unilateral fee waiver
    r"\bi(?:'ll| will)\s+(?:waive|remove|cancel|refund)\b",          # promised waiver/refund
    r"\bguarantee[ds]?\b",                                            # absolute guarantee
    r"\b(?:approved|denied)\s+your\s+(?:loan|application|claim)\b",  # decision without auth
    r"\byour\s+(?:account|balance|password|ssn|social)\s+is\b",      # leaking account details
    r"\bi\s+can\s+process\s+(?:your|the)\s+refund\s+(?:now|immediately|right away)\b",
    r"\b(?:transferred|wired)\s+the\s+funds\b",                       # claiming a transfer happened
    r"\bclick\s+(?:this|the)\s+link\b",                              # phishing-shaped (CS shouldn't push links)
    r"\bsend\s+(?:me\s+)?your\s+(?:password|pin|ssn|social|card\s+number)\b",  # asking for sensitive info
]
_NONCOMPLIANT_RE = [re.compile(p, re.IGNORECASE) for p in NONCOMPLIANT_PATTERNS]


def reward_compliance(response: str) -> float:
    """Binary 0/1: response avoids unauthorized commitments / sensitive leaks."""
    for r in _NONCOMPLIANT_RE:
        if r.search(response):
            return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# politeness — empathy-language density
# ---------------------------------------------------------------------------
EMPATHY_WORDS = [
    "understand", "appreciate", "sorry", "apologize", "happy to",
    "glad to", "thank you", "thanks", "let me help", "of course",
    "absolutely", "no problem", "i hear", "i see how", "i understand",
]
COLD_WORDS = [
    "as i (?:said|stated)", "obviously", "you should have",
    "that's not my", "not my problem", "you need to", "you must",
    "i can't help", "we don't", "we won't", "calm down",
]
_EMPATHY_RE = [re.compile(rf"\b{w}\b", re.IGNORECASE) for w in EMPATHY_WORDS]
_COLD_RE = [re.compile(w, re.IGNORECASE) for w in COLD_WORDS]


def reward_politeness(response: str) -> float:
    """Continuous [0, 1]. Count empathy hits, subtract cold hits, squash."""
    em = sum(1 for r in _EMPATHY_RE if r.search(response))
    cold = sum(1 for r in _COLD_RE if r.search(response))
    raw = em - 1.5 * cold
    # squash: 0 hits → 0.3; 2 empathy → ~0.7; 4+ → ~0.95
    return float(1.0 / (1.0 + math.exp(-(raw - 0.8))))


# ---------------------------------------------------------------------------
# action clarity — does the response end with a clear next step?
# ---------------------------------------------------------------------------
ACTION_PATTERNS = [
    r"\?$",                                                           # ends with a question
    r"(?:would|could|can|may|should)\s+(?:you|i)\s+\w",              # asking the user
    r"please\s+(?:confirm|reply|respond|let\s+me\s+know)",            # explicit call-to-action
    r"(?:next\s+step|to\s+proceed|to\s+continue)",
    r"(?:yes|no)\s+or\s+(?:no|yes)",                                  # explicit Y/N prompt
]
_ACTION_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in ACTION_PATTERNS]


def reward_action(response: str) -> float:
    """Binary 0/1: response ends with a clear next-step indicator."""
    stripped = response.strip()
    if not stripped:
        return 0.0
    last_chunk = stripped[-200:]  # check the last paragraph-ish
    for r in _ACTION_RE:
        if r.search(last_chunk):
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# length — same shape as GSM8K's, but target shorter (40 tokens; CS replies should be brief)
# ---------------------------------------------------------------------------
def reward_length(response: str, target_tokens: int = 40,
                  tokenize: Optional[callable] = None) -> float:
    n = tokenize(response) if tokenize else len(response.split())
    if n <= 0:
        return -1.0
    log_ratio = math.log(n / target_tokens)
    return float(math.tanh(-abs(log_ratio)))


# ---------------------------------------------------------------------------
# full scoring
# ---------------------------------------------------------------------------
@dataclass
class FintechRewardConfig:
    target_tokens: int = 40
    # Prop 4 contamination knob — kept at 0 by default (politeness fully gated
    # by compliance). The Prop 4 sweep recomputes contaminated_politeness from
    # raw_politeness at analysis time.
    contamination_gamma: float = 0.0


def score_response(response: str, config: FintechRewardConfig = FintechRewardConfig(),
                   tokenize: Optional[callable] = None
                  ) -> tuple[float, float, float, float, float]:
    """Return (compliance, politeness_gated, action, raw_length, raw_politeness).

    politeness_gated = (compliance + γ(1-compliance)) * raw_politeness — at
    γ=0 (default), politeness only counts when compliance passes. This is the
    Prop 4 "harder reward gates easier reward" structure, with compliance
    playing the role of correctness and politeness playing the role of length.

    The 3-channel reward vector used downstream is
    (compliance, politeness_gated, action), with raw_length and raw_politeness
    saved separately for the γ-sweep / length analysis.
    """
    c = reward_compliance(response)
    raw_pol = reward_politeness(response)
    gate = c + config.contamination_gamma * (1.0 - c)
    pol_gated = gate * raw_pol
    act = reward_action(response)
    raw_l = reward_length(response, config.target_tokens, tokenize)
    return c, pol_gated, act, raw_l, raw_pol


__all__ = [
    "reward_compliance", "reward_politeness", "reward_action", "reward_length",
    "FintechRewardConfig", "score_response",
]
