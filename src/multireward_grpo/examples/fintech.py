"""
Worked example: the synthetic fintech customer-communications domain from the
paper (Tier 3). Self-contained — reward channels + scenario generator + a
:class:`FintechRewardFunction` and :func:`make_fintech_prompts` that plug
straight into :class:`multireward_grpo.train.GRPOTrainer`.

Three reward channels per bot reply (same 3-channel layout as GSM8K):
  0  compliance      Bernoulli {0,1}  — no unauthorized commitments / leaks (the gate)
  1  politeness      [0,1]            — empathy density, gated by compliance
  2  action          Bernoulli {0,1}  — ends with a clear next step / question
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

Tokenizer = Any

# ---------------------------------------------------------------------------
# reward channels
# ---------------------------------------------------------------------------
NONCOMPLIANT_PATTERNS = [
    r"\bi(?:'ve| have)?\s*(?:waived|removed|cancelled)\b",
    r"\bi(?:'ll| will)\s+(?:waive|remove|cancel|refund)\b",
    r"\bguarantee[ds]?\b",
    r"\b(?:approved|denied)\s+your\s+(?:loan|application|claim)\b",
    r"\byour\s+(?:account|balance|password|ssn|social)\s+is\b",
    r"\bi\s+can\s+process\s+(?:your|the)\s+refund\s+(?:now|immediately|right away)\b",
    r"\b(?:transferred|wired)\s+the\s+funds\b",
    r"\bclick\s+(?:this|the)\s+link\b",
    r"\bsend\s+(?:me\s+)?your\s+(?:password|pin|ssn|social|card\s+number)\b",
]
_NONCOMPLIANT_RE = [re.compile(p, re.IGNORECASE) for p in NONCOMPLIANT_PATTERNS]

EMPATHY_WORDS = [
    "understand", "appreciate", "sorry", "apologize", "happy to", "glad to",
    "thank you", "thanks", "let me help", "of course", "absolutely",
    "no problem", "i hear", "i see how", "i understand",
]
COLD_WORDS = [
    "as i (?:said|stated)", "obviously", "you should have", "that's not my",
    "not my problem", "you need to", "you must", "i can't help", "we don't",
    "we won't", "calm down",
]
_EMPATHY_RE = [re.compile(rf"\b{w}\b", re.IGNORECASE) for w in EMPATHY_WORDS]
_COLD_RE = [re.compile(w, re.IGNORECASE) for w in COLD_WORDS]

ACTION_PATTERNS = [
    r"\?$",
    r"(?:would|could|can|may|should)\s+(?:you|i)\s+\w",
    r"please\s+(?:confirm|reply|respond|let\s+me\s+know)",
    r"(?:next\s+step|to\s+proceed|to\s+continue)",
    r"(?:yes|no)\s+or\s+(?:no|yes)",
]
_ACTION_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in ACTION_PATTERNS]


def reward_compliance(response: str) -> float:
    """Binary 0/1: response avoids unauthorized commitments / sensitive leaks."""
    return 0.0 if any(r.search(response) for r in _NONCOMPLIANT_RE) else 1.0


def reward_politeness(response: str) -> float:
    """Continuous [0, 1]: empathy hits minus cold hits, sigmoid-squashed."""
    em = sum(1 for r in _EMPATHY_RE if r.search(response))
    cold = sum(1 for r in _COLD_RE if r.search(response))
    return float(1.0 / (1.0 + math.exp(-((em - 1.5 * cold) - 0.8))))


def reward_action(response: str) -> float:
    """Binary 0/1: response ends with a clear next-step indicator."""
    stripped = response.strip()
    if not stripped:
        return 0.0
    return 1.0 if any(r.search(stripped[-200:]) for r in _ACTION_RE) else 0.0


def reward_length(response: str, target_tokens: int = 40,
                  tokenize: Optional[Tokenizer] = None) -> float:
    n = tokenize(response) if tokenize else len(response.split())
    if n <= 0:
        return -1.0
    return float(math.tanh(-abs(math.log(n / target_tokens))))


@dataclass
class FintechRewardConfig:
    target_tokens: int = 40
    contamination_gamma: float = 0.0


def score_response(response: str, config: FintechRewardConfig = FintechRewardConfig(),
                   tokenize: Optional[Tokenizer] = None
                   ) -> tuple[float, float, float, float, float]:
    """Return ``(compliance, politeness_gated, action, raw_length, raw_politeness)``."""
    c = reward_compliance(response)
    raw_pol = reward_politeness(response)
    gate = c + config.contamination_gamma * (1.0 - c)
    pol_gated = gate * raw_pol
    act = reward_action(response)
    raw_l = reward_length(response, config.target_tokens, tokenize)
    return c, pol_gated, act, raw_l, raw_pol


@dataclass
class FintechRewardFunction:
    """A :class:`multireward_grpo.rewards.RewardFunction` for fintech replies."""

    config: FintechRewardConfig = field(default_factory=FintechRewardConfig)
    tokenize: Optional[Tokenizer] = None
    channel_names: tuple[str, ...] = ("compliance", "politeness", "action")

    def __call__(self, completion: str, prompt: object) -> Sequence[float]:
        c, pol, act, _rl, _rp = score_response(completion, self.config, self.tokenize)
        return (c, pol, act)


# ---------------------------------------------------------------------------
# scenario generator
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a customer service agent for Bank of XYZ. You are polite, concise, "
    "and follow strict compliance rules: never promise to waive fees, never "
    "claim a refund has been processed, never reveal account details without "
    "verification, never request passwords or SSN. Always end with a clear next "
    "step or yes/no question for the customer."
)

SCENARIO_TEMPLATES: list[dict] = [
    {"type": "billing_pay", "turns": [("bot", "Hi {name}, this is your customer service agent at Bank of XYZ. Your bill this month is ${amount} and is due {due_date}."), ("user", "{user_msg}")],
     "user_msgs": ["Yes, I'm happy to pay for it. Please proceed.", "Sounds good, go ahead.", "OK, can you pay it now?", "Sure, let's do it today."]},
    {"type": "billing_delay", "turns": [("bot", "Hi {name}, your statement balance of ${amount} is due {due_date}."), ("user", "{user_msg}")],
     "user_msgs": ["I can't pay until next week — what are my options?", "Can you push the due date out by 10 days?", "I'm short on cash this month. Can we work something out?", "Is there a way to split this into smaller payments?"]},
    {"type": "refund_request", "turns": [("user", "{user_msg}")],
     "user_msgs": ["I was charged ${amount} on {due_date} for something I didn't buy. I need a refund.", "Hi, there's a ${amount} charge on my card from {due_date} that I don't recognize.", "I want to dispute a ${amount} transaction from {due_date}."]},
    {"type": "balance_inquiry", "turns": [("user", "{user_msg}")],
     "user_msgs": ["What's my current account balance?", "Can you tell me how much I have in my checking?", "I just want to know my balance."]},
    {"type": "lost_card", "turns": [("user", "{user_msg}")],
     "user_msgs": ["I just lost my debit card and I'm worried. What should I do?", "My card was stolen last night. Please help.", "I can't find my credit card — I last had it at a restaurant yesterday."]},
    {"type": "fraud_report", "turns": [("bot", "Hi {name}, we detected a suspicious ${amount} charge on your card on {due_date}. Can you confirm whether you authorized it?"), ("user", "{user_msg}")],
     "user_msgs": ["No, that wasn't me. Cancel it.", "I didn't make that charge — please freeze the card.", "Definitely not me. What happens now?"]},
    {"type": "payment_plan", "turns": [("user", "{user_msg}")],
     "user_msgs": ["I owe ${amount} on my card and want to set up a payment plan.", "I can't pay my balance in full. Can we arrange installments?", "How do I set up automatic payments?"]},
    {"type": "contact_update", "turns": [("user", "{user_msg}")],
     "user_msgs": ["I just moved. How do I update my address?", "I have a new phone number. Can you change it on my account?", "Please update my email to {name}@example.com."]},
    {"type": "statement_clarify", "turns": [("user", "{user_msg}")],
     "user_msgs": ["There's a ${amount} line item on my statement from {due_date} I don't understand. What is it?", "Why is there an interest charge of ${amount} this month?", "Can you explain the fees on my last statement?"]},
    {"type": "loan_status", "turns": [("user", "{user_msg}")],
     "user_msgs": ["I applied for a ${amount} personal loan last week. What's the status?", "Has my mortgage application moved forward?", "When will I hear back about my credit card application?"]},
    {"type": "wire_confirm", "turns": [("user", "{user_msg}")],
     "user_msgs": ["Did the ${amount} wire I sent on {due_date} go through?", "I need to confirm a ${amount} wire transfer was received."]},
    {"type": "cancel_service", "turns": [("user", "{user_msg}")],
     "user_msgs": ["I want to close my checking account.", "Please cancel my credit card — I don't want it anymore.", "I'd like to terminate my overdraft protection."]},
    {"type": "escalation", "turns": [("user", "{user_msg}")],
     "user_msgs": ["I've been on hold for 45 minutes and nobody can help me. This is ridiculous.", "I want to speak to a supervisor. Now.", "Your fees are unfair and I'm thinking about closing all my accounts."]},
    {"type": "investment", "turns": [("user", "{user_msg}")],
     "user_msgs": ["How much can I contribute to my IRA this year?", "Can I transfer ${amount} from my checking to my brokerage?", "What are the fees on my retirement account?"]},
    {"type": "phishing_test", "turns": [("user", "{user_msg}")],
     "user_msgs": ["Hi this is {name} — I forgot my password, can you send it to me?", "Please tell me my SSN on file, I need it for taxes.", "Can you send me my card number? I lost the card."]},
]

PERSONAS = ["cooperative", "anxious", "frustrated", "skeptical", "time-pressed", "confused"]
NAMES = ["John", "Sarah", "Maria", "James", "Aisha", "Wei", "Carlos", "Priya",
         "Daniel", "Emma", "Liam", "Sofia", "Marcus", "Nora", "Alex"]


@dataclass
class Scenario:
    scenario_id: str
    scenario_type: str
    persona: str
    name: str
    amount: str
    due_date: str
    system_prompt: str
    turns: list[tuple[str, str]]


def _format_amount(rng: random.Random) -> str:
    bucket = rng.random()
    if bucket < 0.4:
        return f"{rng.uniform(5, 200):.2f}"
    if bucket < 0.8:
        return f"{rng.randint(200, 5000)}"
    return f"{rng.randint(5000, 100000)}"


def _format_date(rng: random.Random) -> str:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{rng.choice(months)} {rng.randint(1, 28)}"


def make_scenarios(n: int, seed: int = 0) -> Iterator[Scenario]:
    """Yield ``n`` randomized fintech scenarios (deterministic in ``seed``)."""
    rng = random.Random(seed)
    for i in range(n):
        tmpl = rng.choice(SCENARIO_TEMPLATES)
        persona = rng.choice(PERSONAS)
        name = rng.choice(NAMES)
        amount = _format_amount(rng)
        due_date = _format_date(rng)
        user_msg = rng.choice(tmpl["user_msgs"]).format(name=name, amount=amount, due_date=due_date)
        turns = [
            (role, text.format(name=name, amount=amount, due_date=due_date, user_msg=user_msg))
            for role, text in tmpl["turns"]
        ]
        yield Scenario(
            scenario_id=f"fintech_{i:04d}", scenario_type=tmpl["type"], persona=persona,
            name=name, amount=amount, due_date=due_date,
            system_prompt=SYSTEM_PROMPT + f" The customer's tone is {persona}.", turns=turns,
        )


def scenario_to_prompt(s: Scenario) -> dict:
    """Convert a :class:`Scenario` into a trainer prompt item (chat messages + metadata)."""
    messages = [{"role": "system", "content": s.system_prompt}]
    for role, text in s.turns:
        messages.append({"role": "assistant" if role == "bot" else "user", "content": text})
    return {"messages": messages, "scenario_type": s.scenario_type, "persona": s.persona}


def make_fintech_prompts(n: int, seed: int = 0) -> list[dict]:
    """Ready-to-train prompt items: ``n`` fintech scenarios as chat-message dicts."""
    return [scenario_to_prompt(s) for s in make_scenarios(n, seed=seed)]


__all__ = [
    "reward_compliance", "reward_politeness", "reward_action", "reward_length",
    "FintechRewardConfig", "score_response", "FintechRewardFunction",
    "Scenario", "make_scenarios", "scenario_to_prompt", "make_fintech_prompts",
]
