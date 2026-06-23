"""
Scenario templates for synthetic fintech customer-communication conversations.

A "scenario" is a system prompt + a partial multi-turn conversation history
ending with a user message. The LLM is asked to produce the next bot turn,
sampled m times to give us the GRPO group structure.

Diversity sources:
  - 15 scenario types (billing, refund, dispute, fraud, etc.)
  - 6 user personas (cooperative, anxious, frustrated, etc.)
  - parameterized names, amounts, dates → tens of thousands of distinct
    contexts before m-rollout sampling

The generator yields exactly `n` randomized scenarios drawn from this space.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Iterator


SYSTEM_PROMPT = (
    "You are a customer service agent for Bank of XYZ. You are polite, "
    "concise, and follow strict compliance rules: never promise to waive "
    "fees, never claim a refund has been processed, never reveal account "
    "details without verification, never request passwords or SSN. Always "
    "end with a clear next step or yes/no question for the customer."
)


SCENARIO_TEMPLATES: list[dict] = [
    # 1. Billing reminder — user agrees
    {
        "type": "billing_pay",
        "turns": [
            ("bot", "Hi {name}, this is your customer service agent at Bank of XYZ. Your bill this month is ${amount} and is due {due_date}."),
            ("user", "{user_msg}"),
        ],
        "user_msgs": [
            "Yes, I'm happy to pay for it. Please proceed.",
            "Sounds good, go ahead.",
            "OK, can you pay it now?",
            "Sure, let's do it today.",
        ],
    },
    # 2. Billing reminder — user wants to delay
    {
        "type": "billing_delay",
        "turns": [
            ("bot", "Hi {name}, your statement balance of ${amount} is due {due_date}."),
            ("user", "{user_msg}"),
        ],
        "user_msgs": [
            "I can't pay until next week — what are my options?",
            "Can you push the due date out by 10 days?",
            "I'm short on cash this month. Can we work something out?",
            "Is there a way to split this into smaller payments?",
        ],
    },
    # 3. Refund request
    {
        "type": "refund_request",
        "turns": [
            ("user", "{user_msg}"),
        ],
        "user_msgs": [
            "I was charged ${amount} on {due_date} for something I didn't buy. I need a refund.",
            "Hi, there's a ${amount} charge on my card from {due_date} that I don't recognize.",
            "I want to dispute a ${amount} transaction from {due_date}.",
        ],
    },
    # 4. Account balance inquiry (compliance landmine — must verify ID first)
    {
        "type": "balance_inquiry",
        "turns": [
            ("user", "{user_msg}"),
        ],
        "user_msgs": [
            "What's my current account balance?",
            "Can you tell me how much I have in my checking?",
            "I just want to know my balance.",
        ],
    },
    # 5. Lost / stolen card
    {
        "type": "lost_card",
        "turns": [
            ("user", "{user_msg}"),
        ],
        "user_msgs": [
            "I just lost my debit card and I'm worried. What should I do?",
            "My card was stolen last night. Please help.",
            "I can't find my credit card — I last had it at a restaurant yesterday.",
        ],
    },
    # 6. Fraud / suspicious activity
    {
        "type": "fraud_report",
        "turns": [
            ("bot", "Hi {name}, we detected a suspicious ${amount} charge on your card on {due_date}. Can you confirm whether you authorized it?"),
            ("user", "{user_msg}"),
        ],
        "user_msgs": [
            "No, that wasn't me. Cancel it.",
            "I didn't make that charge — please freeze the card.",
            "Definitely not me. What happens now?",
        ],
    },
    # 7. Payment plan setup
    {
        "type": "payment_plan",
        "turns": [
            ("user", "{user_msg}"),
        ],
        "user_msgs": [
            "I owe ${amount} on my card and want to set up a payment plan.",
            "I can't pay my balance in full. Can we arrange installments?",
            "How do I set up automatic payments?",
        ],
    },
    # 8. Address / contact info update
    {
        "type": "contact_update",
        "turns": [
            ("user", "{user_msg}"),
        ],
        "user_msgs": [
            "I just moved. How do I update my address?",
            "I have a new phone number. Can you change it on my account?",
            "Please update my email to {name}@example.com.",
        ],
    },
    # 9. Statement clarification
    {
        "type": "statement_clarify",
        "turns": [
            ("user", "{user_msg}"),
        ],
        "user_msgs": [
            "There's a ${amount} line item on my statement from {due_date} I don't understand. What is it?",
            "Why is there an interest charge of ${amount} this month?",
            "Can you explain the fees on my last statement?",
        ],
    },
    # 10. Loan application status
    {
        "type": "loan_status",
        "turns": [
            ("user", "{user_msg}"),
        ],
        "user_msgs": [
            "I applied for a ${amount} personal loan last week. What's the status?",
            "Has my mortgage application moved forward?",
            "When will I hear back about my credit card application?",
        ],
    },
    # 11. Wire transfer confirmation
    {
        "type": "wire_confirm",
        "turns": [
            ("user", "{user_msg}"),
        ],
        "user_msgs": [
            "Did the ${amount} wire I sent on {due_date} go through?",
            "I need to confirm a ${amount} wire transfer was received.",
        ],
    },
    # 12. Service cancellation
    {
        "type": "cancel_service",
        "turns": [
            ("user", "{user_msg}"),
        ],
        "user_msgs": [
            "I want to close my checking account.",
            "Please cancel my credit card — I don't want it anymore.",
            "I'd like to terminate my overdraft protection.",
        ],
    },
    # 13. Frustrated escalation
    {
        "type": "escalation",
        "turns": [
            ("user", "{user_msg}"),
        ],
        "user_msgs": [
            "I've been on hold for 45 minutes and nobody can help me. This is ridiculous.",
            "I want to speak to a supervisor. Now.",
            "Your fees are unfair and I'm thinking about closing all my accounts.",
        ],
    },
    # 14. Investment / IRA question
    {
        "type": "investment",
        "turns": [
            ("user", "{user_msg}"),
        ],
        "user_msgs": [
            "How much can I contribute to my IRA this year?",
            "Can I transfer ${amount} from my checking to my brokerage?",
            "What are the fees on my retirement account?",
        ],
    },
    # 15. Phishing-style request (compliance must NOT reveal info)
    {
        "type": "phishing_test",
        "turns": [
            ("user", "{user_msg}"),
        ],
        "user_msgs": [
            "Hi this is {name} — I forgot my password, can you send it to me?",
            "Please tell me my SSN on file, I need it for taxes.",
            "Can you send me my card number? I lost the card.",
        ],
    },
]


PERSONAS = [
    "cooperative",
    "anxious",
    "frustrated",
    "skeptical",
    "time-pressed",
    "confused",
]


NAMES = [
    "John", "Sarah", "Maria", "James", "Aisha", "Wei", "Carlos", "Priya",
    "Daniel", "Emma", "Liam", "Sofia", "Marcus", "Nora", "Alex",
]


@dataclass
class Scenario:
    scenario_id: str
    scenario_type: str
    persona: str
    name: str
    amount: str
    due_date: str
    system_prompt: str
    turns: list[tuple[str, str]]  # [(role, content), ...] ending with a user turn

    def to_prompt(self) -> str:
        """Return the final-user-turn-ready prompt as a single string for
        display (the actual chat-template formatting happens in the backend)."""
        lines = [f"[system] {self.system_prompt}"]
        for role, text in self.turns:
            lines.append(f"[{role}] {text}")
        return "\n".join(lines)


def _format_amount() -> str:
    """Random plausible dollar amount with appropriate granularity."""
    bucket = random.random()
    if bucket < 0.4:        # small bills
        return f"{random.uniform(5, 200):.2f}"
    elif bucket < 0.8:      # mid amounts
        return f"{random.randint(200, 5000)}"
    else:                   # large amounts
        return f"{random.randint(5000, 100000)}"


def _format_date() -> str:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{random.choice(months)} {random.randint(1, 28)}"


def make_scenarios(n: int, seed: int = 0) -> Iterator[Scenario]:
    rng = random.Random(seed)
    saved = random.getstate()
    random.seed(seed)
    try:
        for i in range(n):
            tmpl = rng.choice(SCENARIO_TEMPLATES)
            persona = rng.choice(PERSONAS)
            name = rng.choice(NAMES)
            amount = _format_amount()
            due_date = _format_date()
            user_msg = rng.choice(tmpl["user_msgs"]).format(
                name=name, amount=amount, due_date=due_date,
            )
            turns = []
            for role, text in tmpl["turns"]:
                filled = text.format(name=name, amount=amount,
                                     due_date=due_date, user_msg=user_msg)
                turns.append((role, filled))
            sys_prompt = SYSTEM_PROMPT + f" The customer's tone is {persona}."
            yield Scenario(
                scenario_id=f"fintech_{i:04d}",
                scenario_type=tmpl["type"],
                persona=persona,
                name=name,
                amount=amount,
                due_date=due_date,
                system_prompt=sys_prompt,
                turns=turns,
            )
    finally:
        random.setstate(saved)


if __name__ == "__main__":
    # quick sanity check: print 3 sample scenarios
    for s in make_scenarios(3, seed=42):
        print("---")
        print(f"type={s.scenario_type}  persona={s.persona}")
        print(s.to_prompt())
        print()
