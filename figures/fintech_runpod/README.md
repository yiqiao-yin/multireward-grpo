---
license: cc-by-4.0
language:
- en
size_categories:
- 1K<n<10K
tags:
- reinforcement-learning
- grpo
- multi-reward
- llm
- fintech
- customer-service
- synthetic
- variance-reduction
task_categories:
- text-generation
- reinforcement-learning
pretty_name: Multi-Reward GRPO — Synthetic Fintech Customer-Comms
---

# Multi-Reward GRPO — Synthetic Fintech Customer Communications

Synthetic multi-turn customer-service conversations for a fictional bank
("Bank of XYZ"), generated for the empirical Section of **"Conditioned
Multi-Reward Advantage Estimation: A Finite-Sample Analysis"**.

Each conversation ends with `m` parallel sampled bot replies, each scored
on three verifiable reward channels designed for fintech customer service.
This is the multi-reward GRPO group structure on a *real* generation
distribution — analog to GSM8K but in a domain where multi-reward shaping
matters most: compliance gating politeness/empathy.

## Why fintech, why this structure

Fintech customer service is a high-stakes setting where the model's
*compliance* (no unauthorized fee waivers, no leaking account details, no
asking for SSN/passwords) is non-negotiable, and the easier-to-satisfy
channels (politeness, brevity, clear next-step) only matter conditional on
compliance. That's exactly the gating structure Proposition 4 in the paper
analyzes, and it provides a non-math-reasoning anchor for the Theorem 3
correlation-floor claim.

## Reward channels

| Channel | Type | Definition | Role in paper |
|---|---|---|---|
| `compliance` | Bernoulli {0, 1} | Response avoids any of 9 patterns: unilateral fee waivers, claims about refund processing, leaks of account details, requests for sensitive info, phishing-shaped link pushes, etc. | "Harder gate" — analog of correctness on GSM8K |
| `politeness_gated` | Continuous [0, 1] | Sigmoid(empathy hits − 1.5·cold hits − 0.8), then multiplied by compliance | "Easier conditioned" — analog of `length·correct` |
| `action` | Bernoulli {0, 1} | Response ends with a clear next step (question, call-to-action, Y/N prompt) | Third channel for resolution (Prop 2) |

Plus saved separately for the Prop 4 γ-sweep:

| Field | Type | Definition |
|---|---|---|
| `raw_politeness` | Continuous [0, 1] | Politeness BEFORE the compliance gate |
| `raw_length` | Continuous [-1, 0] | `tanh(-|log(n_tokens / 40)|)` — target 40-token reply |

## Diversity

- **15 scenario types**: billing reminder (pay/delay), refund request,
  balance inquiry (compliance landmine — must verify ID first), lost/stolen
  card, fraud report, payment plan, address update, statement clarification,
  loan/credit-card application status, wire transfer confirmation, account
  cancellation, frustrated escalation, investment/IRA question, **phishing
  test** (compliance must NOT reveal info).
- **6 user personas**: cooperative, anxious, frustrated, skeptical,
  time-pressed, confused.
- Randomized names, dollar amounts (spanning $5 to $100K), and due dates.
- Generated with Qwen2.5-7B-Instruct at temperature 0.8 to maximize
  reply diversity within scenarios.

## File layout

```
fintech_rewards.npz             — main reward tensor (numpy)
  rewards        : (P, K, m, R=3) — (compliance, politeness_gated, action)
  raw_length     : (P, K, m)      — ungated length per rollout
  raw_politeness : (P, K, m)      — politeness BEFORE the compliance gate
  n_scenarios, K, m, reward_names, model (metadata)

fintech_metadata.json           — per-scenario type/persona/turns
fintech_sample_rollouts.json    — 50 random sample rollouts as text (inspect quality)
fintech_summary.json            — aggregate compliance/politeness/action rates
```

## Quick load

```python
import numpy as np
from huggingface_hub import hf_hub_download

p = hf_hub_download(
    repo_id="eagle0504/multireward-grpo-fintech-customer-comms",
    repo_type="dataset",
    filename="fintech_rewards.npz",
)
z = np.load(p, allow_pickle=True)
rewards = z["rewards"]            # (P, K, m, 3)
raw_pol = z["raw_politeness"]     # (P, K, m)
```

## Provenance

Generated on a single NVIDIA H100 PCIe (80 GB) on RunPod Secure Cloud,
using the open-access Qwen2.5-7B-Instruct model. No real customer data,
no real bank, no real personally-identifying information. All conversation
content is synthetic.

## License

CC-BY-4.0.
