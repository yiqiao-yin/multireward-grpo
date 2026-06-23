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
- gsm8k
- variance-reduction
task_categories:
- text-generation
- reinforcement-learning
pretty_name: Multi-Reward GRPO GSM8K Rewards (Qwen2.5-7B-Instruct)
---

# Multi-Reward GRPO — GSM8K Rewards (Qwen2.5-7B-Instruct)

Raw rollout-level reward observations from the empirical Section of
**"Conditioned Multi-Reward Advantage Estimation: A Finite-Sample Analysis"**.

This is the data that produced the headline Theorem 3 (correlation-dependent
MSE floor) and Proposition 4 (sign-changing conditioning bias) figures on
real LLM rollouts. Each rollout was sampled from `Qwen/Qwen2.5-7B-Instruct` on
GSM8K test prompts at temperature 0.7.

## What's in here

For each GSM8K test prompt we sampled K independent seeds × m rollouts per
seed, scored on three verifiable reward channels:

| Channel | Type | Definition |
|---|---|---|
| `correctness` | Bernoulli | Final boxed answer matches GSM8K gold |
| `length` | Continuous on [-1, 0] | `tanh(-|log(n_tokens / 200)|)` — peaks at 200-token target |
| `format` | Bernoulli | Response contains `\boxed{...}` or `#### N` |

The raw (uncontaminated) length reward is stored separately so the Prop 4
γ-sweep is reproducible as cheap CPU post-processing.

## File layout

```
llm_rewards_gsm8k.npz             — main rewards tensor (numpy)
  rewards_m{m}     : (P, K, m, R=3)  for m in m_grid
  raw_length_m{m}  : (P, K, m)        uncontaminated length per rollout
  m_grid, K_seeds, n_prompts, reward_names, backend_name (metadata)

llm_summary_gsm8k.json            — aggregate Thm 3 numbers per m
llm_prop4_summary_gsm8k.json      — γ-sweep numbers (Prop 4 bias law), if produced

llm_mse_vs_m_gsm8k.png            — headline Theorem 3 figure
llm_money_scatter_gsm8k_m{m}.png  — per-prompt predicted-vs-realized scatters
llm_prop4_bias_law_gsm8k.png      — Proposition 4 bias law (if produced)
```

## Quick load

```python
import numpy as np
from huggingface_hub import hf_hub_download

p = hf_hub_download(
    repo_id="eagle0504/multireward-grpo-gsm8k-rewards-qwen2.5-7b",
    repo_type="dataset",
    filename="llm_rewards_gsm8k.npz",
)
z = np.load(p, allow_pickle=True)
rewards = z["rewards_m32"]    # (P, K, 32, 3)
raw_len = z["raw_length_m32"] # (P, K, 32)
```

## Provenance

Generated on NVIDIA H100 PCIe (80 GB) on RunPod Secure Cloud, using the
open-access `Qwen/Qwen2.5-7B-Instruct` model. Reproduce with:

```bash
uv run scripts/runpod_launch.py \
    --n-prompts {N} --K {K} --m-grid 4 8 16 32 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --wall-clock-cap 18000
```

## Citation

```bibtex
@misc{yin2026multireward,
  title={Conditioned Multi-Reward Advantage Estimation: A Finite-Sample Analysis},
  author={Yin, Yiqiao},
  year={2026},
}
```

## License

CC-BY-4.0.
