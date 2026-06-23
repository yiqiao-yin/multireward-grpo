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
pretty_name: Multi-Reward GRPO GSM8K Rewards (Qwen2.5-1.5B)
---

# Multi-Reward GRPO — GSM8K Rewards (Qwen2.5-1.5B-Instruct)

Raw rollout-level reward observations from the empirical Section of
**"Conditioned Multi-Reward Advantage Estimation: A Finite-Sample Analysis"**.

This is the data that produced the headline Theorem 3 (correlation-dependent
MSE floor) and Proposition 4 (sign-changing conditioning bias) figures on
real LLM rollouts.

## What's in here

For each of 150 GSM8K test prompts, we sampled 16 independent seeds × 32
rollouts per seed = 512 rollouts per prompt, from `Qwen/Qwen2.5-1.5B-Instruct`
at temperature 0.7. Each rollout was scored on three verifiable reward
channels:

| Channel | Type | Definition |
|---|---|---|
| `correctness` | Bernoulli | Final boxed answer matches GSM8K gold |
| `length` | Continuous on [-1, 0] | `tanh(-|log(n_tokens / 200)|)` — peaks at 200-token target |
| `format` | Bernoulli | Response contains `\boxed{...}` or `#### N` |

The raw (uncontaminated) length reward is stored separately so the Prop 4
γ-sweep is reproducible as cheap CPU post-processing — see the
[paper](https://github.com/<TBD>) for the construction.

## File layout

```
llm_rewards_gsm8k.npz             — main rewards tensor (numpy)
  rewards_m{m}     : (P=150, K=16, m, R=3)  for m in {4, 8, 16, 32}
  raw_length_m{m}  : (P=150, K=16, m)        uncontaminated length per rollout
  m_grid, K_seeds, n_prompts, reward_names, backend_name (metadata)

llm_summary_gsm8k.json            — aggregate Thm 3 numbers per m
llm_prop4_summary_gsm8k.json      — γ-sweep numbers (Prop 4 bias law)

llm_mse_vs_m_gsm8k.png            — headline Theorem 3 figure
llm_money_scatter_gsm8k_m{m}.png  — per-prompt predicted-vs-realized scatters
llm_prop4_bias_law_gsm8k.png      — Proposition 4 bias law on real data
```

## Quick load

```python
import numpy as np
from huggingface_hub import hf_hub_download

p = hf_hub_download(
    repo_id="eagle0504/multireward-grpo-gsm8k-rewards",
    repo_type="dataset",
    filename="llm_rewards_gsm8k.npz",
)
z = np.load(p, allow_pickle=True)
rewards = z["rewards_m32"]    # (150, 16, 32, 3)
raw_len = z["raw_length_m32"] # (150, 16, 32)
```

## Headline numbers (Thm 3 verified to ±7%)

| m | Theorem 3 prediction | Realized NA | NA / pred |
|---|---|---|---|
| 4  | 0.121 | 0.124 | **1.025** |
| 8  | 0.072 | 0.067 | **0.928** |
| 16 | 0.043 | 0.041 | **0.954** |
| 32 | 0.023 | 0.022 | **0.955** |

Pooled reward correlation matrix (m=32):

```
                correctness    length      format
correctness    +0.794         -0.641      +0.276
length         -0.641         +0.842      -0.096
format         +0.276         -0.096      +0.868
```

The strongly negative `correctness × length` correlation is the key
empirical signal — it pulls `w^T C w` below its uncorrelated value and so
*lowers* the MSE floor, exactly as Theorem 3 predicts.

## Reproducibility

All experiment code is in the companion repository (TBD). The raw rewards
were produced by:

```bash
uv run scripts/runpod_launch.py \
    --n-prompts 150 --K 16 --m-grid 4 8 16 32 \
    --wall-clock-cap 18000
```

Generation hardware: 1× NVIDIA H100 PCIe (80 GB) on RunPod Secure Cloud.
Wall-clock: 185.6 min. Cost: $7.39 at $2.39/hr.

## Citation

```bibtex
@misc{yin2026multireward,
  title={Conditioned Multi-Reward Advantage Estimation: A Finite-Sample Analysis},
  author={Yin, Yiqiao},
  year={2026},
  note={Preprint in preparation}
}
```

## License

CC-BY-4.0. You may use, modify, and redistribute these rewards for research,
provided attribution.
