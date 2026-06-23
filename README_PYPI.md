# multireward-grpo

**Decoupled & conditioned multi-reward GRPO** — advantage estimators, a
generalized trainer, and the Theorem-3 verification harness from the paper
*"When and Why Decoupling and Conditioning Beat Reweighting in Multi-Reward
GRPO: A U-Statistic Treatment."*

This package modularizes the experiment code so you can train your own
multi-reward GRPO models, verify the correlation-aware MSE law on your own
rollouts, and (optionally) run the whole thing on a cloud GPU.

```bash
pip install multireward-grpo            # core (numpy/scipy): advantage + analysis
pip install "multireward-grpo[llm]"     # + torch/transformers/peft: training & real generation
pip install "multireward-grpo[viz]"     # + matplotlib: the money-plot figure
pip install "multireward-grpo[data]"    # + datasets/hf-hub: dataset loaders & model push
pip install "multireward-grpo[runpod]"  # + requests: cloud GPU orchestration
```

## The two orderings

Given a group of `m` rollouts each scored on `R` reward channels with weights `w`:

- **AN — Aggregate-then-Normalize** (classic GRPO baseline): scalarize `s = wᵀr`,
  then group-normalize. The high-variance channel dominates (Prop 1) and the
  advantage resolution collapses under heterogeneous scales (Prop 2).
- **NA — Normalize-then-Aggregate** (the decoupled estimator, = MO-GRPO/GDPO):
  group-normalize each channel, then take the weighted sum. Restores
  weight-proportional influence and gives the correlation-aware gradient-MSE
  floor `(τ²/m)·wᵀCw` (Theorem 3).

```python
import numpy as np
from multireward_grpo import compute_advantage

rewards = np.array([[1.0, 0.3, 1.0],   # (m=4 rollouts, R=3 channels)
                    [0.0, 0.9, 1.0],
                    [1.0, 0.1, 0.0],
                    [0.0, 0.5, 1.0]])
w = np.array([1.0, 1.0, 0.5])
A_na = compute_advantage(rewards, w, mode="na")   # recommended
A_an = compute_advantage(rewards, w, mode="an")   # GRPO baseline
```

## Train your own model

Bring **your own prompts** and **your own reward function**; the trainer runs
group-relative policy optimization with a KL anchor and saves a LoRA adapter.

```python
from multireward_grpo import GRPOConfig, GRPOTrainer

# prompts: list of strings, chat-message lists, or dicts with metadata
prompts = ["Write a polite refusal to a refund demand.", ...]

# reward_fn(completion, prompt) -> R channel scores (len == len(weights))
def reward_fn(completion, prompt):
    return (compliance(completion), politeness(completion), action(completion))

cfg = GRPOConfig(model="Qwen/Qwen2.5-1.5B-Instruct",
                 mode="na", weights=(1.0, 1.0, 0.5), n_steps=200, m=8)
history = GRPOTrainer(cfg, reward_fn, prompts).train()
```

### Data format

| Input | Shape / type | Notes |
|---|---|---|
| `prompts` | `list[str \| list[dict] \| dict]` | a string (user msg), chat messages `[{"role","content"}]`, or `{"prompt": ..., "gold": ...}` with metadata passed through to the reward fn |
| `reward_fn(completion, prompt)` | returns `Sequence[float]` of length `R` | one score per reward channel; **channel 0 is the gate** for conditioning |
| `weights` | `tuple[float, ...]` length `R` | objective weights `w` |
| `mode` | `"na" \| "an" \| "single"` | `na` is the paper's recommendation |

Reward tensors for the analysis tools use shape **`(P, K, m, R)`** = prompts ×
seeds × rollouts × reward channels.

### Ready-made examples

```python
from multireward_grpo.examples import FintechRewardFunction, make_fintech_prompts
from multireward_grpo import GRPOConfig, GRPOTrainer

prompts = make_fintech_prompts(400, seed=0)
cfg = GRPOConfig(mode="na", weights=(1.0, 1.0, 0.5))
GRPOTrainer(cfg, FintechRewardFunction(), prompts).train()
```

`multireward_grpo.examples.gsm8k` provides GSM8K loaders paired with
`multireward_grpo.rewards.MathRewardFunction` (correctness / length / format).

## Verify Theorem 3 on your rollouts

```python
from multireward_grpo import analyze, summary_print
from multireward_grpo.generation import MockBackend, run_corpus, pack_for_analysis
import numpy as np

C = np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])   # reward correlation
corpus = run_corpus(MockBackend(C=C), [(f"p{i}", "0") for i in range(40)],
                    m_grid=[8], K_seeds=200)
rewards = pack_for_analysis(corpus, m=8)               # (P, K, m, R)
result = analyze(rewards, w=np.array([1.0, 1.0, 0.5]))
summary_print(result)
```

Or from the shell:

```bash
multireward-grpo thm3-check --rho 0.5      # CPU, no GPU
multireward-grpo train --mode na --n-steps 50   # needs [llm] + GPU
```

## Run on a cloud GPU (RunPod)

```python
from multireward_grpo.runpod import RunPodClient
client = RunPodClient()  # reads RUNPOD_API_KEY from env or .env
client.run_command('pip install "multireward-grpo[llm]" && multireward-grpo train --mode na',
                   wall_clock_cap=1800)
```

## Released artifacts (Hugging Face)

Datasets and fine-tuned models from the paper live under the
[`eagle0504`](https://huggingface.co/eagle0504) namespace:

**Datasets**
- [multireward-grpo-gsm8k-rewards](https://huggingface.co/datasets/eagle0504/multireward-grpo-gsm8k-rewards) — 76,800 Qwen2.5-1.5B GSM8K rollouts (rewards + chains-of-thought)
- [multireward-grpo-gsm8k-rewards-qwen2.5-7b](https://huggingface.co/datasets/eagle0504/multireward-grpo-gsm8k-rewards-qwen2.5-7b) — 25,600 Qwen2.5-7B rollouts
- [multireward-grpo-fintech-customer-comms](https://huggingface.co/datasets/eagle0504/multireward-grpo-fintech-customer-comms) — 2,400 fintech conversations

**Models** (LoRA adapters for Qwen2.5-1.5B-Instruct)
- [multireward-grpo-fintech-na-qwen2.5-1.5b](https://huggingface.co/eagle0504/multireward-grpo-fintech-na-qwen2.5-1.5b) — NA (paper's recommendation)
- [multireward-grpo-fintech-an-qwen2.5-1.5b](https://huggingface.co/eagle0504/multireward-grpo-fintech-an-qwen2.5-1.5b) — AN baseline
- [multireward-grpo-fintech-single-qwen2.5-1.5b](https://huggingface.co/eagle0504/multireward-grpo-fintech-single-qwen2.5-1.5b) — single-reward ablation

## Citation

If you use this package, please cite the paper (see the
[GitHub repository](https://github.com/yiqiao-yin/multireward-grpo) for the
current BibTeX entry).

## License

MIT
