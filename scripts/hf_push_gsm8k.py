"""
Push GSM8K Thm 3 + Prop 4 validation artifacts to HuggingFace Hub.

Usage:
    # Qwen-1.5B (original headline run):
    uv run scripts/hf_push_gsm8k.py \\
        --src-dir figures/gsm8k_runpod_qwen2.5-1.5b \\
        --dataset-name multireward-grpo-gsm8k-rewards \\
        --model-label "Qwen2.5-1.5B-Instruct"

    # Qwen-7B (run #3):
    uv run scripts/hf_push_gsm8k.py \\
        --src-dir figures/gsm8k_runpod \\
        --dataset-name multireward-grpo-gsm8k-rewards-qwen2.5-7b \\
        --model-label "Qwen2.5-7B-Instruct"

Reads HF_TOKEN from .env or environment.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, whoami

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env():
    env = {}
    p = REPO_ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


DATASET_CARD_TEMPLATE = """\
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
pretty_name: Multi-Reward GRPO GSM8K Rewards ({model_label})
---

# Multi-Reward GRPO — GSM8K Rewards ({model_label})

Raw rollout-level reward observations from the empirical Section of
**"Conditioned Multi-Reward Advantage Estimation: A Finite-Sample Analysis"**.

This is the data that produced the headline Theorem 3 (correlation-dependent
MSE floor) and Proposition 4 (sign-changing conditioning bias) figures on
real LLM rollouts. Each rollout was sampled from `{model_full_id}` on
GSM8K test prompts at temperature 0.7.

## What's in here

For each GSM8K test prompt we sampled K independent seeds × m rollouts per
seed, scored on three verifiable reward channels:

| Channel | Type | Definition |
|---|---|---|
| `correctness` | Bernoulli | Final boxed answer matches GSM8K gold |
| `length` | Continuous on [-1, 0] | `tanh(-|log(n_tokens / 200)|)` — peaks at 200-token target |
| `format` | Bernoulli | Response contains `\\boxed{{...}}` or `#### N` |

The raw (uncontaminated) length reward is stored separately so the Prop 4
γ-sweep is reproducible as cheap CPU post-processing.

## File layout

```
llm_rewards_gsm8k.npz             — main rewards tensor (numpy)
  rewards_m{{m}}     : (P, K, m, R=3)  for m in m_grid
  raw_length_m{{m}}  : (P, K, m)        uncontaminated length per rollout
  m_grid, K_seeds, n_prompts, reward_names, backend_name (metadata)

llm_summary_gsm8k.json            — aggregate Thm 3 numbers per m
llm_prop4_summary_gsm8k.json      — γ-sweep numbers (Prop 4 bias law), if produced

llm_mse_vs_m_gsm8k.png            — headline Theorem 3 figure
llm_money_scatter_gsm8k_m{{m}}.png  — per-prompt predicted-vs-realized scatters
llm_prop4_bias_law_gsm8k.png      — Proposition 4 bias law (if produced)
```

## Quick load

```python
import numpy as np
from huggingface_hub import hf_hub_download

p = hf_hub_download(
    repo_id="{user}/{dataset_name}",
    repo_type="dataset",
    filename="llm_rewards_gsm8k.npz",
)
z = np.load(p, allow_pickle=True)
rewards = z["rewards_m32"]    # (P, K, 32, 3)
raw_len = z["raw_length_m32"] # (P, K, 32)
```

## Provenance

Generated on NVIDIA H100 PCIe (80 GB) on RunPod Secure Cloud, using the
open-access `{model_full_id}` model. Reproduce with:

```bash
uv run scripts/runpod_launch.py \\
    --n-prompts {{N}} --K {{K}} --m-grid 4 8 16 32 \\
    --model {model_full_id} \\
    --wall-clock-cap 18000
```

## Citation

```bibtex
@misc{{yin2026multireward,
  title={{Conditioned Multi-Reward Advantage Estimation: A Finite-Sample Analysis}},
  author={{Yin, Yiqiao}},
  year={{2026}},
}}
```

## License

CC-BY-4.0.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=str, default="figures/gsm8k_runpod_qwen2.5-1.5b")
    ap.add_argument("--dataset-name", type=str,
                    default="multireward-grpo-gsm8k-rewards")
    ap.add_argument("--model-label", type=str, default="Qwen2.5-1.5B-Instruct")
    ap.add_argument("--model-full-id", type=str, default=None,
                    help="Full HF model ID (default: Qwen/<model-label>)")
    args = ap.parse_args()

    src_dir = (REPO_ROOT / args.src_dir).resolve()
    if not src_dir.exists():
        sys.exit(f"src-dir not found: {src_dir}")
    model_full_id = args.model_full_id or f"Qwen/{args.model_label}"

    env = load_env()
    token = env.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set in .env or environment")
    os.environ["HF_TOKEN"] = token

    api = HfApi(token=token)
    me = whoami(token=token)
    user = me["name"]
    repo_id = f"{user}/{args.dataset_name}"
    print(f"target repo: huggingface.co/datasets/{repo_id}")

    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True,
                token=token, private=False)
    print(f"  repo ready")

    card = DATASET_CARD_TEMPLATE.format(
        model_label=args.model_label,
        model_full_id=model_full_id,
        user=user,
        dataset_name=args.dataset_name,
    )
    (src_dir / "README.md").write_text(card)
    print(f"  wrote dataset card")

    print(f"  uploading {src_dir} ...")
    api.upload_folder(
        folder_path=str(src_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Initial upload — GSM8K rewards ({args.model_label})",
        ignore_patterns=["*.tmp", "*.lock"],
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"\n=== DONE ===")
    print(f"  {url}")


if __name__ == "__main__":
    main()
