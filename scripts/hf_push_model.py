"""
Push a trained GRPO model (LoRA adapter) to HuggingFace Hub.

Runs ON the pod after grpo_train.py completes. Reads HF_MODE (na or an)
from the environment to pick the right local checkpoint folder and HF
repo name.

Usage (from runpod_train.py's REMOTE_BOOTSTRAP):
    HF_MODE=na uv run scripts/hf_push_model.py
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, whoami

REPO_ROOT = Path(__file__).resolve().parent.parent


MODEL_CARD_TEMPLATE = """\
---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
language:
- en
library_name: peft
pipeline_tag: text-generation
tags:
- grpo
- multi-reward
- reinforcement-learning
- fintech
- customer-service
- {mode}-advantage
---

# Multi-Reward GRPO — {mode_upper} Advantage — Qwen2.5-1.5B on Fintech Customer Comms

LoRA adapter trained with **{mode_upper}** advantage formulation from
**"Conditioned Multi-Reward Advantage Estimation: A Finite-Sample Analysis"**.

## Advantage formulation

- **{mode_upper}** = {mode_long}

This is the difference vs the AN baseline:

| | AN (baseline) | NA (paper's recommendation) |
|---|---|---|
| Step 1 | aggregate channels: `s_j = w^T r_j` | per-channel normalize: `z_jl = (r_jl - mean_l) / std_l` |
| Step 2 | group-normalize: `A_j = (s_j - mean) / std` | aggregate: `A_j = sum_l w_l z_jl` |
| Influence | dominated by high-σ channel (Prop 1) | weight-proportional (Prop 1) |
| MSE floor | sensitive to scalarization | `(τ²/m) w^T C w` (Thm 3) |

## Training data

`huggingface.co/datasets/eagle0504/multireward-grpo-fintech-customer-comms`
— synthetic fintech customer-service conversations, 300 scenarios, with
reward channels:
- `compliance` (binary): the harder gate
- `politeness_gated` (continuous): gated by compliance
- `action` (binary): clear next-step indicator

## How to use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-1.5B-Instruct"
adapter = "eagle0504/multireward-grpo-fintech-{mode}-qwen2.5-1.5b"

tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16")
model = PeftModel.from_pretrained(model, adapter)
model.eval()
# ... generate ...
```

## Hyperparameters (from training)

See `metrics.json` in the repo for the full training trajectory (loss,
PG loss, KL, per-step reward).

## Citation

```bibtex
@misc{{yin2026multireward,
  title={{Conditioned Multi-Reward Advantage Estimation: A Finite-Sample Analysis}},
  author={{Yin, Yiqiao}},
  year={{2026}},
}}
```

## License

Apache-2.0 (matches the Qwen base model).
"""

MODE_DESCRIPTIONS = {
    "na":     "Normalize-then-Aggregate: per-channel group-normalize, then weighted sum",
    "an":     "Aggregate-then-Normalize: weighted sum then group-normalize (standard GRPO baseline)",
    "single": "Single-reward GRPO: ignore the other channels, train on correctness only (multi-reward-ablation baseline)",
}


def find_latest_checkpoint(mode: str) -> Path:
    """Look first for the new layout `{mode}_seed{N}/checkpoint_step*` (we
    canonicalize on seed 0); fall back to the legacy `{mode}/checkpoint_step*`
    layout for older runs."""
    candidates = [
        REPO_ROOT / "figures" / "grpo_train" / f"{mode}_seed0",
        REPO_ROOT / "figures" / "grpo_train" / mode,
    ]
    for train_dir in candidates:
        if not train_dir.exists():
            continue
        ckpts = sorted(train_dir.glob("checkpoint_step*"))
        if ckpts:
            return ckpts[-1]
    sys.exit(f"no checkpoints for mode={mode!r} in any of: "
             + ", ".join(str(c) for c in candidates))


def main():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set in environment")
    mode = (os.environ.get("HF_MODE") or "na").lower()
    if mode not in ("na", "an", "single"):
        sys.exit(f"HF_MODE must be 'na', 'an', or 'single', got {mode!r}")

    api = HfApi(token=token)
    me = whoami(token=token)
    user = me["name"]
    repo_id = f"{user}/multireward-grpo-fintech-{mode}-qwen2.5-1.5b"
    print(f"target repo: huggingface.co/{repo_id}", flush=True)

    ckpt = find_latest_checkpoint(mode)
    print(f"  source checkpoint: {ckpt}", flush=True)

    # metrics.json sits alongside the checkpoint folder
    metrics_path = ckpt.parent / "metrics.json"

    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True,
                token=token, private=False)
    print(f"  repo ready", flush=True)

    card = MODEL_CARD_TEMPLATE.format(
        mode=mode, mode_upper=mode.upper(),
        mode_long=MODE_DESCRIPTIONS[mode],
    )
    card_path = ckpt / "README.md"
    card_path.write_text(card)
    print(f"  wrote model card", flush=True)

    # upload checkpoint folder
    print(f"  uploading checkpoint folder ...", flush=True)
    api.upload_folder(
        folder_path=str(ckpt),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"GRPO ({mode.upper()} advantage) — final checkpoint",
        ignore_patterns=["*.tmp"],
    )
    # also upload metrics.json
    if metrics_path.exists():
        api.upload_file(
            path_or_fileobj=str(metrics_path),
            path_in_repo="metrics.json",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  uploaded metrics.json", flush=True)

    url = f"https://huggingface.co/{repo_id}"
    print(f"\n=== DONE ===", flush=True)
    print(f"  {url}", flush=True)

    # leave a marker file the orchestrator can pull back
    marker = REPO_ROOT / "figures" / "grpo_train" / mode / "HF_MODEL_URL.txt"
    marker.write_text(url + "\n")


if __name__ == "__main__":
    main()
