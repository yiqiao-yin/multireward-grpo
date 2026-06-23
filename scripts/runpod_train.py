"""
Orchestrator: spawn an H100 pod, run NA-GRPO training (and optionally an AN
baseline for comparison), push trained model adapters to HuggingFace Hub,
terminate.

Usage:
    uv run scripts/runpod_train.py --mode na --n-steps 200 \\
        --model Qwen/Qwen2.5-1.5B-Instruct

    # also run AN baseline on the same pod (back-to-back):
    uv run scripts/runpod_train.py --mode both
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runpod_lib import run_pod_experiment, REPO_ROOT


REMOTE_BOOTSTRAP_SINGLE = """\
set -euo pipefail
echo '[remote] uname:'; uname -a
echo '[remote] nvidia-smi:'; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd /workspace/conditioned-multireward

if ! command -v uv >/dev/null 2>&1; then
  echo '[remote] installing uv ...'
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo '[remote] uv sync --extra llm ...'
uv sync --extra llm

echo '[remote] starting GRPO training (mode={mode}) ...'
mkdir -p figures/grpo_train
uv run scripts/grpo_train.py \\
    --mode {mode} \\
    --model {model} \\
    --n-scenarios {n_scenarios} \\
    --n-steps {n_steps} \\
    --batch-prompts {batch_prompts} \\
    --m {m} \\
    --max-new-tokens {max_new_tokens} \\
    --temperature {temperature} \\
    --lr {lr} \\
    --kl-coef {kl_coef} \\
    --out-dir figures/grpo_train \\
    --checkpoint-every {checkpoint_every} \\
    --seed {seed}

echo '[remote] pushing trained model to HuggingFace Hub ...'
HF_MODE={mode} uv run scripts/hf_push_model.py

echo '[remote] DONE'
ls -la figures/grpo_train/
"""

REMOTE_BOOTSTRAP_BOTH = """\
set -euo pipefail
echo '[remote] uname:'; uname -a
echo '[remote] nvidia-smi:'; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd /workspace/conditioned-multireward

if ! command -v uv >/dev/null 2>&1; then
  echo '[remote] installing uv ...'
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo '[remote] uv sync --extra llm ...'
uv sync --extra llm

mkdir -p figures/grpo_train

# === run NA training ===
echo '[remote] starting GRPO training (mode=na) ...'
uv run scripts/grpo_train.py \\
    --mode na \\
    --model {model} \\
    --n-scenarios {n_scenarios} \\
    --n-steps {n_steps} \\
    --batch-prompts {batch_prompts} \\
    --m {m} \\
    --max-new-tokens {max_new_tokens} \\
    --temperature {temperature} \\
    --lr {lr} \\
    --kl-coef {kl_coef} \\
    --out-dir figures/grpo_train \\
    --checkpoint-every {checkpoint_every} \\
    --seed {seed}

# === run AN baseline ===
echo '[remote] starting GRPO training (mode=an, baseline) ...'
uv run scripts/grpo_train.py \\
    --mode an \\
    --model {model} \\
    --n-scenarios {n_scenarios} \\
    --n-steps {n_steps} \\
    --batch-prompts {batch_prompts} \\
    --m {m} \\
    --max-new-tokens {max_new_tokens} \\
    --temperature {temperature} \\
    --lr {lr} \\
    --kl-coef {kl_coef} \\
    --out-dir figures/grpo_train \\
    --checkpoint-every {checkpoint_every} \\
    --seed {seed}

echo '[remote] pushing both trained models to HuggingFace Hub ...'
HF_MODE=na uv run scripts/hf_push_model.py
HF_MODE=an uv run scripts/hf_push_model.py

echo '[remote] DONE'
ls -la figures/grpo_train/
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["na", "an", "both"], default="na")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-scenarios", type=int, default=400)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--batch-prompts", type=int, default=4)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wall-clock-cap", type=int, default=21600,
                    help="seconds; default 6h (covers both modes)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tpl = REMOTE_BOOTSTRAP_BOTH if args.mode == "both" else REMOTE_BOOTSTRAP_SINGLE
    remote = tpl.format(
        mode=args.mode,
        model=args.model,
        n_scenarios=args.n_scenarios,
        n_steps=args.n_steps,
        batch_prompts=args.batch_prompts,
        m=args.m,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        lr=args.lr,
        kl_coef=args.kl_coef,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"mode: {args.mode}")
        print(f"model: {args.model}")
        print(f"steps: {args.n_steps}, P={args.batch_prompts}, m={args.m}")
        print(f"REMOTE SCRIPT (first 250 chars):")
        print(remote[:250] + "...")
        return

    result = run_pod_experiment(
        name=f"grpo-{args.mode}",
        remote_script=remote,
        # pull back small artifacts (metrics, training curves); model
        # checkpoints stay on HF (uploaded by hf_push_model.py)
        pull_glob="figures/grpo_train/*",
        local_out=REPO_ROOT / "figures" / "grpo_train",
        wall_clock_cap=args.wall_clock_cap,
        container_disk_gb=100,  # need extra room for model checkpoints
        needs_hf=True,
    )
    print(f"\n=== DONE ===")
    print(f"  result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
