"""
Orchestrator: spawn one H100 pod, run the full multi-seed training sweep
(NA × 3 seeds, AN × 3 seeds, single-reward × 1 seed) + the held-out eval
on all trained adapters plus the base model. Push canonical adapters to HF,
pull metrics back, terminate.

Why one pod for all 7+: amortizes the model download, transformers install,
and pod boot across all runs.

Expected pod-life: 7 trainings × ~27 min + eval ~10 min + setup ~10 min ≈
~3.5–4 h. Cost ≈ $9–11 at $2.39/hr H100 PCIe.

Usage:
    uv run scripts/runpod_train_multiseed.py
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runpod_lib import run_pod_experiment, REPO_ROOT


REMOTE_BOOTSTRAP = """\
set -uo pipefail   # NOTE: no -e — we want to keep going if one run fails
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

# ============== training sweeps ==============
SEEDS_NA="0 1 2"
SEEDS_AN="0 1 2"
SEEDS_SINGLE="0"

run_one() {{
    local mode="$1" seed="$2"
    echo "[remote] === training mode=$mode seed=$seed ==="
    uv run scripts/grpo_train.py \\
        --mode "$mode" \\
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
        --seed "$seed" \\
        2>&1 | tail -3000
    echo "[remote] === done mode=$mode seed=$seed ==="
}}

for seed in $SEEDS_NA;     do run_one na     "$seed"; done
for seed in $SEEDS_AN;     do run_one an     "$seed"; done
for seed in $SEEDS_SINGLE; do run_one single "$seed"; done

# ============== held-out eval ==============
echo '[remote] === held-out eval (base + all 7 adapters) ==='
ADAPTERS=""
LABELS=""
for mode in na an single; do
    for d in figures/grpo_train/${{mode}}_seed*/checkpoint_step{n_steps_padded}; do
        [ -d "$d" ] || continue
        ADAPTERS="$ADAPTERS $d"
        seed_name=$(basename $(dirname "$d") | sed 's/.*_seed/seed/')
        LABELS="$LABELS ${{mode}}_${{seed_name}}"
    done
done
echo "[remote] eval adapters: $ADAPTERS"
echo "[remote] eval labels:   base $LABELS"
uv run scripts/grpo_eval.py \\
    --base-model {model} \\
    --adapters $ADAPTERS \\
    --labels base $LABELS \\
    --n-scenarios {eval_scenarios} --m {eval_m} \\
    --out figures/grpo_train/eval_comparison.json

# ============== push the 3 canonical adapters to HF ==============
echo '[remote] === pushing canonical adapters to HF ==='
for mode in na an single; do
    HF_MODE=$mode HF_LATEST_OK=1 uv run scripts/hf_push_model.py || \
        echo "[remote] HF push for $mode failed (will retry locally)"
done

echo '[remote] === ALL DONE ==='
ls -la figures/grpo_train/
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-scenarios", type=int, default=400)
    ap.add_argument("--n-steps", type=int, default=500)
    ap.add_argument("--batch-prompts", type=int, default=4)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--eval-scenarios", type=int, default=80)
    ap.add_argument("--eval-m", type=int, default=8)
    ap.add_argument("--wall-clock-cap", type=int, default=21600,
                    help="seconds; default 6 h covers 7 runs + eval comfortably")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    remote = REMOTE_BOOTSTRAP.format(
        model=args.model,
        n_scenarios=args.n_scenarios,
        n_steps=args.n_steps,
        n_steps_padded=f"{args.n_steps:04d}",
        batch_prompts=args.batch_prompts,
        m=args.m,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        lr=args.lr,
        kl_coef=args.kl_coef,
        checkpoint_every=args.checkpoint_every,
        eval_scenarios=args.eval_scenarios,
        eval_m=args.eval_m,
    )

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"model: {args.model}, n_steps: {args.n_steps}, "
              f"P={args.batch_prompts}, m={args.m}")
        print(f"Will run 7 trainings (NA×3, AN×3, single×1) + held-out eval.")
        print(f"REMOTE SCRIPT (first 400 chars):")
        print(remote[:400] + "...")
        return

    result = run_pod_experiment(
        name="grpo-multiseed",
        remote_script=remote,
        pull_glob="figures/grpo_train/*",
        local_out=REPO_ROOT / "figures" / "grpo_train",
        wall_clock_cap=args.wall_clock_cap,
        container_disk_gb=120,  # 7 adapter checkpoints + base model
        needs_hf=True,
    )
    print(f"\n=== DONE ===")
    print(f"  result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
