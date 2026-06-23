"""
Orchestrator: spawn an H100 pod, generate the synthetic fintech dataset,
push to HuggingFace Hub, terminate.

Usage:
    uv run scripts/runpod_fintech.py --n-scenarios 300 --m 8 \\
        --model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runpod_lib import run_pod_experiment, REPO_ROOT


REMOTE_BOOTSTRAP = """\
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

echo '[remote] generating fintech dataset ...'
mkdir -p figures/fintech_runpod
uv run scripts/fintech_generate.py \\
    --n-scenarios {n_scenarios} \\
    --K {K} \\
    --m {m} \\
    --model {model} \\
    --out-dir figures/fintech_runpod \\
    --seed {seed}

echo '[remote] pushing dataset to HuggingFace Hub ...'
uv run scripts/hf_push_fintech.py
echo '[remote] DONE'
ls -la figures/fintech_runpod/
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenarios", type=int, default=300)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wall-clock-cap", type=int, default=10800,
                    help="seconds; 3h default")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    remote = REMOTE_BOOTSTRAP.format(
        n_scenarios=args.n_scenarios,
        K=args.K,
        m=args.m,
        model=args.model,
        seed=args.seed,
    )

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"would generate {args.n_scenarios} scenarios × K={args.K} × m={args.m}")
        print(f"model: {args.model}")
        print(f"REMOTE SCRIPT (first 200 chars):")
        print(remote[:200] + "...")
        return

    result = run_pod_experiment(
        name="fintech-gen",
        remote_script=remote,
        pull_glob="figures/fintech_runpod/*",
        local_out=REPO_ROOT / "figures" / "fintech_runpod",
        wall_clock_cap=args.wall_clock_cap,
        needs_hf=True,
    )
    print(f"\n=== DONE ===")
    print(f"  result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
