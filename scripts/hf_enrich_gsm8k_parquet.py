"""
Re-load the original GSM8K test prompts (deterministic from
`openai/gsm8k` test split + shuffle(seed=0) — matches our `load_gsm8k()`)
and join them onto the existing parquet by `prompt_idx`. Upload the
enriched parquet, replacing the old one. No GPU needed.

Result: each row now has `question` and `gold_answer` columns alongside
the rewards.

Usage:
    set -a; source .env; set +a
    uv run scripts/hf_enrich_gsm8k_parquet.py \\
        --dataset-repo eagle0504/multireward-grpo-gsm8k-rewards \\
        --n-prompts 150
    uv run scripts/hf_enrich_gsm8k_parquet.py \\
        --dataset-repo eagle0504/multireward-grpo-gsm8k-rewards-qwen2.5-7b \\
        --n-prompts 100
"""
from __future__ import annotations
import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> dict:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-repo", required=True)
    ap.add_argument("--n-prompts", type=int, required=True,
                    help="must match what load_gsm8k() pulled at experiment time")
    ap.add_argument("--parquet-name", default="data/train.parquet")
    args = ap.parse_args()

    env = load_env()
    token = env.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set")

    # 1. fetch existing parquet
    print(f"  fetching existing parquet from {args.dataset_repo} ...")
    p = hf_hub_download(
        repo_id=args.dataset_repo,
        repo_type="dataset",
        filename=args.parquet_name,
        token=token,
    )
    df = pd.read_parquet(p)
    print(f"  loaded: {len(df):,} rows, cols={list(df.columns)}")

    # 2. reconstruct GSM8K prompts with the same shuffle seed used at experiment time
    print(f"  loading openai/gsm8k test split (with shuffle seed=0) ...")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=0).select(range(args.n_prompts))
    prompt_table = pd.DataFrame({
        "prompt_idx": list(range(args.n_prompts)),
        "question": [ex["question"] for ex in ds],
        "gold_answer": [ex["answer"].split("####")[-1].strip().replace(",", "")
                        for ex in ds],
        "gold_reasoning": [ex["answer"].split("####")[0].strip() for ex in ds],
    })

    # 3. join
    enriched = df.merge(prompt_table, on="prompt_idx", how="left")
    print(f"  enriched: {len(enriched):,} rows, cols={list(enriched.columns)}")
    assert enriched["question"].isna().sum() == 0, "join failed somewhere"

    # 4. write + upload, replacing the old parquet
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp = Path(f.name)
    enriched.to_parquet(tmp, index=False, compression="snappy")
    print(f"  wrote {tmp.stat().st_size/1e6:.2f} MB")

    api = HfApi(token=token)
    print(f"  uploading to {args.dataset_repo}:{args.parquet_name} ...")
    api.upload_file(
        path_or_fileobj=str(tmp),
        path_in_repo=args.parquet_name,
        repo_id=args.dataset_repo,
        repo_type="dataset",
        commit_message="Enrich parquet with GSM8K question + gold_answer columns",
    )
    tmp.unlink()
    print(f"\n=== DONE ===")
    print(f"  https://huggingface.co/datasets/{args.dataset_repo}")


if __name__ == "__main__":
    main()
