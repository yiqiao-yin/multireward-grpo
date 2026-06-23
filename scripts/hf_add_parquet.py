"""
Add a Parquet view to an existing HF dataset (alongside the .npz/.json/.png
files already there). HuggingFace's UI renders the first Parquet/CSV file
it finds — adding one unlocks the tabular preview without removing any
existing assets.

The Parquet is a long-format flattening of the (P, K, m, R) reward tensor:
one row per rollout, with columns for prompt_idx, seed_idx, rollout_idx,
plus each reward channel. This is the format reviewers expect to scroll
through.

Usage:
    set -a; source .env; set +a

    # GSM8K Qwen-1.5B
    uv run scripts/hf_add_parquet.py \\
        --npz figures/gsm8k_runpod_qwen2.5-1.5b/llm_rewards_gsm8k.npz \\
        --kind gsm8k \\
        --model-label "Qwen2.5-1.5B-Instruct" \\
        --dataset-repo eagle0504/multireward-grpo-gsm8k-rewards

    # GSM8K Qwen-7B
    uv run scripts/hf_add_parquet.py \\
        --npz figures/gsm8k_runpod/llm_rewards_gsm8k.npz \\
        --kind gsm8k \\
        --model-label "Qwen2.5-7B-Instruct" \\
        --dataset-repo eagle0504/multireward-grpo-gsm8k-rewards-qwen2.5-7b

    # Fintech
    uv run scripts/hf_add_parquet.py \\
        --npz figures/fintech_runpod/fintech_rewards.npz \\
        --metadata-json figures/fintech_runpod/fintech_metadata.json \\
        --kind fintech \\
        --model-label "Qwen2.5-7B-Instruct" \\
        --dataset-repo eagle0504/multireward-grpo-fintech-customer-comms
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi

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


# ---------------------------------------------------------------------------
# GSM8K flattening: takes (P, K, m, R) at the largest m value and emits one
# row per rollout
# ---------------------------------------------------------------------------
def flatten_gsm8k(npz_path: Path, model_label: str) -> pd.DataFrame:
    z = np.load(npz_path, allow_pickle=True)
    m_grid = list(z["m_grid"].tolist())
    m = max(m_grid)
    rewards = z[f"rewards_m{m}"]            # (P, K, m, R)
    raw_length = z[f"raw_length_m{m}"]      # (P, K, m)
    P, K, m_, R = rewards.shape
    assert m_ == m

    reward_names = [str(x) for x in z["reward_names"].tolist()]
    rows = []
    for p in range(P):
        for k in range(K):
            for j in range(m):
                row = {
                    "model": model_label,
                    "prompt_idx": int(p),
                    "seed_idx": int(k),
                    "rollout_idx": int(j),
                    "group_size_m_max": int(m),
                }
                for r_idx, name in enumerate(reward_names):
                    row[name] = float(rewards[p, k, j, r_idx])
                row["raw_length"] = float(raw_length[p, k, j])
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fintech flattening: (P, K, m, R=3) + metadata (per-scenario info)
# ---------------------------------------------------------------------------
def flatten_fintech(npz_path: Path, metadata_json: Path,
                    model_label: str,
                    generations_jsonl: Path | None = None) -> pd.DataFrame:
    z = np.load(npz_path, allow_pickle=True)
    rewards = z["rewards"]            # (P, K, m, R=3)
    raw_length = z["raw_length"]      # (P, K, m)
    raw_pol = z["raw_politeness"]     # (P, K, m)
    P, K, m, R = rewards.shape
    reward_names = [str(x) for x in z["reward_names"].tolist()]

    meta = json.loads(metadata_json.read_text())
    scenarios = {s["scenario_id"]: s for s in meta["scenarios"]}
    scenario_ids = sorted(scenarios.keys())[:P]

    # load per-rollout generations if available, key by (scenario_id, k, j)
    completions: dict[tuple[str, int, int], str] = {}
    if generations_jsonl is not None and generations_jsonl.exists():
        with open(generations_jsonl) as f:
            for line in f:
                e = json.loads(line)
                completions[(e["scenario_id"], int(e["seed_idx"]),
                            int(e["rollout_idx"]))] = e["completion"]
        print(f"  loaded {len(completions)} per-rollout generations from "
              f"{generations_jsonl}", flush=True)

    rows = []
    for p, sid in enumerate(scenario_ids):
        s = scenarios[sid]
        for k in range(K):
            for j in range(m):
                row = {
                    "model": model_label,
                    "scenario_id": sid,
                    "scenario_type": s["scenario_type"],
                    "persona": s["persona"],
                    "name": s["name"],
                    "amount": s["amount"],
                    "due_date": s["due_date"],
                    "seed_idx": int(k),
                    "rollout_idx": int(j),
                    "group_size_m": int(m),
                }
                turns = s.get("turns", [])
                row["conversation_so_far"] = " | ".join(
                    f"{role}: {text}" for role, text in turns
                )
                # the generated bot reply (None if regenerated dataset doesn't
                # ship this — fall back gracefully)
                row["bot_reply"] = completions.get((sid, k, j), "")
                for r_idx, name in enumerate(reward_names):
                    row[name] = float(rewards[p, k, j, r_idx])
                row["raw_length"] = float(raw_length[p, k, j])
                row["raw_politeness"] = float(raw_pol[p, k, j])
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--metadata-json", type=Path, default=None,
                    help="required for --kind fintech")
    ap.add_argument("--generations-jsonl", type=Path, default=None,
                    help="(fintech) JSONL with full per-rollout generations")
    ap.add_argument("--kind", choices=["gsm8k", "fintech"], required=True)
    ap.add_argument("--model-label", type=str, required=True)
    ap.add_argument("--dataset-repo", type=str, required=True,
                    help="full HF repo id, e.g. eagle0504/foo")
    ap.add_argument("--parquet-name", type=str, default="data/train.parquet",
                    help="path within the HF dataset repo")
    args = ap.parse_args()

    if not args.npz.exists():
        sys.exit(f"npz not found: {args.npz}")
    if args.kind == "fintech" and (args.metadata_json is None or
                                    not args.metadata_json.exists()):
        sys.exit("--metadata-json required for fintech kind")

    env = load_env()
    token = env.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set")

    # flatten
    if args.kind == "gsm8k":
        df = flatten_gsm8k(args.npz, args.model_label)
    else:
        df = flatten_fintech(args.npz, args.metadata_json, args.model_label,
                             generations_jsonl=args.generations_jsonl)
    print(f"flattened {len(df):,} rows, columns: {list(df.columns)}", flush=True)

    # write to a temp parquet
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp = Path(f.name)
    df.to_parquet(tmp, index=False, compression="snappy")
    size_mb = tmp.stat().st_size / 1e6
    print(f"  wrote {tmp} ({size_mb:.2f} MB)", flush=True)

    # upload
    api = HfApi(token=token)
    print(f"  uploading to {args.dataset_repo}:{args.parquet_name} ...", flush=True)
    api.upload_file(
        path_or_fileobj=str(tmp),
        path_in_repo=args.parquet_name,
        repo_id=args.dataset_repo,
        repo_type="dataset",
        commit_message=f"Add Parquet view ({args.kind}, {args.model_label})",
    )
    url = (f"https://huggingface.co/datasets/{args.dataset_repo}/blob/main/"
           f"{args.parquet_name}")
    print(f"\n=== DONE ===")
    print(f"  {url}")
    print(f"  HF UI will now render a tabular preview at the top of "
          f"https://huggingface.co/datasets/{args.dataset_repo}")
    tmp.unlink()


if __name__ == "__main__":
    main()
