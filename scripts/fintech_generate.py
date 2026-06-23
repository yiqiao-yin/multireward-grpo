"""
Generate the synthetic fintech customer-communication dataset.

For each randomized scenario (system prompt + 1–2 turn history ending with
a user message), sample m next-bot-turn rollouts from Qwen2.5-7B-Instruct,
score each on (compliance, politeness_gated, action, raw_length,
raw_politeness), save in npz format compatible with the Thm 3 + Prop 4
analysis tools.

This script is the fintech analog of `llm_validate.py` — but it generates a
*publishable corpus* (uploaded to HF afterwards), not just a measurement
sweep. Output:

    figures/fintech_runpod/
        fintech_rewards.npz       — main (P, K, m, R) reward tensor + raw_*
        fintech_metadata.json     — scenario metadata indexed to the tensor
        fintech_sample_rollouts.json — 50 random sample rollouts for inspection
        fintech_summary.json      — aggregate statistics

Usage (on a pod with GPU):
    uv run scripts/fintech_generate.py --n-scenarios 500 --K 1 --m 8 \\
        --model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Allow running from project root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fintech_scenarios import make_scenarios, Scenario
from fintech_rewards import FintechRewardConfig, score_response

REPO_ROOT = Path(__file__).resolve().parent.parent


def _format_chat(scenario: Scenario, tokenizer) -> str:
    """Apply the model's chat template to the scenario, leaving the
    assistant turn open for generation."""
    messages = [{"role": "system", "content": scenario.system_prompt}]
    for role, text in scenario.turns:
        # our internal roles: 'bot' -> 'assistant', 'user' -> 'user'
        m_role = "assistant" if role == "bot" else "user"
        messages.append({"role": m_role, "content": text})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenarios", type=int, default=500)
    ap.add_argument("--K", type=int, default=1,
                    help="seeds per scenario (we just need m rollouts per "
                         "scenario for a publishable dataset; K=1 is enough)")
    ap.add_argument("--m", type=int, default=8,
                    help="rollouts per scenario (the m in Thm 3 setup)")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--out-dir", type=str, default="figures/fintech_runpod")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-rollouts", type=int, default=50,
                    help="how many rollouts to save as text for HF inspection")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== fintech generation: {args.n_scenarios} scenarios × "
          f"K={args.K} × m={args.m} on {args.model} ===")
    print(f"  out_dir: {out_dir}")

    # ----- model load -----
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    print(f"  loading tokenizer + model ({args.model}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.0f}s")

    # ----- generation loop -----
    P = args.n_scenarios
    K = args.K
    m = args.m
    R = 3  # (compliance, politeness_gated, action)
    rewards_tensor = np.zeros((P, K, m, R), dtype=float)
    raw_length_tensor = np.zeros((P, K, m), dtype=float)
    raw_politeness_tensor = np.zeros((P, K, m), dtype=float)
    metadata_rows: list[dict] = []
    sample_rollouts: list[dict] = []
    # full per-rollout generation log: written to a JSONL so it's HF-friendly
    full_generations_path = out_dir / "fintech_generations.jsonl"
    full_generations_file = open(full_generations_path, "w")
    sample_indices = set(np.random.default_rng(args.seed).choice(
        P, size=min(args.sample_rollouts, P), replace=False
    ))

    scenarios = list(make_scenarios(P, seed=args.seed))
    t_start = time.time()
    config = FintechRewardConfig()
    tokenize_fn = lambda s: len(tokenizer.encode(s, add_special_tokens=False))

    for p_idx, scenario in enumerate(scenarios):
        prompt = _format_chat(scenario, tokenizer)
        metadata_rows.append({
            "scenario_id": scenario.scenario_id,
            "scenario_type": scenario.scenario_type,
            "persona": scenario.persona,
            "name": scenario.name,
            "amount": scenario.amount,
            "due_date": scenario.due_date,
            "turns": scenario.turns,
        })

        for k in range(K):
            # generate m rollouts in one batch
            inputs = tokenizer(
                [prompt] * m, return_tensors="pt", padding=True
            ).to("cuda")
            seed = args.seed * 1009 + p_idx * 7919 + k
            torch.manual_seed(seed)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
            in_len = inputs.input_ids.shape[1]
            completions = tokenizer.batch_decode(
                out[:, in_len:], skip_special_tokens=True
            )

            # score + log every rollout (text + rewards) to the JSONL
            for j, gen in enumerate(completions):
                c, pol, act, raw_l, raw_pol = score_response(
                    gen, config, tokenize=tokenize_fn,
                )
                rewards_tensor[p_idx, k, j] = (c, pol, act)
                raw_length_tensor[p_idx, k, j] = raw_l
                raw_politeness_tensor[p_idx, k, j] = raw_pol
                full_generations_file.write(json.dumps({
                    "scenario_id": scenario.scenario_id,
                    "seed_idx": int(k),
                    "rollout_idx": int(j),
                    "completion": gen,
                    "compliance": c,
                    "politeness_gated": pol,
                    "action": act,
                    "raw_length": raw_l,
                    "raw_politeness": raw_pol,
                }) + "\n")
                full_generations_file.flush()

            if p_idx in sample_indices and k == 0:
                sample_rollouts.append({
                    "scenario_id": scenario.scenario_id,
                    "scenario_type": scenario.scenario_type,
                    "persona": scenario.persona,
                    "turns": scenario.turns,
                    "completions": completions,
                    "rewards": rewards_tensor[p_idx, k].tolist(),
                    "raw_length": raw_length_tensor[p_idx, k].tolist(),
                    "raw_politeness": raw_politeness_tensor[p_idx, k].tolist(),
                })

        if (p_idx + 1) % 25 == 0 or p_idx == P - 1:
            elapsed = time.time() - t_start
            rate = (p_idx + 1) / max(elapsed, 1e-6)
            eta = (P - p_idx - 1) / max(rate, 1e-6)
            print(f"  [{p_idx+1}/{P}] elapsed={elapsed/60:.1f}min rate={rate*60:.1f}/min  ETA={eta/60:.1f}min")

    # ----- save artifacts -----
    npz_path = out_dir / "fintech_rewards.npz"
    np.savez_compressed(
        npz_path,
        rewards=rewards_tensor,
        raw_length=raw_length_tensor,
        raw_politeness=raw_politeness_tensor,
        n_scenarios=np.array(P, dtype=int),
        K=np.array(K, dtype=int),
        m=np.array(m, dtype=int),
        reward_names=np.array(["compliance", "politeness_gated", "action"]),
        model=np.array(args.model),
    )
    print(f"  saved {npz_path}")

    meta_path = out_dir / "fintech_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "n_scenarios": P, "K": K, "m": m,
            "model": args.model, "seed": args.seed,
            "scenarios": metadata_rows,
        }, f, indent=2)
    print(f"  saved {meta_path}")

    sample_path = out_dir / "fintech_sample_rollouts.json"
    with open(sample_path, "w") as f:
        json.dump(sample_rollouts, f, indent=2)
    print(f"  saved {sample_path}")

    # aggregate stats
    summary = {
        "n_scenarios": P, "K": K, "m": m,
        "model": args.model,
        "mean_compliance": float(rewards_tensor[..., 0].mean()),
        "mean_politeness_gated": float(rewards_tensor[..., 1].mean()),
        "mean_action": float(rewards_tensor[..., 2].mean()),
        "mean_raw_length": float(raw_length_tensor.mean()),
        "mean_raw_politeness": float(raw_politeness_tensor.mean()),
        "compliance_failures_by_scenario_type": {},
    }
    # compliance failure rate by scenario type
    for st in set(r["scenario_type"] for r in metadata_rows):
        idxs = [i for i, r in enumerate(metadata_rows) if r["scenario_type"] == st]
        if idxs:
            sub = rewards_tensor[idxs, :, :, 0]
            summary["compliance_failures_by_scenario_type"][st] = {
                "n": len(idxs),
                "mean_compliance": float(sub.mean()),
            }
    summary_path = out_dir / "fintech_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved {summary_path}")

    full_generations_file.close()
    print(f"  saved full per-rollout generations: {full_generations_path}")

    total = time.time() - t_start
    print(f"\n=== DONE in {total/60:.1f} min ===")
    print(f"  overall: mean compliance={summary['mean_compliance']:.3f}, "
          f"politeness={summary['mean_politeness_gated']:.3f}, "
          f"action={summary['mean_action']:.3f}")


if __name__ == "__main__":
    main()
