"""
Evaluate NA-trained and AN-trained checkpoints (and the base model) on a
held-out set of fintech scenarios. Generates m completions per scenario
and reports mean reward per channel + aggregate.

Use after grpo_train.py has produced checkpoints (or load them from HF).

Usage on a GPU pod:
    uv run scripts/grpo_eval.py \\
        --base-model Qwen/Qwen2.5-1.5B-Instruct \\
        --adapters figures/grpo_train/na/checkpoint_step0150 \\
                   figures/grpo_train/an/checkpoint_step0150 \\
        --labels base NA AN \\
        --n-scenarios 80 --m 8 \\
        --out figures/grpo_train/eval_comparison.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fintech_scenarios import make_scenarios
from fintech_rewards import FintechRewardConfig, score_response

REPO_ROOT = Path(__file__).resolve().parent.parent


def format_chat(scenario, tokenizer) -> str:
    messages = [{"role": "system", "content": scenario.system_prompt}]
    for role, text in scenario.turns:
        m_role = "assistant" if role == "bot" else "user"
        messages.append({"role": m_role, "content": text})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def eval_model(model, tokenizer, scenarios, m, max_new_tokens, temperature, label):
    """Generate m completions per scenario, score each, return per-channel
    means + aggregate."""
    import torch
    config = FintechRewardConfig()
    tokenize_fn = lambda s: len(tokenizer.encode(s, add_special_tokens=False))

    all_rewards = []  # list of (m, 3) arrays
    sample_outputs = []
    for i, sc in enumerate(scenarios):
        prompt = format_chat(sc, tokenizer)
        enc = tokenizer([prompt] * m, return_tensors="pt", padding=True).to("cuda")
        torch.manual_seed(1000 + i)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
        in_len = enc.input_ids.shape[1]
        comps = tokenizer.batch_decode(out[:, in_len:], skip_special_tokens=True)
        r_arr = np.zeros((m, 3), dtype=float)
        for j, gen in enumerate(comps):
            c, pol, act, _, _ = score_response(gen, config, tokenize=tokenize_fn)
            r_arr[j] = (c, pol, act)
        all_rewards.append(r_arr)
        if i < 5:
            sample_outputs.append({
                "scenario_type": sc.scenario_type,
                "persona": sc.persona,
                "completion": comps[0],
                "rewards": r_arr[0].tolist(),
            })
        if (i + 1) % 20 == 0 or i == len(scenarios) - 1:
            print(f"  [{label}] {i+1}/{len(scenarios)}", flush=True)

    arr = np.stack(all_rewards, axis=0)  # (P, m, 3)
    return {
        "label": label,
        "n_scenarios": len(scenarios),
        "m": m,
        "mean_compliance": float(arr[..., 0].mean()),
        "mean_politeness_gated": float(arr[..., 1].mean()),
        "mean_action": float(arr[..., 2].mean()),
        "mean_aggregate": float((arr * np.array([1.0, 1.0, 0.5])).sum(axis=-1).mean()),
        "std_aggregate": float((arr * np.array([1.0, 1.0, 0.5])).sum(axis=-1).std()),
        "sample_outputs": sample_outputs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapters", type=str, nargs="*", default=[])
    ap.add_argument("--labels", type=str, nargs="*", default=None,
                    help="labels for [base, *adapters]; defaults to ['base', *adapter-dirs]")
    ap.add_argument("--n-scenarios", type=int, default=80)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=999,
                    help="different seed from training data to get a held-out set")
    ap.add_argument("--out", type=str, default="figures/grpo_train/eval_comparison.json")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    labels = args.labels or (["base"] + [Path(a).name for a in args.adapters])
    if len(labels) != 1 + len(args.adapters):
        sys.exit(f"need {1 + len(args.adapters)} labels (got {len(labels)})")

    print(f"=== eval: {len(labels)} models on {args.n_scenarios} held-out scenarios ===")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    scenarios = list(make_scenarios(args.n_scenarios, seed=args.seed))

    results = []
    # base model first
    print(f"\n--- loading base model {args.base_model} ---")
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    base.eval()
    print(f"  loaded in {time.time()-t0:.0f}s")
    r = eval_model(base, tokenizer, scenarios, args.m, args.max_new_tokens,
                   args.temperature, labels[0])
    results.append(r)
    print(f"  {labels[0]}: aggregate={r['mean_aggregate']:.3f} "
          f"(c={r['mean_compliance']:.2f} p={r['mean_politeness_gated']:.2f} "
          f"a={r['mean_action']:.2f})")

    # each adapter
    for adapter, label in zip(args.adapters, labels[1:]):
        print(f"\n--- loading adapter {adapter} as {label!r} ---")
        # reload base each time to avoid double-applying adapters
        del base; torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="cuda",
        )
        model = PeftModel.from_pretrained(base, adapter)
        model.eval()
        r = eval_model(model, tokenizer, scenarios, args.m, args.max_new_tokens,
                       args.temperature, label)
        results.append(r)
        print(f"  {label}: aggregate={r['mean_aggregate']:.3f} "
              f"(c={r['mean_compliance']:.2f} p={r['mean_politeness_gated']:.2f} "
              f"a={r['mean_action']:.2f})")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "base_model": args.base_model,
            "adapters": args.adapters,
            "labels": labels,
            "n_scenarios": args.n_scenarios,
            "m": args.m,
            "seed": args.seed,
            "results": results,
        }, f, indent=2)
    print(f"\n=== DONE ===")
    print(f"  saved {out_path}")
    print(f"\n  comparison:")
    for r in results:
        print(f"    {r['label']:12s}: agg={r['mean_aggregate']:+.3f} ± {r['std_aggregate']:.3f}  "
              f"c={r['mean_compliance']:.3f}  p={r['mean_politeness_gated']:.3f}  "
              f"a={r['mean_action']:.3f}")


if __name__ == "__main__":
    main()
