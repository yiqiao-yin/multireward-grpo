"""
Multi-reward GRPO training on synthetic fintech scenarios, with the
**N**ormalize-then-**A**ggregate (NA) advantage formulation from §2 of the
paper as the headline configuration (`--mode na`) and a baseline AN run
selectable via `--mode an` for direct comparison.

The training loop is intentionally hand-rolled (not trl's GRPOTrainer) so
the advantage swap is one line and there are no hidden defaults. The four
GRPO ingredients we implement:

  1. **Group rollouts.** For each prompt, sample m completions from the
     current policy at temperature τ.
  2. **Per-rollout multi-channel rewards.** Score each completion on the
     fintech reward channels (compliance, politeness, action).
  3. **Advantage computation.**
       - AN: aggregate s = w^T r per rollout, then `(s - mean) / (std + ε)`.
       - NA: per-channel `(r_l - mean_l) / (std_l + ε)`, then weighted sum.
  4. **Policy gradient + KL anchor.** Standard PPO-style ratio with KL
     penalty against a frozen reference policy (the initial weights).

Output: trained checkpoint (LoRA adapter — full FT is too memory-hungry for
a quick demo run) + training metrics JSON. Push to HF as a model repo.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fintech_scenarios import make_scenarios, Scenario
from fintech_rewards import FintechRewardConfig, score_response

REPO_ROOT = Path(__file__).resolve().parent.parent


def format_chat(scenario: Scenario, tokenizer) -> str:
    messages = [{"role": "system", "content": scenario.system_prompt}]
    for role, text in scenario.turns:
        m_role = "assistant" if role == "bot" else "user"
        messages.append({"role": m_role, "content": text})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def compute_advantages(rewards_pmr: "np.ndarray", w: "np.ndarray",
                       mode: str, eps: float = 1e-4) -> "np.ndarray":
    """rewards_pmr: (P, m, R)  → advantages (P, m).

    AN     : aggregate, then group-normalize (classic GRPO baseline).
    NA     : per-channel group-normalize, then aggregate (paper's proposal).
    single : ignore all channels except the first (= correctness), AN-normalize
             — the "ignore the multi-reward structure" baseline that shows
             multi-reward shaping is non-trivial beyond just notation.
    """
    P, m, R = rewards_pmr.shape
    if mode == "an":
        s_pm = rewards_pmr @ w  # (P, m)
        mu = s_pm.mean(axis=1, keepdims=True)
        sd = s_pm.std(axis=1, ddof=0, keepdims=True)
        return (s_pm - mu) / (sd + eps)
    elif mode == "na":
        mu = rewards_pmr.mean(axis=1, keepdims=True)
        sd = rewards_pmr.std(axis=1, ddof=0, keepdims=True)
        z = (rewards_pmr - mu) / (sd + eps)  # (P, m, R)
        return z @ w  # (P, m)
    elif mode == "single":
        # use only channel 0 (the "harder" gate, e.g. correctness/compliance)
        s_pm = rewards_pmr[..., 0]  # (P, m)
        mu = s_pm.mean(axis=1, keepdims=True)
        sd = s_pm.std(axis=1, ddof=0, keepdims=True)
        return (s_pm - mu) / (sd + eps)
    else:
        raise ValueError(f"mode must be 'an', 'na', or 'single', got {mode!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["na", "an", "single"], default="na",
                    help="advantage formulation: na (paper), an (baseline), single (correctness-only baseline)")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-scenarios", type=int, default=400,
                    help="distinct scenarios to cycle through")
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--batch-prompts", type=int, default=4,
                    help="P (prompts per step)")
    ap.add_argument("--m", type=int, default=8,
                    help="rollouts per prompt (group size)")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--ppo-clip", type=float, default=0.2)
    ap.add_argument("--weights", type=float, nargs=3, default=[1.0, 1.0, 0.5],
                    help="reward weights (compliance, politeness_gated, action)")
    ap.add_argument("--out-dir", type=str, default="figures/grpo_train")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-lora", action="store_true", default=True,
                    help="Train a LoRA adapter (much faster than full FT)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    # seed-suffix the output dir so multi-seed sweeps don't collide
    out_dir = REPO_ROOT / args.out_dir / f"{args.mode}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== GRPO training: mode={args.mode}, model={args.model} ===")
    print(f"  out_dir: {out_dir}")
    print(f"  hyperparameters: steps={args.n_steps}, P={args.batch_prompts}, m={args.m}, "
          f"lr={args.lr}, kl={args.kl_coef}, weights={args.weights}")

    # ----- model + tokenizer -----
    import torch
    from torch.nn.utils import clip_grad_norm_
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType

    torch.manual_seed(args.seed)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  loading policy ({args.model}) ...")
    policy = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    if args.use_lora:
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        policy = get_peft_model(policy, lora_cfg)
        policy.print_trainable_parameters()
    policy.train()

    print(f"  loading reference (frozen) ...")
    ref_policy = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    print(f"  loaded in {time.time()-t0:.0f}s, VRAM={torch.cuda.memory_allocated()/1e9:.1f} GB")

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
    )

    # ----- scenarios -----
    scenarios = list(make_scenarios(args.n_scenarios, seed=args.seed))
    print(f"  prepared {len(scenarios)} fintech scenarios")
    w = np.array(args.weights, dtype=float)
    config = FintechRewardConfig()
    tokenize_fn = lambda s: len(tokenizer.encode(s, add_special_tokens=False))

    metrics_history = []
    rng = np.random.default_rng(args.seed)

    # ----- helper: compute per-token logprobs for a batch of (input + completion) -----
    def logprobs_of(model, full_input_ids, prompt_len, attention_mask):
        """Return per-completion-token logprobs of shape (B,) — the SUM of
        token logprobs over the completion span."""
        with torch.set_grad_enabled(model.training):
            out = model(input_ids=full_input_ids, attention_mask=attention_mask)
            logits = out.logits[:, :-1, :]  # predict next token
            targets = full_input_ids[:, 1:]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            tok_logp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            # mask: only count completion tokens
            B, L = tok_logp.shape
            # token at position i corresponds to predicting input_ids[:, i+1].
            # completion is positions prompt_len .. seq_len-1 (inclusive),
            # so the LOGITS predicting them sit at positions prompt_len-1 .. seq_len-2,
            # which in tok_logp are indices prompt_len-1 .. L-1.
            position_ids = torch.arange(L, device=tok_logp.device).unsqueeze(0)
            comp_mask = (position_ids >= (prompt_len.unsqueeze(1) - 1)) & \
                        (attention_mask[:, 1:].bool())
            return (tok_logp * comp_mask.float()).sum(dim=1), comp_mask.float().sum(dim=1)

    # =========================================================================
    # training loop
    # =========================================================================
    t_train_start = time.time()
    for step in range(args.n_steps):
        # --- 1. sample a batch of prompts ---
        prompt_indices = rng.integers(0, len(scenarios), size=args.batch_prompts)
        batch_scenarios = [scenarios[i] for i in prompt_indices]
        prompt_strs = [format_chat(s, tokenizer) for s in batch_scenarios]

        # --- 2. generate m rollouts per prompt with the current policy ---
        # we batch as (P*m) since the prompts may differ
        all_prompts = []
        prompt_idx_repeat = []
        for p_idx, ps in enumerate(prompt_strs):
            all_prompts.extend([ps] * args.m)
            prompt_idx_repeat.extend([p_idx] * args.m)
        enc = tokenizer(all_prompts, return_tensors="pt", padding=True).to("cuda")
        prompt_lens = enc.attention_mask.sum(dim=1)

        policy.eval()
        with torch.no_grad():
            gen_out = policy.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
        policy.train()

        # decode + score
        completions = []
        full_seqs = gen_out  # (P*m, L)
        # we need to know the prompt length per row to extract just the completion
        for i in range(full_seqs.shape[0]):
            comp_tokens = full_seqs[i, prompt_lens[i]:]
            text = tokenizer.decode(comp_tokens, skip_special_tokens=True)
            completions.append(text)

        # rewards: (P, m, R)
        rewards_pmr = np.zeros((args.batch_prompts, args.m, 3), dtype=float)
        for i, gen in enumerate(completions):
            p_idx = prompt_idx_repeat[i]
            j_idx = i % args.m
            c, pol, act, _, _ = score_response(gen, config, tokenize=tokenize_fn)
            rewards_pmr[p_idx, j_idx] = (c, pol, act)

        # --- 3. compute advantages (AN or NA) ---
        adv_pm = compute_advantages(rewards_pmr, w, args.mode)  # (P, m)
        adv_flat = torch.tensor(adv_pm.reshape(-1), dtype=torch.float32, device="cuda")

        # --- 4. PG loss with KL anchor ---
        # Re-tokenize the full prompt+completion pairs with proper attention mask
        # (the generate() output already includes prompt + completion).
        # We compute logp of completion tokens under current policy and ref.
        attn = (full_seqs != tokenizer.pad_token_id).long()
        # logp shapes: (P*m,)
        logp_cur, n_tok = logprobs_of(policy, full_seqs, prompt_lens, attn)
        with torch.no_grad():
            logp_ref, _ = logprobs_of(ref_policy, full_seqs, prompt_lens, attn)

        # importance ratio (we used the SAME policy to sample, so ratio ≈ 1
        # in the first step — but the loss still propagates through logp_cur)
        # standard PG: loss = -E[A * logp_cur]
        # plus KL anchor: kl = (logp_cur - logp_ref) summed per token, ×n_tok
        # normalize per token to keep scale stable
        logp_cur_norm = logp_cur / (n_tok + 1e-6)
        logp_ref_norm = logp_ref / (n_tok + 1e-6)
        pg_loss = -(adv_flat * logp_cur_norm).mean()
        kl = (logp_cur_norm - logp_ref_norm).mean()
        loss = pg_loss + args.kl_coef * kl

        # --- 5. backward + step ---
        optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad], max_norm=1.0,
        )
        optimizer.step()

        # --- log ---
        m_reward = float(rewards_pmr.mean())
        m_compliance = float(rewards_pmr[..., 0].mean())
        m_politeness = float(rewards_pmr[..., 1].mean())
        m_action = float(rewards_pmr[..., 2].mean())
        m_aggregate = float((rewards_pmr @ w).mean())
        metrics_history.append({
            "step": step,
            "loss": float(loss.item()),
            "pg_loss": float(pg_loss.item()),
            "kl": float(kl.item()),
            "grad_norm": float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm),
            "mean_reward": m_reward,
            "mean_compliance": m_compliance,
            "mean_politeness_gated": m_politeness,
            "mean_action": m_action,
            "mean_aggregate_reward": m_aggregate,
        })

        if step % 10 == 0 or step == args.n_steps - 1:
            elapsed = time.time() - t_train_start
            print(f"  step {step:4d}/{args.n_steps}  loss={loss.item():+.4f} "
                  f"pg={pg_loss.item():+.4f} kl={kl.item():+.4f} "
                  f"reward_agg={m_aggregate:+.3f} "
                  f"(c={m_compliance:.2f} p={m_politeness:.2f} a={m_action:.2f}) "
                  f"elapsed={elapsed/60:.1f}min")

        # --- 6. periodic checkpoint ---
        if (step + 1) % args.checkpoint_every == 0 or step == args.n_steps - 1:
            ckpt_dir = out_dir / f"checkpoint_step{step+1:04d}"
            ckpt_dir.mkdir(exist_ok=True, parents=True)
            policy.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            with open(out_dir / "metrics.json", "w") as f:
                json.dump({
                    "mode": args.mode,
                    "model": args.model,
                    "n_steps": args.n_steps,
                    "batch_prompts": args.batch_prompts,
                    "m": args.m,
                    "lr": args.lr,
                    "kl_coef": args.kl_coef,
                    "weights": args.weights,
                    "history": metrics_history,
                }, f, indent=2)
            print(f"    saved checkpoint: {ckpt_dir}")

    print(f"\n=== DONE training in {(time.time()-t_train_start)/60:.1f} min ===")


if __name__ == "__main__":
    main()
