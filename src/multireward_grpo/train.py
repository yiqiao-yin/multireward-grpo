"""
Generalized multi-reward GRPO trainer.

This is the democratized version of the paper's hand-rolled training loop: you
bring **your own prompts**, **your own reward function**, a **weight vector**,
and an **advantage mode** (``"na"`` recommended, ``"an"`` baseline, ``"single"``
ablation), and :class:`GRPOTrainer` runs group-relative policy optimization with
a KL anchor and saves a LoRA adapter.

The four GRPO ingredients (identical to the paper's experiments):

1. **Group rollouts** — for each prompt, sample ``m`` completions at temperature
   ``tau`` from the current policy.
2. **Per-rollout multi-channel rewards** — score each completion with your
   reward function into an ``R``-vector.
3. **Advantage computation** — AN / NA / single via
   :func:`multireward_grpo.advantage.compute_advantage_batch`.
4. **Policy gradient + KL anchor** — PG loss against advantages, KL penalty to a
   frozen reference policy.

Data contract
-------------
``prompts`` is a list; each item is one of:

- ``str`` — used as the user message.
- ``list[dict]`` — chat messages ``[{"role": ..., "content": ...}, ...]``.
- ``dict`` — either ``{"messages": [...]}`` or ``{"prompt": "..."}`` /
  ``{"text": "..."}``, plus any extra keys (e.g. ``"gold"``) which are passed
  through to the reward function.

``reward_fn(completion: str, prompt_item) -> Sequence[float]`` must return
exactly ``len(config.weights)`` channel scores. See
:mod:`multireward_grpo.rewards` for ready-made examples.

``torch``/``transformers``/``peft`` are imported lazily inside :meth:`train`, so
importing this module does not require a GPU stack.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .advantage import compute_advantage_batch

RewardFn = Callable[[str, Any], Sequence[float]]


@dataclass
class GRPOConfig:
    """Hyperparameters for :class:`GRPOTrainer`."""

    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    mode: str = "na"  # "na" (paper) | "an" (baseline) | "single" (ablation)
    weights: tuple[float, ...] = (1.0, 1.0, 0.5)
    n_steps: int = 200
    batch_prompts: int = 4  # P, prompts per step
    m: int = 8  # rollouts per prompt (group size)
    max_new_tokens: int = 96
    temperature: float = 0.8
    lr: float = 5e-6
    kl_coef: float = 0.05
    ppo_clip: float = 0.2
    seed: int = 0
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    device: str = "cuda"
    dtype: str = "bfloat16"
    out_dir: str = "grpo_out"
    checkpoint_every: int = 50
    system_prompt: Optional[str] = None  # prepended to str/{"prompt"} items
    verbose: bool = True


class GRPOTrainer:
    """Hand-rolled multi-reward GRPO trainer with a one-line advantage swap.

    Parameters
    ----------
    config : GRPOConfig
    reward_fn : callable ``(completion, prompt_item) -> Sequence[float]``
        Returns ``len(config.weights)`` channel scores for one completion.
    prompts : list
        Prompt items (see module docstring for accepted shapes).
    """

    def __init__(self, config: GRPOConfig, reward_fn: RewardFn, prompts: Sequence[Any]):
        if config.mode not in ("na", "an", "single"):
            raise ValueError(f"mode must be na|an|single; got {config.mode!r}")
        if not prompts:
            raise ValueError("prompts must be a non-empty list")
        self.config = config
        self.reward_fn = reward_fn
        self.prompts = list(prompts)
        self.w = np.asarray(config.weights, dtype=float)
        self.R = len(self.w)
        self.metrics_history: list[dict] = []
        self._policy = None
        self._tokenizer = None

    # -- prompt formatting ---------------------------------------------------
    def _to_messages(self, item: Any) -> list[dict]:
        sys_prompt = self.config.system_prompt
        if isinstance(item, str):
            msgs = []
            if sys_prompt:
                msgs.append({"role": "system", "content": sys_prompt})
            msgs.append({"role": "user", "content": item})
            return msgs
        if isinstance(item, dict):
            if "messages" in item:
                return list(item["messages"])
            text = item.get("prompt", item.get("text"))
            if text is None:
                raise ValueError(
                    "dict prompt items need a 'messages', 'prompt', or 'text' key"
                )
            msgs = []
            if sys_prompt:
                msgs.append({"role": "system", "content": sys_prompt})
            msgs.append({"role": "user", "content": text})
            return msgs
        if isinstance(item, (list, tuple)):
            return list(item)
        raise TypeError(f"unsupported prompt item type: {type(item)}")

    def _format_prompt(self, item: Any) -> str:
        return self._tokenizer.apply_chat_template(
            self._to_messages(item), tokenize=False, add_generation_prompt=True
        )

    def _score(self, completion: str, item: Any) -> np.ndarray:
        vec = self.reward_fn(completion, item)
        vec = np.asarray(list(vec), dtype=float)
        if vec.shape != (self.R,):
            raise ValueError(
                f"reward_fn must return {self.R} channels (len(weights)); "
                f"got {vec.shape}"
            )
        return vec

    # -- training ------------------------------------------------------------
    def train(self) -> list[dict]:
        """Run the GRPO loop. Returns the per-step metrics history."""
        cfg = self.config
        import torch
        from torch.nn.utils import clip_grad_norm_
        from transformers import AutoModelForCausalLM, AutoTokenizer

        out_dir = Path(cfg.out_dir) / f"{cfg.mode}_seed{cfg.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"=== GRPO training: mode={cfg.mode}, model={cfg.model} ===")
        self._log(f"  out_dir={out_dir}  R={self.R}  weights={self.w.tolist()}")

        torch.manual_seed(cfg.seed)
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer

        torch_dtype = getattr(torch, cfg.dtype)
        policy = AutoModelForCausalLM.from_pretrained(
            cfg.model, torch_dtype=torch_dtype, device_map=cfg.device,
        )
        if cfg.use_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            policy = get_peft_model(
                policy,
                LoraConfig(
                    r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                    target_modules=list(cfg.lora_target_modules),
                    lora_dropout=cfg.lora_dropout, bias="none",
                    task_type=TaskType.CAUSAL_LM,
                ),
            )
            if cfg.verbose:
                policy.print_trainable_parameters()
        policy.train()
        self._policy = policy

        ref_policy = AutoModelForCausalLM.from_pretrained(
            cfg.model, torch_dtype=torch_dtype, device_map=cfg.device,
        )
        ref_policy.eval()
        for p in ref_policy.parameters():
            p.requires_grad_(False)
        self._log(f"  loaded in {time.time()-t0:.0f}s")

        optimizer = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad],
            lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0,
        )
        rng = np.random.default_rng(cfg.seed)

        def logprobs_of(model, full_input_ids, prompt_len, attention_mask):
            with torch.set_grad_enabled(model.training):
                out = model(input_ids=full_input_ids, attention_mask=attention_mask)
                logits = out.logits[:, :-1, :]
                targets = full_input_ids[:, 1:]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                tok_logp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
                _, L = tok_logp.shape
                position_ids = torch.arange(L, device=tok_logp.device).unsqueeze(0)
                comp_mask = (position_ids >= (prompt_len.unsqueeze(1) - 1)) & (
                    attention_mask[:, 1:].bool()
                )
                return (tok_logp * comp_mask.float()).sum(dim=1), comp_mask.float().sum(dim=1)

        t_train = time.time()
        for step in range(cfg.n_steps):
            idx = rng.integers(0, len(self.prompts), size=cfg.batch_prompts)
            batch = [self.prompts[i] for i in idx]
            prompt_strs = [self._format_prompt(it) for it in batch]

            all_prompts, prompt_idx_repeat = [], []
            for p_idx, ps in enumerate(prompt_strs):
                all_prompts.extend([ps] * cfg.m)
                prompt_idx_repeat.extend([p_idx] * cfg.m)
            enc = tokenizer(all_prompts, return_tensors="pt", padding=True).to(cfg.device)
            prompt_lens = enc.attention_mask.sum(dim=1)

            policy.eval()
            with torch.no_grad():
                gen_out = policy.generate(
                    **enc, max_new_tokens=cfg.max_new_tokens, do_sample=True,
                    temperature=cfg.temperature, pad_token_id=tokenizer.pad_token_id,
                )
            policy.train()

            completions = []
            for i in range(gen_out.shape[0]):
                completions.append(
                    tokenizer.decode(gen_out[i, prompt_lens[i]:], skip_special_tokens=True)
                )

            rewards_pmr = np.zeros((cfg.batch_prompts, cfg.m, self.R), dtype=float)
            for i, gen in enumerate(completions):
                rewards_pmr[prompt_idx_repeat[i], i % cfg.m] = self._score(gen, batch[prompt_idx_repeat[i]])

            adv_pm = compute_advantage_batch(rewards_pmr, self.w, cfg.mode)
            adv_flat = torch.tensor(adv_pm.reshape(-1), dtype=torch.float32, device=cfg.device)

            attn = (gen_out != tokenizer.pad_token_id).long()
            logp_cur, n_tok = logprobs_of(policy, gen_out, prompt_lens, attn)
            with torch.no_grad():
                logp_ref, _ = logprobs_of(ref_policy, gen_out, prompt_lens, attn)
            logp_cur_norm = logp_cur / (n_tok + 1e-6)
            logp_ref_norm = logp_ref / (n_tok + 1e-6)
            pg_loss = -(adv_flat * logp_cur_norm).mean()
            kl = (logp_cur_norm - logp_ref_norm).mean()
            loss = pg_loss + cfg.kl_coef * kl

            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

            rec = {
                "step": step, "loss": float(loss.item()),
                "pg_loss": float(pg_loss.item()), "kl": float(kl.item()),
                "grad_norm": float(grad_norm) if not hasattr(grad_norm, "item") else float(grad_norm.item()),
                "mean_reward": float(rewards_pmr.mean()),
                "mean_aggregate_reward": float((rewards_pmr @ self.w).mean()),
            }
            for c in range(self.R):
                rec[f"mean_channel_{c}"] = float(rewards_pmr[..., c].mean())
            self.metrics_history.append(rec)

            if cfg.verbose and (step % 10 == 0 or step == cfg.n_steps - 1):
                ch = " ".join(f"c{c}={rec[f'mean_channel_{c}']:.2f}" for c in range(self.R))
                self._log(
                    f"  step {step:4d}/{cfg.n_steps}  loss={loss.item():+.4f} "
                    f"kl={kl.item():+.4f} reward_agg={rec['mean_aggregate_reward']:+.3f} "
                    f"({ch}) elapsed={(time.time()-t_train)/60:.1f}min"
                )

            if (step + 1) % cfg.checkpoint_every == 0 or step == cfg.n_steps - 1:
                self._checkpoint(out_dir, step)

        self._log(f"=== DONE in {(time.time()-t_train)/60:.1f} min ===")
        return self.metrics_history

    def _checkpoint(self, out_dir: Path, step: int) -> None:
        ckpt = out_dir / f"checkpoint_step{step+1:04d}"
        ckpt.mkdir(parents=True, exist_ok=True)
        self._policy.save_pretrained(str(ckpt))
        self._tokenizer.save_pretrained(str(ckpt))
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(
                {"config": asdict(self.config), "history": self.metrics_history},
                f, indent=2, default=list,
            )
        self._log(f"    saved checkpoint: {ckpt}")

    def save(self, path: str) -> None:
        """Save the current LoRA adapter (or full model) + tokenizer to ``path``."""
        if self._policy is None:
            raise RuntimeError("nothing to save; call train() first")
        Path(path).mkdir(parents=True, exist_ok=True)
        self._policy.save_pretrained(path)
        self._tokenizer.save_pretrained(path)

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(msg, flush=True)


def train_grpo(config: GRPOConfig, reward_fn: RewardFn, prompts: Sequence[Any]) -> list[dict]:
    """Convenience one-shot: build a :class:`GRPOTrainer` and run :meth:`train`."""
    return GRPOTrainer(config, reward_fn, prompts).train()


__all__ = ["GRPOConfig", "GRPOTrainer", "train_grpo"]
