"""
Rollout generators for the Thm 3 LLM validation pipeline.

Two backends behind a common interface:

  MockBackend       — synthetic Bernoulli rewards with controllable correlation.
                      No GPU, no transformers, no internet. Used to validate
                      the analysis code end-to-end before any LLM is touched.

  QwenBackend       — real generation via Qwen2.5-1.5B-Instruct using
                      transformers + batched sampling. Lazy-imported so the
                      mock pipeline does not require torch/transformers.

Both backends return a dataclass with shape
    Rollouts(prompt, gold, generations[m], rewards (m, R))
which downstream `llm_analysis` consumes.

The contract: a single Backend exposes
    .generate_for_prompt(prompt, gold, m, seed) -> Rollouts
and aggregates over a corpus with
    .run_corpus(prompts, m_grid, K_seeds) -> list[Rollouts]
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol
import numpy as np

from llm_rewards import RewardConfig, score_generation


# ---------------------------------------------------------------------------
# common dataclasses
# ---------------------------------------------------------------------------
@dataclass
class Rollouts:
    """One group of m rollouts for one prompt at one seed."""
    prompt: str
    gold: str
    seed: int
    m: int
    generations: list[str]
    rewards: np.ndarray  # (m, R)
    raw_length: Optional[np.ndarray] = None  # (m,) — ungated length, for Prop 4

    def __post_init__(self):
        assert self.rewards.shape[0] == self.m, "rewards rows must equal m"
        assert self.rewards.ndim == 2, "rewards must be (m, R)"
        if self.raw_length is not None:
            assert self.raw_length.shape == (self.m,), "raw_length must be (m,)"


@dataclass
class CorpusResult:
    """Output of run_corpus: indexed by (prompt_idx, m, seed) -> Rollouts."""
    rollouts: list[Rollouts]
    m_grid: list[int]
    K_seeds: int
    n_prompts: int
    reward_names: tuple[str, ...]
    backend_name: str


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------
class Backend(Protocol):
    name: str
    reward_names: tuple[str, ...]

    def generate_for_prompt(self, prompt: str, gold: str, m: int, seed: int) -> Rollouts: ...


def run_corpus(backend: Backend, prompts_with_gold: list[tuple[str, str]],
               m_grid: list[int], K_seeds: int,
               subsample_from_max: bool = False) -> CorpusResult:
    """Generate K seeds × m_grid rollouts for each prompt. Yields a flat list.

    If subsample_from_max is True, generate once at m_max = max(m_grid) per
    (prompt, seed) and slice the leading m rollouts for each smaller m. This
    is ~4× cheaper for m_grid = [4, 8, 16, 32] on real LLM backends because a
    contiguous prefix of an i.i.d. group is itself a valid i.i.d. group of
    the smaller size — no information is lost relative to fresh draws.
    """
    all_rollouts: list[Rollouts] = []
    if subsample_from_max:
        m_max = max(m_grid)
        for prompt, gold in prompts_with_gold:
            for k in range(K_seeds):
                full = backend.generate_for_prompt(prompt, gold, m_max, seed=k)
                for m in m_grid:
                    if m == m_max:
                        all_rollouts.append(full)
                    else:
                        all_rollouts.append(Rollouts(
                            prompt=full.prompt, gold=full.gold, seed=full.seed,
                            m=m,
                            generations=full.generations[:m],
                            rewards=full.rewards[:m],
                            raw_length=(full.raw_length[:m]
                                        if full.raw_length is not None else None),
                        ))
    else:
        for prompt, gold in prompts_with_gold:
            for m in m_grid:
                for k in range(K_seeds):
                    all_rollouts.append(backend.generate_for_prompt(prompt, gold, m, seed=k))
    return CorpusResult(
        rollouts=all_rollouts,
        m_grid=list(m_grid),
        K_seeds=K_seeds,
        n_prompts=len(prompts_with_gold),
        reward_names=backend.reward_names,
        backend_name=backend.name,
    )


# ---------------------------------------------------------------------------
# MockBackend — no GPU, synthetic rewards
# ---------------------------------------------------------------------------
@dataclass
class MockBackend:
    """Generate synthetic reward vectors with controllable per-prompt
    correctness rate and inter-channel correlation.

    Per-prompt rewards r_j ∈ R^R are sampled from a Gaussian copula:
      z_j ~ N(0, C),  r_j^{(ell)} = G_ell( Phi(z_j^{(ell)}) )
    where G_ell is the inverse CDF of the per-channel reward distribution
    and Phi is the standard normal CDF. This lets us:
      - tune C exactly (the reward-correlation matrix Thm 3 cares about)
      - keep the marginal of each channel realistic (Bernoulli for
        correctness/format, bounded continuous for length)

    The `prompt_difficulty` argument (in [0, 1]) sets the marginal p_a for
    the correctness channel — closer to 0 = harder. This is what we vary
    across prompts to test Prop 4's p_a sweep without ever touching an LLM.
    """
    name: str = "mock"
    reward_names: tuple[str, ...] = ("correctness", "length", "format")
    # correlation matrix among channels (3x3 by default)
    C: np.ndarray = field(default_factory=lambda: np.eye(3))
    # marginals
    p_format: float = 0.8                # P(format channel = 1)
    length_mean: float = 0.0             # mean of (already-tanh-bounded) length reward
    length_std: float = 0.3              # std of length reward
    # difficulty assignment from prompt content
    difficulty_lookup: Optional[dict] = None

    def __post_init__(self):
        C = np.asarray(self.C, dtype=float)
        assert C.shape == (3, 3), "MockBackend currently fixes R=3 channels"
        assert np.allclose(C, C.T) and np.all(np.diag(C) == 1.0), "C must be a correlation matrix"
        # eigenvalue check; allow tiny negative eigenvalues from numerical noise
        evals = np.linalg.eigvalsh(C)
        assert evals.min() > -1e-9, f"C must be PSD; got min eigvalue {evals.min()}"

    def _p_a(self, prompt: str) -> float:
        """Correctness rate for this prompt. Default: extract from `prompt_difficulty`
        lookup, fall back to 0.5."""
        if self.difficulty_lookup and prompt in self.difficulty_lookup:
            return float(self.difficulty_lookup[prompt])
        return 0.5

    def generate_for_prompt(self, prompt: str, gold: str, m: int, seed: int) -> Rollouts:
        rng = np.random.default_rng(seed * 7919 + hash(prompt) % (2**31))
        # draw correlated standard-normal latents
        L = np.linalg.cholesky(self.C + 1e-10 * np.eye(3))
        z = rng.standard_normal((m, 3)) @ L.T  # (m, R)

        # transform to marginals
        from scipy.stats import norm
        u = norm.cdf(z)  # uniform marginals, same copula
        p_a = self._p_a(prompt)
        correctness = (u[:, 0] < p_a).astype(float)
        format_ = (u[:, 2] < self.p_format).astype(float)
        # length: bounded continuous; use normal -> tanh scaling so reward ∈ (-1, 1)
        length = np.tanh(self.length_mean + self.length_std * z[:, 1])

        # gate length by correctness (Prop 4 contamination knob lives in caller)
        rewards = np.stack([correctness, length, format_], axis=1)  # (m, R)

        # we do not actually generate text in mock mode; we fabricate a marker
        gens = [f"<mock generation prompt={prompt[:20]!r} seed={seed} idx={j}>" for j in range(m)]
        return Rollouts(prompt=prompt, gold=gold, seed=seed, m=m,
                        generations=gens, rewards=rewards,
                        raw_length=length.copy())


# ---------------------------------------------------------------------------
# QwenBackend — real generation (lazy-imported so mock works without torch)
# ---------------------------------------------------------------------------
class QwenBackend:
    """Wraps a Hugging Face causal-LM with batched sampling.

    Defaults to Qwen2.5-1.5B-Instruct which fits on a single 24 GB consumer
    GPU and generates GSM8K-quality answers at adequate speed. Caller can
    override model_name for a larger model with more VRAM.

    Lazy imports torch/transformers in `__init__`, so importing this module
    by itself does not require those deps.
    """
    name: str = "qwen-1.5b"
    reward_names: tuple[str, ...] = ("correctness", "length", "format")

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 device: str = "cuda",
                 dtype: str = "bfloat16",
                 max_new_tokens: int = 256,
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 reward_config: Optional[RewardConfig] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name
        self.device = device
        torch_dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reward_config = reward_config or RewardConfig()
        self._torch = torch

    def _format_prompt(self, prompt: str) -> str:
        """Apply Qwen's chat template; for GSM8K-style we wrap in a clear
        instruction asking for \\boxed{} answer."""
        messages = [
            {"role": "system", "content": (
                "You are a precise mathematical assistant. Solve the problem "
                "step by step, then put the final numerical answer inside "
                "\\boxed{...}."
            )},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _token_counter(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate_for_prompt(self, prompt: str, gold: str, m: int, seed: int) -> Rollouts:
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer([formatted] * m, return_tensors="pt", padding=True).to(self.device)
        # seed for reproducibility within this group
        self._torch.manual_seed(seed * 1009 + hash(prompt) % (2**16))
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # decode only the newly-generated portion
        in_len = inputs.input_ids.shape[1]
        completions = self.tokenizer.batch_decode(out[:, in_len:], skip_special_tokens=True)

        # score each completion
        R = len(self.reward_names)
        rewards = np.zeros((m, R), dtype=float)
        raw_length = np.zeros(m, dtype=float)
        for j, gen in enumerate(completions):
            c, length, fmt, raw_l = score_generation(
                gen, gold, self.reward_config, tokenize=self._token_counter,
            )
            rewards[j] = (c, length, fmt)
            raw_length[j] = raw_l
        return Rollouts(prompt=prompt, gold=gold, seed=seed, m=m,
                        generations=list(completions), rewards=rewards,
                        raw_length=raw_length)


# ---------------------------------------------------------------------------
# tiny convenience: pack/unpack a CorpusResult to (K, m, R) tensors for analysis
# ---------------------------------------------------------------------------
def pack_for_analysis(corpus: CorpusResult, m: int, n_prompts: Optional[int] = None,
                      return_raw_length: bool = False):
    """Return reward tensor of shape (P, K, m, R) for a single m value.

    If return_raw_length=True, also returns the raw-length tensor of shape
    (P, K, m), aligned to the rewards tensor, suitable for Prop 4 γ sweeps.
    """
    prompts = []
    by_prompt: dict[str, list[Rollouts]] = {}
    for r in corpus.rollouts:
        if r.m != m:
            continue
        if r.prompt not in by_prompt:
            by_prompt[r.prompt] = []
            prompts.append(r.prompt)
        by_prompt[r.prompt].append(r)
    if n_prompts:
        prompts = prompts[:n_prompts]
    R = corpus.rollouts[0].rewards.shape[1]
    K = corpus.K_seeds
    out = np.zeros((len(prompts), K, m, R), dtype=float)
    raw_len_out = np.zeros((len(prompts), K, m), dtype=float) if return_raw_length else None
    for i, p in enumerate(prompts):
        # sort by seed for determinism
        seeds = sorted(by_prompt[p], key=lambda r: r.seed)
        for k, r in enumerate(seeds[:K]):
            out[i, k] = r.rewards
            if return_raw_length and r.raw_length is not None:
                raw_len_out[i, k] = r.raw_length
    if return_raw_length:
        return out, raw_len_out
    return out


def save_corpus_generations_jsonl(
    corpus: CorpusResult, path: str,
    prompts_with_gold: list[tuple[str, str]] | None = None,
) -> None:
    """Save per-rollout generations (the actual text the LLM produced) to JSONL.

    To avoid duplication when `--subsample-from-max` is used (where the same
    rollouts appear in multiple m-groups), we only emit one JSON line per
    rollout at the largest m. The optional `prompts_with_gold` maps each
    rollout's `prompt` string back to its position in the original corpus
    (so the parquet can join by `prompt_idx`).
    """
    import json as _json
    m_max = max(corpus.m_grid)
    prompt_to_idx: dict[str, int] = {}
    if prompts_with_gold is not None:
        for i, (p, _g) in enumerate(prompts_with_gold):
            prompt_to_idx[p] = i

    n_written = 0
    with open(path, "w") as f:
        for r in corpus.rollouts:
            if r.m != m_max:
                continue
            p_idx = prompt_to_idx.get(r.prompt, -1)
            for j, gen in enumerate(r.generations):
                rec = {
                    "prompt_idx": int(p_idx),
                    "gold": r.gold,
                    "seed_idx": int(r.seed),
                    "rollout_idx": int(j),
                    "completion": gen,
                }
                # rewards per channel
                for r_idx, name in enumerate(corpus.reward_names):
                    rec[str(name)] = float(r.rewards[j, r_idx])
                if r.raw_length is not None:
                    rec["raw_length"] = float(r.raw_length[j])
                f.write(_json.dumps(rec) + "\n")
                n_written += 1
    print(f"  saved {n_written} rollout records → {path}")


def save_corpus_npz(corpus: CorpusResult, path: str) -> None:
    """Save per-m reward + raw_length tensors to a compressed npz.

    Layout: for each m in m_grid we save `rewards_m{m}` (P, K, m, R) and
    `raw_length_m{m}` (P, K, m). The metadata dict (m_grid, K, reward names,
    backend) is also stored for downstream tools.
    """
    out: dict[str, np.ndarray] = {}
    for m in corpus.m_grid:
        rewards, raw_l = pack_for_analysis(corpus, m=m, return_raw_length=True)
        out[f"rewards_m{m}"] = rewards
        out[f"raw_length_m{m}"] = raw_l
    out["m_grid"] = np.array(corpus.m_grid, dtype=int)
    out["K_seeds"] = np.array(corpus.K_seeds, dtype=int)
    out["n_prompts"] = np.array(corpus.n_prompts, dtype=int)
    out["reward_names"] = np.array(corpus.reward_names)
    out["backend_name"] = np.array(corpus.backend_name)
    np.savez_compressed(path, **out)


__all__ = [
    "Backend", "Rollouts", "CorpusResult",
    "MockBackend", "QwenBackend",
    "run_corpus", "pack_for_analysis", "save_corpus_npz",
    "save_corpus_generations_jsonl",
]
