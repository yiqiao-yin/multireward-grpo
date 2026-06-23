"""
Rollout generation backends for verifying Theorem 3 on your own data.

Two backends behind a common :class:`Backend` interface:

- :class:`MockBackend` — synthetic Bernoulli/continuous rewards with a tunable
  inter-channel correlation matrix. No GPU, no transformers, no internet. Use it
  to validate the analysis end-to-end before spending GPU time.
- :class:`QwenBackend` — real generation via any Hugging Face causal LM (default
  ``Qwen2.5-1.5B-Instruct``). ``torch``/``transformers`` are imported lazily so
  importing this module never requires them.

:func:`run_corpus` produces a :class:`CorpusResult`; :func:`pack_for_analysis`
reshapes it into the ``(P, K, m, R)`` tensor that
:func:`multireward_grpo.analysis.analyze` consumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

from .rewards import RewardConfig, score_generation


@dataclass
class Rollouts:
    """One group of ``m`` rollouts for one prompt at one seed."""

    prompt: str
    gold: str
    seed: int
    m: int
    generations: list[str]
    rewards: np.ndarray  # (m, R)
    raw_length: Optional[np.ndarray] = None  # (m,) ungated length, for Prop 4

    def __post_init__(self):
        assert self.rewards.shape[0] == self.m, "rewards rows must equal m"
        assert self.rewards.ndim == 2, "rewards must be (m, R)"
        if self.raw_length is not None:
            assert self.raw_length.shape == (self.m,), "raw_length must be (m,)"


@dataclass
class CorpusResult:
    """Output of :func:`run_corpus`, a flat list of :class:`Rollouts`."""

    rollouts: list[Rollouts]
    m_grid: list[int]
    K_seeds: int
    n_prompts: int
    reward_names: tuple[str, ...]
    backend_name: str


class Backend(Protocol):
    name: str
    reward_names: tuple[str, ...]

    def generate_for_prompt(self, prompt: str, gold: str, m: int, seed: int) -> Rollouts: ...


def run_corpus(
    backend: Backend,
    prompts_with_gold: list[tuple[str, str]],
    m_grid: list[int],
    K_seeds: int,
    subsample_from_max: bool = False,
) -> CorpusResult:
    """Generate ``K_seeds`` x ``m_grid`` rollouts for each prompt.

    With ``subsample_from_max=True``, generate once at ``max(m_grid)`` per
    (prompt, seed) and slice leading prefixes for the smaller ``m`` values
    (~4x cheaper for ``[4, 8, 16, 32]`` and statistically valid because a
    prefix of an i.i.d. group is itself a valid i.i.d. group).
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
                        all_rollouts.append(
                            Rollouts(
                                prompt=full.prompt, gold=full.gold, seed=full.seed,
                                m=m, generations=full.generations[:m],
                                rewards=full.rewards[:m],
                                raw_length=(full.raw_length[:m]
                                            if full.raw_length is not None else None),
                            )
                        )
    else:
        for prompt, gold in prompts_with_gold:
            for m in m_grid:
                for k in range(K_seeds):
                    all_rollouts.append(backend.generate_for_prompt(prompt, gold, m, seed=k))
    return CorpusResult(
        rollouts=all_rollouts, m_grid=list(m_grid), K_seeds=K_seeds,
        n_prompts=len(prompts_with_gold), reward_names=backend.reward_names,
        backend_name=backend.name,
    )


@dataclass
class MockBackend:
    """Synthetic rewards with a tunable inter-channel correlation matrix ``C``.

    Per-prompt rewards are drawn from a Gaussian copula ``z ~ N(0, C)`` then
    mapped to realistic marginals (Bernoulli correctness/format, bounded
    continuous length). ``prompt_difficulty`` sets the correctness rate ``p_a``.
    Fixes ``R = 3`` channels (correctness, length, format).
    """

    name: str = "mock"
    reward_names: tuple[str, ...] = ("correctness", "length", "format")
    C: np.ndarray = field(default_factory=lambda: np.eye(3))
    p_format: float = 0.8
    length_mean: float = 0.0
    length_std: float = 0.3
    difficulty_lookup: Optional[dict] = None

    def __post_init__(self):
        C = np.asarray(self.C, dtype=float)
        assert C.shape == (3, 3), "MockBackend fixes R=3 channels"
        assert np.allclose(C, C.T) and np.all(np.diag(C) == 1.0), "C must be a correlation matrix"
        assert np.linalg.eigvalsh(C).min() > -1e-9, "C must be PSD"

    def _p_a(self, prompt: str) -> float:
        if self.difficulty_lookup and prompt in self.difficulty_lookup:
            return float(self.difficulty_lookup[prompt])
        return 0.5

    def generate_for_prompt(self, prompt: str, gold: str, m: int, seed: int) -> Rollouts:
        from scipy.stats import norm

        rng = np.random.default_rng(seed * 7919 + hash(prompt) % (2**31))
        L = np.linalg.cholesky(self.C + 1e-10 * np.eye(3))
        z = rng.standard_normal((m, 3)) @ L.T
        u = norm.cdf(z)
        p_a = self._p_a(prompt)
        correctness = (u[:, 0] < p_a).astype(float)
        format_ = (u[:, 2] < self.p_format).astype(float)
        length = np.tanh(self.length_mean + self.length_std * z[:, 1])
        rewards = np.stack([correctness, length, format_], axis=1)
        gens = [f"<mock prompt={prompt[:20]!r} seed={seed} idx={j}>" for j in range(m)]
        return Rollouts(prompt=prompt, gold=gold, seed=seed, m=m,
                        generations=gens, rewards=rewards, raw_length=length.copy())


class QwenBackend:
    """Real generation via a Hugging Face causal LM (lazy torch/transformers import)."""

    name: str = "qwen-1.5b"
    reward_names: tuple[str, ...] = ("correctness", "length", "format")

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        reward_config: Optional[RewardConfig] = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype), device_map=device,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reward_config = reward_config or RewardConfig()
        self._torch = torch

    def _format_prompt(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": (
                "You are a precise mathematical assistant. Solve the problem step "
                "by step, then put the final numerical answer inside \\boxed{...}."
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
        self._torch.manual_seed(seed * 1009 + hash(prompt) % (2**16))
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=True,
                temperature=self.temperature, top_p=self.top_p,
                num_return_sequences=1, pad_token_id=self.tokenizer.eos_token_id,
            )
        in_len = inputs.input_ids.shape[1]
        completions = self.tokenizer.batch_decode(out[:, in_len:], skip_special_tokens=True)
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
                        generations=list(completions), rewards=rewards, raw_length=raw_length)


def pack_for_analysis(
    corpus: CorpusResult,
    m: int,
    n_prompts: Optional[int] = None,
    return_raw_length: bool = False,
):
    """Reshape a :class:`CorpusResult` into a ``(P, K, m, R)`` reward tensor.

    With ``return_raw_length=True`` also returns the aligned ``(P, K, m)``
    raw-length tensor for Proposition-4 gamma sweeps.
    """
    prompts: list[str] = []
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
        seeds = sorted(by_prompt[p], key=lambda r: r.seed)
        for k, r in enumerate(seeds[:K]):
            out[i, k] = r.rewards
            if return_raw_length and r.raw_length is not None:
                raw_len_out[i, k] = r.raw_length
    if return_raw_length:
        return out, raw_len_out
    return out


__all__ = [
    "Backend", "Rollouts", "CorpusResult",
    "MockBackend", "QwenBackend",
    "run_corpus", "pack_for_analysis",
]
