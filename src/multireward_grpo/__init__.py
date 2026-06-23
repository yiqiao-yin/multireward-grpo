"""
multireward-grpo — decoupled & conditioned multi-reward GRPO advantage
estimators, a generalized trainer, and the Theorem-3 verification harness from
the paper "When and Why Decoupling and Conditioning Beat Reweighting in
Multi-Reward GRPO: A U-Statistic Treatment".

Quick start
-----------
>>> from multireward_grpo import GRPOConfig, GRPOTrainer
>>> from multireward_grpo.examples import FintechRewardFunction, make_fintech_prompts
>>> prompts = make_fintech_prompts(400, seed=0)
>>> cfg = GRPOConfig(mode="na", weights=(1.0, 1.0, 0.5), n_steps=200)
>>> trainer = GRPOTrainer(cfg, FintechRewardFunction(), prompts)  # doctest: +SKIP
>>> history = trainer.train()                                     # doctest: +SKIP

Core pieces
-----------
- :func:`compute_advantage` / :func:`compute_advantage_batch` — AN / NA / single.
- :func:`apply_conditioning` — gate one channel on another (Prop 4).
- :class:`GRPOConfig` / :class:`GRPOTrainer` — bring your own prompts + reward fn.
- :func:`analyze` — verify Theorem 3 on a ``(P, K, m, R)`` reward tensor.
- :class:`MockBackend` / :class:`QwenBackend` — generate rollouts.
- :class:`multireward_grpo.runpod.RunPodClient` — run on a cloud GPU.
"""
from __future__ import annotations

from .advantage import (
    apply_conditioning,
    compute_advantage,
    compute_advantage_batch,
)
from .analysis import Thm3Result, analyze, plot_money_scatter, summary_print
from .generation import (
    MockBackend,
    QwenBackend,
    Rollouts,
    pack_for_analysis,
    run_corpus,
)
from .rewards import MathRewardFunction, RewardConfig, RewardFunction, score_generation
from .train import GRPOConfig, GRPOTrainer, train_grpo

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # advantage
    "compute_advantage",
    "compute_advantage_batch",
    "apply_conditioning",
    # training
    "GRPOConfig",
    "GRPOTrainer",
    "train_grpo",
    # rewards
    "RewardFunction",
    "RewardConfig",
    "MathRewardFunction",
    "score_generation",
    # generation
    "MockBackend",
    "QwenBackend",
    "Rollouts",
    "run_corpus",
    "pack_for_analysis",
    # analysis
    "analyze",
    "Thm3Result",
    "plot_money_scatter",
    "summary_print",
]
