"""
Multi-reward GRPO advantage estimators — AN, NA, single — and conditioning.

These are the central objects of the paper "Decoupling and Conditioning
Reshape Influence Allocation and the Gradient-Noise Floor in Multi-Reward GRPO
under a Finite-Sample U-Statistic Analysis". Given a group of ``m``
rollouts each scored on ``R`` reward channels, an *advantage estimator* turns
the ``(m, R)`` reward matrix into a per-rollout scalar advantage ``A_j`` that
the policy-gradient update consumes.

Two orderings:

- **AN — Aggregate-then-Normalize** (the classic GRPO baseline): scalarize the
  reward vector with the weights ``w`` first, then group-normalize the scalar.
  Subject to the high-variance-channel dominance of Proposition 1 and the
  resolution collapse of Proposition 2.
- **NA — Normalize-then-Aggregate** (the "decoupled" estimator analysed in the
  paper; = MO-GRPO/GDPO): group-normalize each channel independently, then take
  the weighted sum. Restores weight-proportional influence (Prop 1) and
  advantage resolution under heterogeneous scales (Prop 2). Theorem 3 gives its
  gradient-MSE floor ``(tau^2 / m) * w^T C w``.

All functions are NumPy-only and operate on a single group ``(m, R)``. Batched
helpers operating on ``(P, m, R)`` are provided for the training loop.
"""
from __future__ import annotations

import numpy as np

# delta in the paper's notation; guards the per-channel std near zero-variance
# groups. Matches the value used in the released analysis pipeline.
EPS = 1e-8

VALID_MODES = ("na", "an", "single")


def compute_advantage(
    rewards: np.ndarray,
    w: np.ndarray,
    mode: str = "na",
    eps: float = EPS,
) -> np.ndarray:
    """Per-rollout advantages for one group of rollouts.

    Parameters
    ----------
    rewards : np.ndarray, shape ``(m, R)``
        ``m`` rollouts scored on ``R`` reward channels.
    w : np.ndarray, shape ``(R,)``
        Non-negative objective weights.
    mode : {"na", "an", "single"}
        - ``"na"`` Normalize-then-Aggregate (paper's recommendation).
        - ``"an"`` Aggregate-then-Normalize (GRPO baseline).
        - ``"single"`` ignore every channel except channel 0, then normalize
          (the "throw away the multi-reward structure" baseline).
    eps : float
        Numerical floor added to every standard deviation.

    Returns
    -------
    np.ndarray, shape ``(m,)``
        The advantage of each rollout.
    """
    rewards = np.asarray(rewards, dtype=float)
    w = np.asarray(w, dtype=float)
    if rewards.ndim != 2:
        raise ValueError(f"rewards must be (m, R); got shape {rewards.shape}")
    if rewards.shape[1] != w.shape[0]:
        raise ValueError(
            f"rewards has R={rewards.shape[1]} channels but w has {w.shape[0]} weights"
        )

    if mode == "an":
        s = rewards @ w  # (m,) scalarize first
        return (s - s.mean()) / (s.std(ddof=0) + eps)  # then group-normalize
    if mode == "na":
        mu = rewards.mean(axis=0, keepdims=True)  # (1, R)
        sd = rewards.std(axis=0, ddof=0, keepdims=True) + eps
        z = (rewards - mu) / sd  # decoupled per-channel standardize
        return z @ w  # then weighted aggregate
    if mode == "single":
        s = rewards[:, 0]  # channel 0 only (e.g. correctness / compliance)
        return (s - s.mean()) / (s.std(ddof=0) + eps)
    raise ValueError(f"mode must be one of {VALID_MODES}; got {mode!r}")


def compute_advantage_batch(
    rewards_pmr: np.ndarray,
    w: np.ndarray,
    mode: str = "na",
    eps: float = EPS,
) -> np.ndarray:
    """Vectorized :func:`compute_advantage` over a batch of ``P`` groups.

    Parameters
    ----------
    rewards_pmr : np.ndarray, shape ``(P, m, R)``
    w : np.ndarray, shape ``(R,)``
    mode, eps : see :func:`compute_advantage`.

    Returns
    -------
    np.ndarray, shape ``(P, m)``
    """
    rewards_pmr = np.asarray(rewards_pmr, dtype=float)
    w = np.asarray(w, dtype=float)
    if rewards_pmr.ndim != 3:
        raise ValueError(f"rewards must be (P, m, R); got shape {rewards_pmr.shape}")
    P, m, R = rewards_pmr.shape
    if R != w.shape[0]:
        raise ValueError(f"rewards has R={R} channels but w has {w.shape[0]} weights")

    if mode == "an":
        s_pm = rewards_pmr @ w
        mu = s_pm.mean(axis=1, keepdims=True)
        sd = s_pm.std(axis=1, ddof=0, keepdims=True)
        return (s_pm - mu) / (sd + eps)
    if mode == "na":
        mu = rewards_pmr.mean(axis=1, keepdims=True)
        sd = rewards_pmr.std(axis=1, ddof=0, keepdims=True)
        z = (rewards_pmr - mu) / (sd + eps)
        return z @ w
    if mode == "single":
        s_pm = rewards_pmr[..., 0]
        mu = s_pm.mean(axis=1, keepdims=True)
        sd = s_pm.std(axis=1, ddof=0, keepdims=True)
        return (s_pm - mu) / (sd + eps)
    raise ValueError(f"mode must be one of {VALID_MODES}; got {mode!r}")


def apply_conditioning(
    rewards: np.ndarray,
    gate_idx: int,
    cond_idx: int,
    impl: str = "zero_fill",
) -> np.ndarray:
    """Gate channel ``cond_idx`` on channel ``gate_idx`` ("b counts only if a passes").

    Implements the conditioning rule of Proposition 4. Returns a *new* array.

    Parameters
    ----------
    rewards : np.ndarray, shape ``(m, R)``
    gate_idx : int
        Channel ``a`` — the "harder" gate (e.g. correctness / compliance).
    cond_idx : int
        Channel ``b`` — the conditioned channel (e.g. length / politeness).
    impl : {"none", "zero_fill", "subgroup"}
        - ``"none"`` identity (baseline for the bias law).
        - ``"zero_fill"`` ``r_b <- r_b * 1[r_a == 1]`` (recommended; preserves the
          U-statistic structure, centering happens inside NA).
        - ``"subgroup"`` center ``r_b`` only over the passing subgroup; degenerates
          structurally when fewer than 2 rollouts pass the gate.

    Returns
    -------
    np.ndarray, shape ``(m, R)``
    """
    r = np.asarray(rewards, dtype=float).copy()
    if impl == "none":
        return r
    gate = (r[:, gate_idx] == 1).astype(r.dtype)
    if impl == "zero_fill":
        r[:, cond_idx] = r[:, cond_idx] * gate
        return r
    if impl == "subgroup":
        if int(gate.sum()) < 2:
            r[:, cond_idx] = 0.0  # degenerate; < 2 rollouts pass the gate
            return r
        sub_mean = r[gate.astype(bool), cond_idx].mean()
        r[:, cond_idx] = (r[:, cond_idx] - sub_mean) * gate
        return r
    raise ValueError(f"impl must be none|zero_fill|subgroup; got {impl!r}")


__all__ = [
    "compute_advantage",
    "compute_advantage_batch",
    "apply_conditioning",
    "EPS",
    "VALID_MODES",
]
