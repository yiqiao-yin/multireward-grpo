"""
gdpo_adapter.py — framework-agnostic AN/NA + conditioning switch for the
multi-reward GRPO validation pipeline.

This is a *stub*. It is not runnable end-to-end. It documents the four
hand-offs the user needs to wire into GDPO's (or DAPO's, or any verl/TRL
fork's) training loop so the sweeps in `gdpo_reproduction_plan.md` (§c)
become a flag flip rather than a fork. Tie-backs to problem-statement.md
are in each docstring.

Drop-in contract: replace the host repo's `compute_advantage(rewards, ...)`
call with `compute_multireward_advantage(rewards, w, mode, ...)` and route
the `mode` from a config flag. Nothing else changes.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-8  # δ in problem-statement.md §2; guards ĉ_ℓ near zero-variance groups.


def compute_multireward_advantage(
    rewards: np.ndarray,        # (m, R): m rollouts, R reward channels
    w: np.ndarray,              # (R,):   non-negative objective weights
    mode: str = "NA",           # "AN" (scalarize-then-normalize, GRPO baseline)
                                # or "NA" (normalize-then-aggregate, GDPO decoupled)
) -> np.ndarray:
    """Return per-rollout advantage A_j of shape (m,).

    AN — problem-statement.md §2, GRPO baseline. Aggregate, then group-normalize.
         Subject to Prop 1 (high-σ channel dominates) and Prop 2 (sum-lattice
         resolution collapse under discrete rewards).
    NA — problem-statement.md §2, GDPO / decoupled. Per-channel normalize, then
         weighted sum. Restores weight-proportional influence (Prop 1) and
         product-lattice resolution under heterogeneous σ (Prop 2). Thm 3 says
         its MSE floor is (w^T C w) ζ_1 / m.
    """
    assert rewards.ndim == 2 and rewards.shape[1] == w.shape[0]
    if mode == "AN":
        s = rewards @ w                                # scalarize first
        return (s - s.mean()) / (s.std(ddof=0) + EPS)  # then group-normalize
    if mode == "NA":
        mu = rewards.mean(axis=0, keepdims=True)       # (1, R)
        sd = rewards.std(axis=0, ddof=0, keepdims=True) + EPS
        z = (rewards - mu) / sd                        # decoupled per-channel std
        return z @ w                                   # weighted aggregate
    raise ValueError(f"mode must be 'AN' or 'NA', got {mode!r}")


def apply_conditioning(
    rewards: np.ndarray,        # (m, R)
    gate_idx: int,              # channel a: the "harder" gate ("correct")
    cond_idx: int,              # channel b: the conditioned channel ("length"/"format")
    impl: str = "zero_fill",    # "none" | "zero_fill" | "subgroup"
) -> np.ndarray:
    """Return rewards with channel `cond_idx` gated on channel `gate_idx`.

    Implements the gate `r^(b) counts only if r^(a)==1` from problem-statement.md
    §3, Prop 4. Three implementations the validation plan compares (§c, Sweep 2):

    - "none":      identity. Baseline for the bias law (Prop 4).
    - "zero_fill": r^(b) ← r^(b) * 1[r^(a)==1], centered over all m rollouts.
                   Keeps the U-statistic structure intact (problem-statement.md §5);
                   recommended estimator per Prop 4'.
    - "subgroup":  center r^(b) only over the passing subgroup r^(a)==1. Suffers
                   structural degeneracy when fewer than 2 rollouts pass the gate
                   (figT1 gray curve); for m=16 this matters whenever p_a ≲ 2/m.
    """
    r = rewards.copy()
    if impl == "none":
        return r
    gate = (r[:, gate_idx] == 1).astype(r.dtype)
    if impl == "zero_fill":
        r[:, cond_idx] = r[:, cond_idx] * gate         # centering happens in NA above
        return r
    if impl == "subgroup":
        m_pass = int(gate.sum())
        if m_pass < 2:
            r[:, cond_idx] = 0.0                       # degenerate; flag for Prop 4'
            return r
        sub_mean = r[gate.astype(bool), cond_idx].mean()
        r[:, cond_idx] = (r[:, cond_idx] - sub_mean) * gate
        return r
    raise ValueError(f"impl must be none|zero_fill|subgroup, got {impl!r}")


def measure_gradient_mse(per_seed_grads: np.ndarray) -> float:
    """Money-plot estimator: realized gradient-noise MSE across K seeds.

    Used in `gdpo_reproduction_plan.md` §d to produce the predicted-vs-realized
    scatter without an oracle gradient: treat the seed-mean as a proxy oracle
    and report mean squared deviation around it. Same trick `grpo_harness.py`
    uses for the synthetic checks; the only difference is the gradient now
    comes from a backward pass on the policy network.
    """
    assert per_seed_grads.ndim >= 2  # (K, ...)
    proxy_oracle = per_seed_grads.mean(axis=0, keepdims=True)
    return float(((per_seed_grads - proxy_oracle) ** 2).mean())
