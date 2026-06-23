"""
Theorem 3 verification: turn a ``(P, K, m, R)`` reward tensor into the
predicted-vs-realized gradient-noise MSE scatter (the paper's "money plot").

For each prompt ``p`` and seed ``k`` we form the self-normalized NA advantage on
the group of ``m`` rollouts, then a gradient estimator under a random
unit-variance score ``g ~ N(0, 1)``::

    A_j^{NA} = sum_l w_l (r_{j,l} - rbar_l) / sigmahat_l      (per group)
    ghat_k   = (1/m) sum_j A_j^{NA} * g_{j,k}                  (per seed)

The realized MSE for prompt ``p`` is ``Var_k(ghat_k)`` across ``K`` seeds. The
Theorem 3 prediction at score variance ``tau^2 = 1`` is ``(1/m) w^T Chat w``,
with ``Chat`` the sample reward correlation (per-prompt or pooled).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def advantage_AN(rewards: np.ndarray, w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """AN advantage. ``rewards`` (m, R) -> (m,)."""
    s = rewards @ w
    return (s - s.mean()) / (s.std(ddof=0) + eps)


def advantage_NA(rewards: np.ndarray, w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """NA (decoupled) advantage. ``rewards`` (m, R) -> (m,)."""
    mu = rewards.mean(axis=0, keepdims=True)
    sd = rewards.std(axis=0, ddof=0, keepdims=True) + eps
    z = (rewards - mu) / sd
    return z @ w


def gradient_per_seed(
    rewards_pkmr: np.ndarray, w: np.ndarray, estimator: str, score_seed: int = 0
) -> np.ndarray:
    """For each ``(p, k)`` compute ``ghat = (1/m) sum_j A_j g_j``.

    The random score ``g`` is drawn once (deterministic by ``score_seed``) so
    AN and NA are compared on the *same* scores. Returns ``(P, K)``.
    """
    P, K, m, R = rewards_pkmr.shape
    rng = np.random.default_rng(score_seed)
    g = rng.standard_normal((P, K, m))  # tau^2 = 1
    A = np.zeros((P, K, m))
    advantage = advantage_AN if estimator == "AN" else advantage_NA
    for p in range(P):
        for k in range(K):
            A[p, k] = advantage(rewards_pkmr[p, k], w)
    return (A * g).mean(axis=2)


def realized_mse(ghat_pk: np.ndarray) -> np.ndarray:
    """Realized MSE per prompt = ``Var_k(ghat)`` across seeds. (P,)."""
    return ghat_pk.var(axis=1, ddof=0)


def sample_correlation(rewards_mr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Within-group sample correlation matrix ``Chat`` in R^{RxR}."""
    rc = rewards_mr - rewards_mr.mean(axis=0, keepdims=True)
    cov = (rc.T @ rc) / rewards_mr.shape[0]
    sd = np.sqrt(np.diag(cov) + eps)
    return cov / np.outer(sd, sd)


def predicted_mse_per_prompt(rewards_pkmr: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-prompt prediction ``(1/m) w^T E[Chat] w`` (E[Chat] averaged over K seeds)."""
    P, K, m, R = rewards_pkmr.shape
    out = np.zeros(P)
    for p in range(P):
        Cbar = np.mean([sample_correlation(rewards_pkmr[p, k]) for k in range(K)], axis=0)
        out[p] = (w @ Cbar @ w) / m
    return out


def predicted_mse_pooled(rewards_pkmr: np.ndarray, w: np.ndarray) -> float:
    """Single pooled prediction using the corpus-pooled correlation."""
    P, K, m, R = rewards_pkmr.shape
    Chats = [sample_correlation(rewards_pkmr[p, k]) for p in range(P) for k in range(K)]
    Cbar = np.mean(Chats, axis=0)
    return float((w @ Cbar @ w) / m)


def influence_per_channel(
    rewards_pkmr: np.ndarray, w: np.ndarray, estimator: str
) -> np.ndarray:
    """Realized ``Cov(A_j, z_{j,l})`` averaged across ``(p, k)``. Returns (R,)."""
    P, K, m, R = rewards_pkmr.shape
    advantage = advantage_AN if estimator == "AN" else advantage_NA
    accum = np.zeros(R)
    n = 0
    for p in range(P):
        for k in range(K):
            r = rewards_pkmr[p, k]
            A = advantage(r, w)
            sd = r.std(axis=0, ddof=0) + 1e-8
            z = (r - r.mean(axis=0)) / sd
            for ell in range(R):
                accum[ell] += float(((A - A.mean()) * (z[:, ell] - z[:, ell].mean())).mean())
            n += 1
    return accum / n


@dataclass
class Thm3Result:
    m: int
    P: int
    K: int
    w: np.ndarray
    realized_an: np.ndarray
    realized_na: np.ndarray
    predicted_per_prompt: np.ndarray
    predicted_pooled: float
    influence_an: np.ndarray
    influence_na: np.ndarray
    pooled_C: np.ndarray


def analyze(rewards_pkmr: np.ndarray, w: np.ndarray, score_seed: int = 0) -> Thm3Result:
    """End-to-end Theorem 3 analysis: realized vs predicted MSE, plus side-checks."""
    rewards_pkmr = np.asarray(rewards_pkmr, dtype=float)
    w = np.asarray(w, dtype=float)
    P, K, m, R = rewards_pkmr.shape
    if w.shape != (R,):
        raise ValueError(f"w must have shape ({R},) but got {w.shape}")
    g_an = gradient_per_seed(rewards_pkmr, w, "AN", score_seed)
    g_na = gradient_per_seed(rewards_pkmr, w, "NA", score_seed)
    Chats = [sample_correlation(rewards_pkmr[p, k]) for p in range(P) for k in range(K)]
    return Thm3Result(
        m=m, P=P, K=K, w=w,
        realized_an=realized_mse(g_an), realized_na=realized_mse(g_na),
        predicted_per_prompt=predicted_mse_per_prompt(rewards_pkmr, w),
        predicted_pooled=predicted_mse_pooled(rewards_pkmr, w),
        influence_an=influence_per_channel(rewards_pkmr, w, "AN"),
        influence_na=influence_per_channel(rewards_pkmr, w, "NA"),
        pooled_C=np.mean(Chats, axis=0),
    )


def plot_money_scatter(result: Thm3Result, out_path: str, title_suffix: str = "") -> dict:
    """Predicted-vs-realized scatter (NA blue, AN red, ``y=x`` reference).

    Returns the slope and R^2 of ``log(realized)`` vs ``log(predicted)`` for NA.
    Requires ``matplotlib`` (imported lazily).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    pred = result.predicted_per_prompt
    mask = (pred > 0) & (result.realized_na > 0) & (result.realized_an > 0)
    p, rna, ran = pred[mask], result.realized_na[mask], result.realized_an[mask]

    ax.loglog(p, rna, "o", color="C0", alpha=0.7, label="NA (decoupled)")
    ax.loglog(p, ran, "s", color="C3", alpha=0.6, label="AN (scalarize-then-norm)")
    lo, hi = min(p.min(), rna.min(), ran.min()), max(p.max(), rna.max(), ran.max())
    ax.loglog([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="y = x")

    logp, logr = np.log(p), np.log(rna)
    if len(logp) >= 3:
        slope, intercept = np.polyfit(logp, logr, 1)
        ss_res = np.sum((logr - (slope * logp + intercept)) ** 2)
        ss_tot = np.sum((logr - logr.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
    else:
        slope, r2 = float("nan"), float("nan")

    ax.set_xlabel("predicted MSE  (Thm 3:  $(1/m)\\, w^{\\top}\\hat C_p w$)")
    ax.set_ylabel("realized MSE  (Var across $K$ seeds)")
    ax.set_title(
        f"Thm 3 money plot (m={result.m}, P={result.P}, K={result.K}){title_suffix}\n"
        f"NA log-log fit:  slope={slope:.3f}, $R^2$={r2:.3f}"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return {
        "slope_na": slope, "r2_na": r2,
        "median_realized_na": float(np.median(rna)),
        "median_predicted": float(np.median(p)),
        "median_ratio_na": float(np.median(rna / p)),
    }


def summary_print(result: Thm3Result, label: str = "") -> None:
    print(f"=== Thm 3 analysis {label} (m={result.m}, P={result.P}, K={result.K}) ===")
    print(f"  w = {result.w}")
    print("  pooled reward correlation matrix:")
    for row in result.pooled_C:
        print("    " + "  ".join(f"{x:+.3f}" for x in row))
    pooled_pred = result.predicted_pooled
    print(f"  predicted (pooled):  {pooled_pred:.5f}")
    print(f"  realized NA (mean):  {result.realized_na.mean():.5f}   "
          f"ratio={result.realized_na.mean()/pooled_pred:.3f}")
    print(f"  realized AN (mean):  {result.realized_an.mean():.5f}   "
          f"ratio={result.realized_an.mean()/pooled_pred:.3f}")
    print(f"  influence per channel — AN: {result.influence_an}")
    print(f"  influence per channel — NA: {result.influence_na}")


__all__ = [
    "advantage_AN", "advantage_NA",
    "gradient_per_seed", "realized_mse",
    "sample_correlation", "predicted_mse_per_prompt", "predicted_mse_pooled",
    "influence_per_channel",
    "Thm3Result", "analyze", "plot_money_scatter", "summary_print",
]
