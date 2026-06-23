"""
Thm 3 analysis: take a (P, K, m, R) reward tensor and produce the
predicted-vs-realized MSE scatter that is the headline empirical figure.

The mapping from the math in proofs.md is direct.  For each prompt p, each
seed k, we form the self-normalized NA advantage on the group of m rollouts,
then the gradient estimator under a random unit-variance score g ~ N(0, 1):

    A_{j}^{NA}     = sum_l w_l * (r_{j,l} - r̄_l) / σ̂_l           (per group)
    ĝ_k            = (1/m) sum_j A_j^{NA} * g_{j,k}                  (per seed)

The realized MSE for prompt p is then  Var_k(ĝ_k)  across the K seeds.
The Thm 3 prediction at score variance τ²=1 is

    MSE_pred  =  (1/m) * w^T Ĉ w,         Ĉ = sample reward correlation

evaluated either (a) per-prompt using that prompt's Ĉ_p, or (b) pooled across
prompts using the corpus Ĉ. The pipeline reports both. The boxed identity in
proofs.md is the per-prompt (a) version; the slope-and-R² of realized-vs-
predicted on the scatter measures how well the theory holds in practice.

The same tensor also yields:
  - the AN-vs-NA influence-ratio check for Prop 1 (cheap side-analysis)
  - the AN MSE for comparison (always larger when reward σ's differ)
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Sequence


# ---------------------------------------------------------------------------
# core advantage estimators (oracle + self-normalized, AN + NA)
# ---------------------------------------------------------------------------
def advantage_AN(rewards: np.ndarray, w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """AN advantage. rewards (m, R) -> (m,) advantage."""
    s = rewards @ w
    return (s - s.mean()) / (s.std(ddof=0) + eps)


def advantage_NA(rewards: np.ndarray, w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """NA (decoupled) advantage. rewards (m, R) -> (m,) advantage."""
    mu = rewards.mean(axis=0, keepdims=True)
    sd = rewards.std(axis=0, ddof=0, keepdims=True) + eps
    z = (rewards - mu) / sd
    return z @ w


# ---------------------------------------------------------------------------
# realized MSE via multi-seed variance
# ---------------------------------------------------------------------------
def gradient_per_seed(rewards_pkmr: np.ndarray, w: np.ndarray, estimator: str,
                     score_seed: int = 0) -> np.ndarray:
    """For each (p, k), compute ĝ = (1/m) sum_j A_j g_j.

    rewards_pkmr: (P, K, m, R)
    Returns (P, K) gradients.

    The random score g is drawn once for the whole tensor (deterministic by
    seed) so the comparison across estimators is paired — same scores used
    for both AN and NA, only the advantage formula differs.
    """
    P, K, m, R = rewards_pkmr.shape
    rng = np.random.default_rng(score_seed)
    g = rng.standard_normal((P, K, m))  # τ²=1
    A = np.zeros((P, K, m))
    advantage = advantage_AN if estimator == "AN" else advantage_NA
    for p in range(P):
        for k in range(K):
            A[p, k] = advantage(rewards_pkmr[p, k], w)
    ghat = (A * g).mean(axis=2)  # (P, K)
    return ghat


def realized_mse(ghat_pk: np.ndarray) -> np.ndarray:
    """Realized MSE per prompt = Var_k(ĝ) across seeds. (P,)"""
    return ghat_pk.var(axis=1, ddof=0)


# ---------------------------------------------------------------------------
# Thm 3 prediction
# ---------------------------------------------------------------------------
def sample_correlation(rewards_mr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Within-group sample correlation matrix Ĉ ∈ R^{R×R}."""
    rc = rewards_mr - rewards_mr.mean(axis=0, keepdims=True)
    cov = (rc.T @ rc) / rewards_mr.shape[0]
    sd = np.sqrt(np.diag(cov) + eps)
    return cov / np.outer(sd, sd)


def predicted_mse_per_prompt(rewards_pkmr: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-prompt prediction (1/m) w^T E[Ĉ] w, with E[Ĉ] estimated by
    averaging Ĉ across the K seeds for each prompt."""
    P, K, m, R = rewards_pkmr.shape
    out = np.zeros(P)
    for p in range(P):
        Chats = [sample_correlation(rewards_pkmr[p, k]) for k in range(K)]
        Cbar = np.mean(Chats, axis=0)
        out[p] = (w @ Cbar @ w) / m
    return out


def predicted_mse_pooled(rewards_pkmr: np.ndarray, w: np.ndarray) -> float:
    """Single pooled prediction using the corpus-pooled correlation."""
    P, K, m, R = rewards_pkmr.shape
    Chats = []
    for p in range(P):
        for k in range(K):
            Chats.append(sample_correlation(rewards_pkmr[p, k]))
    Cbar = np.mean(Chats, axis=0)
    return float((w @ Cbar @ w) / m)


# ---------------------------------------------------------------------------
# Prop 1 side-analysis: realized influence ratio (per channel)
# ---------------------------------------------------------------------------
def influence_per_channel(rewards_pkmr: np.ndarray, w: np.ndarray,
                          estimator: str) -> np.ndarray:
    """Realized Cov(A_j, z_{j,l}) averaged across (p, k). Returns (R,)."""
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
                # population covariance within the group
                accum[ell] += float(((A - A.mean()) * (z[:, ell] - z[:, ell].mean())).mean())
            n += 1
    return accum / n


# ---------------------------------------------------------------------------
# the money plot
# ---------------------------------------------------------------------------
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
    """End-to-end analysis: realized vs predicted MSE, plus side-checks."""
    P, K, m, R = rewards_pkmr.shape
    assert w.shape == (R,), f"w must have shape ({R},) but got {w.shape}"
    g_an = gradient_per_seed(rewards_pkmr, w, "AN", score_seed)
    g_na = gradient_per_seed(rewards_pkmr, w, "NA", score_seed)
    realized_an = realized_mse(g_an)
    realized_na = realized_mse(g_na)
    pred_pp = predicted_mse_per_prompt(rewards_pkmr, w)
    pred_pool = predicted_mse_pooled(rewards_pkmr, w)
    infl_an = influence_per_channel(rewards_pkmr, w, "AN")
    infl_na = influence_per_channel(rewards_pkmr, w, "NA")
    # pooled C for diagnostics
    Chats = [sample_correlation(rewards_pkmr[p, k]) for p in range(P) for k in range(K)]
    pooled_C = np.mean(Chats, axis=0)
    return Thm3Result(
        m=m, P=P, K=K, w=w,
        realized_an=realized_an, realized_na=realized_na,
        predicted_per_prompt=pred_pp, predicted_pooled=pred_pool,
        influence_an=infl_an, influence_na=infl_na,
        pooled_C=pooled_C,
    )


def plot_money_scatter(result: Thm3Result, out_path: str,
                       title_suffix: str = "") -> dict:
    """Predicted-vs-realized scatter, NA in blue, AN in red, y=x reference.
    Returns the slope and R² of log(realized) vs log(predicted) for NA."""
    fig, ax = plt.subplots(figsize=(7, 6))
    pred = result.predicted_per_prompt
    # NA expected to follow y=x; AN expected to drift (Prop 1 contamination)
    mask = (pred > 0) & (result.realized_na > 0) & (result.realized_an > 0)
    p = pred[mask]
    rna = result.realized_na[mask]
    ran = result.realized_an[mask]

    ax.loglog(p, rna, 'o', color='C0', alpha=0.7, label="NA (decoupled)")
    ax.loglog(p, ran, 's', color='C3', alpha=0.6, label="AN (scalarize-then-norm)")
    lo, hi = min(p.min(), rna.min(), ran.min()), max(p.max(), rna.max(), ran.max())
    ax.loglog([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.6, label="y = x")

    # fit log-log slope for NA
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
    ax.set_title(f"Thm 3 money plot: predicted vs realized "
                 f"(m={result.m}, P={result.P}, K={result.K}){title_suffix}\n"
                 f"NA log-log fit:  slope={slope:.3f}, $R^2$={r2:.3f}")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return {"slope_na": slope, "r2_na": r2,
            "median_realized_na": float(np.median(rna)),
            "median_predicted": float(np.median(p)),
            "median_ratio_na": float(np.median(rna / p))}


def summary_print(result: Thm3Result, label: str = "") -> None:
    print(f"=== Thm 3 analysis {label} (m={result.m}, P={result.P}, K={result.K}) ===")
    print(f"  w = {result.w}")
    print(f"  pooled reward correlation matrix:")
    for row in result.pooled_C:
        print("    " + "  ".join(f"{x:+.3f}" for x in row))
    pooled_pred = result.predicted_pooled
    pooled_realized_na = result.realized_na.mean()
    pooled_realized_an = result.realized_an.mean()
    print(f"  predicted (pooled):  {pooled_pred:.5f}")
    print(f"  realized NA (mean):  {pooled_realized_na:.5f}   ratio={pooled_realized_na/pooled_pred:.3f}")
    print(f"  realized AN (mean):  {pooled_realized_an:.5f}   ratio={pooled_realized_an/pooled_pred:.3f}")
    print(f"  influence per channel — AN: {result.influence_an}")
    print(f"  influence per channel — NA: {result.influence_na}")


__all__ = [
    "advantage_AN", "advantage_NA",
    "gradient_per_seed", "realized_mse",
    "sample_correlation", "predicted_mse_per_prompt", "predicted_mse_pooled",
    "influence_per_channel",
    "Thm3Result", "analyze", "plot_money_scatter", "summary_print",
]
