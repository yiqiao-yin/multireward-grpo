"""
Prop 4 verification on real LLM rewards (post-processing of a saved corpus).

Approach: we have real (correctness, raw_length, format) observations per
rollout from `llm_validate.py --mode gsm8k`. For each γ in [0, 3] we
construct a synthetic gradient direction with the exact structure Prop 4
assumes,

    g_j = α_c (c_j - p_a)
        + α_f (l_j - p_b) c_j
        + γ  (l_j - p_b) (1 - c_j)
        + ε_j,        ε_j ~ N(0, σ²) indep.

evaluated at the *real* (c_j, l_j) values from Qwen on GSM8K, with l_j the
median-binarized raw_length channel (so the Bernoulli-style p_b assumption
in Prop 4's closed form applies). For each γ we measure the empirical
"bias removed by conditioning",

    β_observed = w_b · [ Cov(l, g) - Cov(l·c, g) ]

averaged within prompt then across prompts, and compare against the closed
form proved in `proofs.md`,

    β_predicted = w_b · p_b (1 - p_a) [ γ (1 - p_b) - α_c p_a ].

Acceptance: β_observed tracks β_predicted across γ with the sign change at
γ* = α_c p_a / (1 - p_b). This is the real-data analog of `figT2`.

Output:
  figures/<dir>/llm_prop4_bias_law_real.png   — bias-vs-γ overlay
  figures/<dir>/llm_prop4_summary.json        — per-γ observed/predicted numbers
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_corpus_npz(path: str):
    """Return rewards_pkmr (P, K, m, R), raw_length_pkm (P, K, m), meta."""
    z = np.load(path, allow_pickle=True)
    m_grid = list(z["m_grid"].tolist())
    K = int(z["K_seeds"])
    n_prompts = int(z["n_prompts"])
    reward_names = list(z["reward_names"].tolist())
    backend_name = str(z["backend_name"])
    # pick the largest m (most data per group)
    m = max(m_grid)
    rewards = z[f"rewards_m{m}"]
    raw_l = z[f"raw_length_m{m}"]
    return rewards, raw_l, {
        "m": m, "K": K, "n_prompts": n_prompts,
        "reward_names": reward_names, "backend_name": backend_name,
    }


def synthesize_g(c: np.ndarray, l_bin: np.ndarray,
                 p_a: float, p_b: float,
                 alpha_c: float, alpha_f: float, gamma: float,
                 sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Return per-rollout synthetic gradient g_j matching Prop 4's model.

    c, l_bin have shape (m,) (Bernoulli-style). g has shape (m,).
    """
    eps = sigma * rng.standard_normal(c.shape)
    return (
        alpha_c * (c - p_a)
        + alpha_f * (l_bin - p_b) * c
        + gamma  * (l_bin - p_b) * (1.0 - c)
        + eps
    )


def empirical_beta_per_prompt(c_pkm: np.ndarray, l_pkm: np.ndarray,
                              alpha_c: float, alpha_f: float, gamma: float,
                              sigma: float, w_b: float,
                              rng: np.random.Generator,
                              binarize_per_prompt: bool = True
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute observed and predicted β per prompt, plus (p_a, p_b) per prompt.

    Returns (beta_obs_p, beta_pred_p, (p_a_p, p_b_p)) all of length P.
    """
    P, K, m = c_pkm.shape
    # Binarize length per-prompt at its in-prompt median (so p_b ≈ 0.5 per
    # prompt) — keeps the Bern-style assumption local and prevents per-prompt
    # heterogeneity from confounding the sweep.
    if binarize_per_prompt:
        thresh = np.median(l_pkm.reshape(P, -1), axis=1, keepdims=True)  # (P, 1)
        l_bin_pkm = (l_pkm.reshape(P, -1) > thresh).astype(float).reshape(P, K, m)
    else:
        l_bin_pkm = (l_pkm > 0).astype(float)

    beta_obs_p = np.zeros(P)
    beta_pred_p = np.zeros(P)
    p_a_p = np.zeros(P)
    p_b_p = np.zeros(P)

    for p in range(P):
        c_flat = c_pkm[p].reshape(-1)
        l_flat = l_bin_pkm[p].reshape(-1)
        p_a = float(c_flat.mean())
        p_b = float(l_flat.mean())
        p_a_p[p] = p_a
        p_b_p[p] = p_b

        g = synthesize_g(c_flat, l_flat, p_a, p_b,
                         alpha_c, alpha_f, gamma, sigma, rng)

        # empirical covariances (population form: N divisor)
        cov_l_g = float(((l_flat - p_b) * (g - g.mean())).mean())
        cov_lc_g = float(((l_flat * c_flat - (l_flat * c_flat).mean()) * (g - g.mean())).mean())
        beta_obs_p[p] = w_b * (cov_l_g - cov_lc_g)

        # closed-form prediction from proofs.md Prop 4
        beta_pred_p[p] = w_b * p_b * (1.0 - p_a) * (gamma * (1.0 - p_b) - alpha_c * p_a)

    return beta_obs_p, beta_pred_p, (p_a_p, p_b_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewards-npz", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default="figures")
    ap.add_argument("--label", type=str, default="real",
                    help="suffix for output filenames")
    ap.add_argument("--gamma-grid", type=float, nargs="+",
                    default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25,
                             1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
    ap.add_argument("--alpha-c", type=float, default=1.0)
    ap.add_argument("--alpha-f", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.5,
                    help="std of independent noise ε in synthetic g")
    ap.add_argument("--w-b", type=float, default=1.0,
                    help="weight on the conditioned channel in β")
    ap.add_argument("--n-replicates", type=int, default=64,
                    help="number of synthetic g draws per γ to average over")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rewards_pkmr, raw_l_pkm, meta = load_corpus_npz(args.rewards_npz)
    P, K, m, R = rewards_pkmr.shape
    print(f"loaded corpus: P={P}, K={K}, m={m}, R={R}")
    print(f"reward channels: {meta['reward_names']}")

    # correctness is channel 0
    c_pkm = rewards_pkmr[..., 0]
    raw_l_pkm = raw_l_pkm  # already (P, K, m)

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_gamma = []
    for gamma in args.gamma_grid:
        # average over n_replicates draws of the noise ε in synthetic g —
        # this isolates the systematic bias from sampling noise
        obs_acc = np.zeros(P)
        pa_acc = np.zeros(P)
        pb_acc = np.zeros(P)
        pred_acc = np.zeros(P)
        for _ in range(args.n_replicates):
            obs, pred, (pa, pb) = empirical_beta_per_prompt(
                c_pkm, raw_l_pkm, args.alpha_c, args.alpha_f, gamma,
                args.sigma, args.w_b, rng,
            )
            obs_acc += obs
            pred_acc += pred
            pa_acc += pa
            pb_acc += pb
        obs_acc /= args.n_replicates
        pred_acc /= args.n_replicates
        pa_acc /= args.n_replicates
        pb_acc /= args.n_replicates
        per_gamma.append({
            "gamma": gamma,
            "obs_mean": float(obs_acc.mean()),
            "obs_sem": float(obs_acc.std() / np.sqrt(P)),
            "pred_mean": float(pred_acc.mean()),
            "p_a_mean": float(pa_acc.mean()),
            "p_b_mean": float(pb_acc.mean()),
        })
        print(f"  γ={gamma:.2f}  obs={obs_acc.mean():+.5f}±{obs_acc.std()/np.sqrt(P):.5f}  "
              f"pred={pred_acc.mean():+.5f}  p_a={pa_acc.mean():.3f}  p_b={pb_acc.mean():.3f}")

    # ----------- plot ------------
    gammas = np.array([d["gamma"] for d in per_gamma])
    obs = np.array([d["obs_mean"] for d in per_gamma])
    obs_sem = np.array([d["obs_sem"] for d in per_gamma])
    pred = np.array([d["pred_mean"] for d in per_gamma])
    # sign-change γ* from the average p_a, p_b
    p_a_avg = np.mean([d["p_a_mean"] for d in per_gamma])
    p_b_avg = np.mean([d["p_b_mean"] for d in per_gamma])
    gamma_star = args.alpha_c * p_a_avg / max(1.0 - p_b_avg, 1e-6)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(0, color="gray", lw=0.8)
    ax.errorbar(gammas, obs, yerr=obs_sem, fmt="o", color="C0",
                label=r"$\beta_{ab}$ observed (real $c$, $\ell$; synth $g$)",
                capsize=3)
    ax.plot(gammas, pred, "-", color="C1",
            label=r"$\beta_{ab}$ predicted $= w_b\, p_b(1-p_a)[\gamma(1-p_b)-\alpha_c p_a]$")
    ax.axvline(gamma_star, ls=":", color="C3",
               label=fr"$\gamma^\star = \alpha_c p_a/(1-p_b) \approx {gamma_star:.2f}$")
    ax.set_xlabel(r"contamination $\gamma$")
    ax.set_ylabel(r"$\beta_{ab}$  (bias removed by conditioning)")
    ax.set_title(
        f"Prop 4 bias law on real GSM8K rewards "
        f"(P={P}, m={m}, K={K}, $\\bar p_a$={p_a_avg:.2f}, $\\bar p_b$={p_b_avg:.2f})"
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = out_dir / f"llm_prop4_bias_law_{args.label}.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"\nsaved figure: {out_png}")

    # summary json
    summary = {
        "n_prompts": P, "K": K, "m": m,
        "alpha_c": args.alpha_c, "alpha_f": args.alpha_f, "sigma": args.sigma,
        "w_b": args.w_b, "n_replicates": args.n_replicates,
        "p_a_mean": float(p_a_avg), "p_b_mean": float(p_b_avg),
        "gamma_star": float(gamma_star),
        "per_gamma": per_gamma,
    }
    out_json = out_dir / f"llm_prop4_summary_{args.label}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved summary: {out_json}")


if __name__ == "__main__":
    main()
