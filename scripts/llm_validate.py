"""
Thm 3 validation entry point.

Usage:
  # validate the analysis pipeline without any GPU (proves the code reproduces
  # Thm 3 to 3 digits on synthetic Bernoulli rewards with known correlation)
  uv run scripts/llm_validate.py --mode mock

  # real GSM8K validation on Qwen2.5-1.5B-Instruct (requires GPU + transformers)
  uv run scripts/llm_validate.py --mode gsm8k --n-prompts 100 --K 8 \
      --m-grid 4 8 16 32

The mock mode is the gating experiment: if it does not reproduce Thm 3
to ≈1% precision on synthetic data, do NOT spend GPU money — there is a bug
in the analysis pipeline that real LLM data will not fix.

The real mode produces three artifacts in `figures/`:
  - llm_money_scatter_m{m}.png : predicted-vs-realized MSE per (prompt, m)
  - llm_influence_law.png      : Prop 1 influence ratio AN vs NA
  - llm_mse_vs_m.png           : MSE-vs-m scaling overlay
plus a JSON summary `figures/llm_summary.json` with all aggregate numbers.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# allow running from project root or from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_generation import (
    Backend, MockBackend, QwenBackend, run_corpus, pack_for_analysis,
    save_corpus_npz, save_corpus_generations_jsonl,
)
from llm_analysis import analyze, plot_money_scatter, summary_print


# ---------------------------------------------------------------------------
# mock-mode prompt / difficulty synthesis
# ---------------------------------------------------------------------------
def make_mock_corpus(n_prompts: int, rng_seed: int = 0):
    """Generate fake prompts with varied p_a (correctness rate) so we hit
    the realistic accuracy-vs-length regime that Thm 3 cares about."""
    rng = np.random.default_rng(rng_seed)
    p_a = rng.uniform(0.15, 0.85, size=n_prompts)
    prompts = [f"mock_problem_{i:04d}" for i in range(n_prompts)]
    golds = [f"{int(rng.integers(1, 1000))}" for _ in range(n_prompts)]
    difficulty = {p: float(pa) for p, pa in zip(prompts, p_a)}
    return list(zip(prompts, golds)), difficulty


# ---------------------------------------------------------------------------
# GSM8K loader (real mode)
# ---------------------------------------------------------------------------
def load_gsm8k(n_prompts: int, split: str = "test") -> list[tuple[str, str]]:
    """Pull `n_prompts` examples from GSM8K. Lazy import of `datasets`."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    ds = ds.shuffle(seed=0).select(range(n_prompts))
    out = []
    for ex in ds:
        # gold final answer comes after the '####' marker in GSM8K
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        out.append((ex["question"], gold))
    return out


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def run_validation(backend: Backend, prompts_with_gold: list[tuple[str, str]],
                   m_grid: list[int], K_seeds: int, w: np.ndarray,
                   out_dir: Path, label: str,
                   subsample_from_max: bool = False) -> dict:
    """Generate, analyze, plot, summarize."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== running corpus: {len(prompts_with_gold)} prompts × "
          f"{len(m_grid)} m's × {K_seeds} seeds ({label}, "
          f"subsample={subsample_from_max}) ===")
    corpus = run_corpus(backend, prompts_with_gold, m_grid, K_seeds,
                       subsample_from_max=subsample_from_max)
    print(f"  generated {len(corpus.rollouts)} groups total")

    per_m = {}
    for m in m_grid:
        print(f"\n-- analyzing m={m} --")
        tensor = pack_for_analysis(corpus, m=m)  # (P, K, m, R)
        result = analyze(tensor, w, score_seed=12345)
        summary_print(result, label=f"({label}, m={m})")
        plot_path = out_dir / f"llm_money_scatter_{label}_m{m}.png"
        fit = plot_money_scatter(result, str(plot_path),
                                  title_suffix=f" [{label}]")
        per_m[m] = {
            "predicted_pooled": result.predicted_pooled,
            "realized_na_mean": float(result.realized_na.mean()),
            "realized_an_mean": float(result.realized_an.mean()),
            "influence_an": result.influence_an.tolist(),
            "influence_na": result.influence_na.tolist(),
            "pooled_C": result.pooled_C.tolist(),
            "log_log_fit": fit,
            "plot": str(plot_path),
        }
        print(f"  saved {plot_path}")

    # cross-m scaling overlay
    overlay_path = out_dir / f"llm_mse_vs_m_{label}.png"
    _plot_mse_vs_m(per_m, w, str(overlay_path), label)
    print(f"\nsaved cross-m overlay: {overlay_path}")

    summary = {
        "label": label,
        "backend": backend.name,
        "n_prompts": len(prompts_with_gold),
        "K_seeds": K_seeds,
        "w": w.tolist(),
        "m_grid": m_grid,
        "per_m": per_m,
    }
    summary_path = out_dir / f"llm_summary_{label}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved summary: {summary_path}")

    # raw rewards + raw_length tensors for downstream γ sweep (Prop 4)
    rewards_path = out_dir / f"llm_rewards_{label}.npz"
    save_corpus_npz(corpus, str(rewards_path))
    print(f"saved raw rewards npz: {rewards_path}")

    # full per-rollout generations text for the published dataset
    generations_path = out_dir / f"llm_generations_{label}.jsonl"
    save_corpus_generations_jsonl(corpus, str(generations_path),
                                  prompts_with_gold=prompts_with_gold)
    print(f"saved full generations jsonl: {generations_path}")

    return summary


def _plot_mse_vs_m(per_m: dict, w: np.ndarray, out_path: str, label: str):
    import matplotlib.pyplot as plt
    ms = sorted(per_m.keys())
    pred = [per_m[m]["predicted_pooled"] for m in ms]
    rna = [per_m[m]["realized_na_mean"] for m in ms]
    ran = [per_m[m]["realized_an_mean"] for m in ms]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ms, pred, 'k--', label="Thm 3 prediction  $(1/m)\\, w^{\\top}\\hat C w$")
    ax.loglog(ms, rna, 'o-', color='C0', label="realized NA (mean across prompts)")
    ax.loglog(ms, ran, 's-', color='C3', label="realized AN")
    ax.set_xlabel("group size $m$"); ax.set_ylabel("MSE")
    ax.set_title(f"Thm 3 MSE vs $m$ — predicted (k) vs realized (NA, AN)  [{label}]")
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Thm 3 LLM validation")
    ap.add_argument("--mode", choices=["mock", "gsm8k"], default="mock")
    ap.add_argument("--n-prompts", type=int, default=80)
    ap.add_argument("--K", type=int, default=8, dest="K")
    ap.add_argument("--m-grid", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--rho-c-length", type=float, default=0.5,
                    help="(mock only) correlation between correctness and length channels")
    ap.add_argument("--rho-c-format", type=float, default=0.3,
                    help="(mock only) correlation between correctness and format channels")
    ap.add_argument("--w", type=float, nargs=3, default=[1.0, 1.0, 0.5],
                    help="reward weights (correctness, length, format)")
    ap.add_argument("--out-dir", type=str, default="figures")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="(gsm8k only) HuggingFace model name")
    ap.add_argument("--subsample-from-max", action="store_true",
                    help="Generate once per (prompt, seed) at m_max and slice "
                         "leading m for smaller m_grid values. ~4× cheaper on "
                         "real LLM backends; statistically equivalent because "
                         "a contiguous prefix of an i.i.d. group is i.i.d.")
    args = ap.parse_args()

    w = np.array(args.w, dtype=float)
    out_dir = Path(args.out_dir)

    if args.mode == "mock":
        prompts_with_gold, difficulty = make_mock_corpus(args.n_prompts)
        C = np.array([
            [1.0,             args.rho_c_length, args.rho_c_format],
            [args.rho_c_length, 1.0,             0.2],
            [args.rho_c_format, 0.2,             1.0],
        ])
        backend = MockBackend(C=C, difficulty_lookup=difficulty)
        run_validation(backend, prompts_with_gold, args.m_grid, args.K,
                       w, out_dir, label="mock",
                       subsample_from_max=args.subsample_from_max)
    elif args.mode == "gsm8k":
        prompts_with_gold = load_gsm8k(args.n_prompts)
        backend = QwenBackend(model_name=args.model)
        run_validation(backend, prompts_with_gold, args.m_grid, args.K,
                       w, out_dir, label="gsm8k",
                       subsample_from_max=args.subsample_from_max)


if __name__ == "__main__":
    main()
