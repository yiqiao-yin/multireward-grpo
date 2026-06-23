"""
Plot GRPO training curves aggregated across seeds.

Discovers all `figures/grpo_train/{mode}_seed{N}/metrics.json` files,
groups by mode, and produces a learning-curve figure with mean line ±
seed-std shaded band per mode. Optional held-out eval bars at the right.

Usage:
    uv run scripts/plot_grpo_training_multiseed.py
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO_ROOT / "figures" / "grpo_train"


COLORS = {
    "na":     "C0",  # paper's proposal — blue
    "an":     "C3",  # baseline — red
    "single": "C2",  # ignore-multi-reward baseline — green
}
NICE_LABELS = {
    "na":     "NA (paper)",
    "an":     "AN (baseline)",
    "single": "single-reward (correctness only)",
}


def discover_runs() -> dict[str, list[Path]]:
    """Return {mode: [metric_path, ...]}."""
    pat = re.compile(r"^(?P<mode>na|an|single)_seed(?P<seed>\d+)$")
    by_mode: dict[str, list[Path]] = defaultdict(list)
    for d in TRAIN_DIR.iterdir() if TRAIN_DIR.exists() else []:
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if m and (d / "metrics.json").exists():
            by_mode[m.group("mode")].append(d / "metrics.json")
    for mode in by_mode:
        by_mode[mode].sort()
    return dict(by_mode)


def load_metrics(p: Path) -> dict:
    with open(p) as f:
        d = json.load(f)
    return d


def smooth(xs: np.ndarray, k: int) -> np.ndarray:
    if len(xs) < k or k <= 1:
        return xs
    return np.convolve(xs, np.ones(k) / k, mode="valid")


def collect_curves(metric_files: list[Path], key: str, smoothing: int):
    """Return (steps, list_of_seed_curves)."""
    curves = []
    for p in metric_files:
        d = load_metrics(p)
        vals = np.array([row[key] for row in d["history"]])
        sm = smooth(vals, smoothing)
        steps = np.arange(smoothing - 1, smoothing - 1 + len(sm)) if smoothing > 1 else np.arange(len(sm))
        curves.append((steps, sm))
    return curves


def plot_panel(ax, by_mode: dict, metric_key: str, title: str, ylabel: str,
               smoothing: int):
    for mode, files in by_mode.items():
        curves = collect_curves(files, metric_key, smoothing)
        if not curves:
            continue
        # align by truncating to the shortest run
        min_len = min(len(c[1]) for c in curves)
        steps = curves[0][0][:min_len]
        mat = np.stack([c[1][:min_len] for c in curves], axis=0)  # (n_seeds, T)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        c = COLORS.get(mode, "gray")
        label = f"{NICE_LABELS.get(mode, mode)} (n={len(files)})"
        ax.plot(steps, mean, color=c, lw=2.0, label=label)
        ax.fill_between(steps, mean - std, mean + std, color=c, alpha=0.18)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoothing", type=int, default=15)
    ap.add_argument("--out", type=str,
                    default="figures/grpo_train/grpo_training_curves_multiseed.png")
    args = ap.parse_args()

    by_mode = discover_runs()
    if not by_mode:
        sys.exit("no runs found in figures/grpo_train/{mode}_seed{N}/metrics.json")
    print(f"discovered runs: { {m: len(v) for m, v in by_mode.items()} }")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_panel(axes[0, 0], by_mode, "mean_aggregate_reward",
               "Mean aggregate reward  $w^T r$",
               "reward", args.smoothing)
    plot_panel(axes[0, 1], by_mode, "mean_compliance",
               "Compliance (gate channel)",
               "reward", args.smoothing)
    plot_panel(axes[1, 0], by_mode, "mean_politeness_gated",
               "Politeness (gated by compliance)",
               "reward", args.smoothing)
    plot_panel(axes[1, 1], by_mode, "kl",
               "KL anchor to base policy",
               "kl", args.smoothing)
    fig.suptitle(
        f"GRPO training: NA (paper) vs AN (baseline) vs single-reward "
        f"(correctness only)\n[mean ± seed std, smoothing={args.smoothing} steps]",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved {out_path}")

    # also print summary table: last-50 mean and std per mode
    print("\n=== last-50-step aggregate reward ===")
    for mode, files in by_mode.items():
        last50_means_across_seeds = []
        for p in files:
            d = load_metrics(p)
            last50 = [r["mean_aggregate_reward"] for r in d["history"][-50:]]
            last50_means_across_seeds.append(np.mean(last50))
        m = np.mean(last50_means_across_seeds)
        s = np.std(last50_means_across_seeds)
        print(f"  {NICE_LABELS.get(mode, mode):40s}: {m:+.4f} ± {s:.4f}  (n={len(files)})")


if __name__ == "__main__":
    main()
