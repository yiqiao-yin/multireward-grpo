"""
Plot training curves from a GRPO run's metrics.json.

Loads figures/grpo_train/{mode}/metrics.json and produces:
  figures/grpo_train/grpo_training_curves_{mode}.png

Usage:
    uv run scripts/plot_grpo_training.py --modes na an
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def load(mode: str) -> dict | None:
    p = REPO_ROOT / "figures" / "grpo_train" / mode / "metrics.json"
    if not p.exists():
        print(f"  missing {p}")
        return None
    with open(p) as f:
        return json.load(f)


def smooth(xs: np.ndarray, k: int = 10) -> np.ndarray:
    if len(xs) < k:
        return xs
    ker = np.ones(k) / k
    return np.convolve(xs, ker, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["na", "an"])
    ap.add_argument("--smoothing", type=int, default=10)
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {"na": "C0", "an": "C3"}

    for mode in args.modes:
        d = load(mode)
        if d is None:
            continue
        h = d["history"]
        steps = np.array([row["step"] for row in h])
        loss = np.array([row["loss"] for row in h])
        pg = np.array([row["pg_loss"] for row in h])
        kl = np.array([row["kl"] for row in h])
        agg_reward = np.array([row["mean_aggregate_reward"] for row in h])
        comp = np.array([row["mean_compliance"] for row in h])
        pol = np.array([row["mean_politeness_gated"] for row in h])
        act = np.array([row["mean_action"] for row in h])

        s = args.smoothing
        x_s = steps[s-1:] if len(steps) >= s else steps

        c = colors.get(mode, "C2")
        axes[0, 0].plot(x_s, smooth(loss, s), color=c, label=mode.upper())
        axes[0, 0].set_title("Total loss (PG + KL)")
        axes[0, 0].set_xlabel("step"); axes[0, 0].set_ylabel("loss")

        axes[0, 1].plot(x_s, smooth(agg_reward, s), color=c, label=mode.upper())
        axes[0, 1].set_title("Mean aggregate reward $w^T r$")
        axes[0, 1].set_xlabel("step"); axes[0, 1].set_ylabel("reward")

        axes[1, 0].plot(x_s, smooth(comp, s), color=c, label=f"{mode.upper()} compliance")
        axes[1, 0].plot(x_s, smooth(act, s), color=c, linestyle="--",
                       alpha=0.6, label=f"{mode.upper()} action")
        axes[1, 0].set_title("Per-channel reward (compliance + action)")
        axes[1, 0].set_xlabel("step"); axes[1, 0].set_ylabel("reward")

        axes[1, 1].plot(x_s, smooth(pol, s), color=c, label=mode.upper())
        axes[1, 1].set_title("Politeness (gated by compliance)")
        axes[1, 1].set_xlabel("step"); axes[1, 1].set_ylabel("reward")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"GRPO training — NA (paper) vs AN (baseline)  "
                 f"[smoothing={args.smoothing} steps]", fontsize=12)
    fig.tight_layout()
    out_path = REPO_ROOT / "figures" / "grpo_train" / "grpo_training_curves.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
