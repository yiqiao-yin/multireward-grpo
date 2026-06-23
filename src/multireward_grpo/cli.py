"""Command-line entry point for multireward-grpo.

Subcommands:
  version       print the installed version
  thm3-check    CPU-only Theorem-3 verification on synthetic rewards (no GPU)
  train         GRPO-train the fintech example (requires the [llm] extra + GPU)
"""
from __future__ import annotations

import argparse
import sys

from . import __version__


def _cmd_version(_args) -> int:
    print(f"multireward-grpo {__version__}")
    return 0


def _cmd_thm3_check(args) -> int:
    import numpy as np

    from .analysis import analyze, summary_print
    from .generation import MockBackend, pack_for_analysis, run_corpus

    rho = args.rho
    C = np.array([[1.0, rho, 0.0], [rho, 1.0, 0.0], [0.0, 0.0, 1.0]])
    backend = MockBackend(C=C)
    prompts = [(f"prompt-{i}", "0") for i in range(args.n_prompts)]
    corpus = run_corpus(backend, prompts, m_grid=[args.m], K_seeds=args.K)
    rewards = pack_for_analysis(corpus, m=args.m)
    w = np.array(args.weights, dtype=float)
    result = analyze(rewards, w)
    summary_print(result, label=f"(mock, rho={rho})")
    ratio = result.realized_na.mean() / result.predicted_pooled
    print(f"\nNA realized/predicted ratio = {ratio:.3f} "
          f"({'OK' if 0.85 <= ratio <= 1.15 else 'OUT OF RANGE'})")
    return 0


def _cmd_train(args) -> int:
    from .examples import FintechRewardFunction, make_fintech_prompts
    from .train import GRPOConfig, GRPOTrainer

    prompts = make_fintech_prompts(args.n_scenarios, seed=args.seed)
    cfg = GRPOConfig(
        model=args.model, mode=args.mode, weights=tuple(args.weights),
        n_steps=args.n_steps, m=args.m, batch_prompts=args.batch_prompts,
        seed=args.seed, out_dir=args.out_dir,
    )
    GRPOTrainer(cfg, FintechRewardFunction(), prompts).train()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="multireward-grpo", description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("version", help="print version").set_defaults(func=_cmd_version)

    p_chk = sub.add_parser("thm3-check", help="CPU Theorem-3 check on synthetic rewards")
    p_chk.add_argument("--n-prompts", type=int, default=40)
    p_chk.add_argument("--K", type=int, default=200)
    p_chk.add_argument("--m", type=int, default=8)
    p_chk.add_argument("--rho", type=float, default=0.5)
    p_chk.add_argument("--weights", type=float, nargs="+", default=[1.0, 1.0, 0.5])
    p_chk.set_defaults(func=_cmd_thm3_check)

    p_tr = sub.add_parser("train", help="GRPO-train the fintech example (needs [llm] + GPU)")
    p_tr.add_argument("--mode", choices=["na", "an", "single"], default="na")
    p_tr.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p_tr.add_argument("--n-scenarios", type=int, default=400)
    p_tr.add_argument("--n-steps", type=int, default=200)
    p_tr.add_argument("--m", type=int, default=8)
    p_tr.add_argument("--batch-prompts", type=int, default=4)
    p_tr.add_argument("--weights", type=float, nargs="+", default=[1.0, 1.0, 0.5])
    p_tr.add_argument("--seed", type=int, default=0)
    p_tr.add_argument("--out-dir", default="grpo_out")
    p_tr.set_defaults(func=_cmd_train)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
