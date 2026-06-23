# GDPO Reproduction Plan — multi-reward GRPO validation anchor

**Status of the anchor (read first).** The package cites GDPO as
*Shih-Yang Liu, Dong et al. 2026, arXiv:2601.05242*. **This plan was prepared
without verified network access**: WebSearch, WebFetch and outbound `curl` were
all denied in the working sandbox, so the arXiv ID, the author list, the GitHub
URL and the reported GPU budget below are **unverified placeholders** carried
forward from `problem-statement.md` and `README.md`. Before executing §a
("Setup"), open `https://arxiv.org/abs/2601.05242` in a browser and confirm
title/authors. If the ID does not resolve or the paper is not about decoupled
multi-reward normalization, fall back to the **DAPO anchor** (Yu et al. 2025,
arXiv:2503.14476 — already a §6 must-cite, ships a verl recipe with decoupled
clip + zero-variance group fix and is the closest *verified* analogue of the
AN/NA contrast we need). The scaffold in `gdpo_adapter.py` is written
framework-agnostically precisely so it ports to either GDPO or DAPO with a
one-import change.

Anchor-verification checklist (do this first, ~5 min):

- [ ] `arxiv.org/abs/2601.05242` resolves and the title contains "DPO",
      "GRPO", "decoupled", or "multi-reward".
- [ ] Author list includes Shih-Yang Liu and/or "Dong" with NVIDIA affiliation.
- [ ] Paper has a public code repo (search `github.com/NVlabs`, `github.com/NVIDIA`,
      or the first author's GitHub).
- [ ] Paper shows two empirical anchors: (i) AN-vs-NA on a math+length or
      math+format multi-reward setup, (ii) a length-given-correctness
      conditioning experiment.
- [ ] If any of the above fails → switch to DAPO (arXiv:2503.14476,
      `github.com/BytedTsinghua-SIA/DAPO`) and re-label this file.

---

## (a) Setup

```bash
# 1. clone the anchor repo (replace with verified URL after the checklist above)
git clone <GDPO_REPO_URL> gdpo && cd gdpo

# 2. dependencies. GDPO is reported to ship verl / TRL / NeMo-RL recipes;
#    verl is the most likely default for an NVIDIA release.
pip install -e .                          # or: pip install verl trl accelerate vllm
pip install numpy scipy matplotlib pandas # for the analysis layer in this package

# 3. datasets used by the two public anchors (typical for this class of paper):
#    - MATH-500 / GSM8K for the correctness channel
#    - generation-length statistic for the length channel
#    - a regex/format checker for the format channel
#    Confirm the exact split names from the repo's README / configs/ dir.
huggingface-cli download HuggingFaceH4/MATH-500 --repo-type dataset
huggingface-cli download openai/gsm8k          --repo-type dataset

# 4. base model. GDPO-class papers typically anchor on Qwen2.5-Math-7B or
#    Llama-3.1-8B-Instruct. Use whatever the repo's default config points to;
#    do NOT substitute — the AN/NA effect size depends on the base policy.
huggingface-cli download Qwen/Qwen2.5-Math-7B
```

Budget sanity check: problem-statement.md §4 quotes "~1–2.5 GPU-hr per LLM
anchor per GDPO's own reported budget." That figure is also unverified;
plan for **8×H100 × 4 h** per AN/NA pair as a safe ceiling for a 7B model
with `m=16` and 512 prompts. Synthetic sweeps in §(c) are CPU-minutes.

## (b) Reproduce the baseline (match the paper's headline figure)

Goal: reproduce the AN-vs-NA gap on the paper's two public anchors at their
reported hyperparameters. **No sweeps yet** — just hit their numbers.

```bash
# Anchor 1: correctness + length (or correctness + format — whichever the
# paper's Fig X uses as its money plot).
python -m gdpo.train --config configs/anchor1_AN.yaml  # scalarize-then-normalize
python -m gdpo.train --config configs/anchor1_NA.yaml  # decoupled per-channel

# Anchor 2: correctness + conditioning (length-given-correctness)
python -m gdpo.train --config configs/anchor2_uncond.yaml
python -m gdpo.train --config configs/anchor2_cond.yaml
```

- [ ] Locate the advantage estimator. Search the repo:
      `grep -rn "compute_advantage\|grpo_advantage\|normalize" gdpo/`.
      Expected: a single function with a string flag (`"AN"`/`"NA"` or
      `"scalarize"`/`"decoupled"`) or a function-dispatch dict.
- [ ] Confirm the AN/NA contrast is the **only** difference between the two
      configs (same data, same `m`, same `w`, same lr, same KL coef).
- [ ] Reproduce their headline number to within their reported seed noise.
      If you can't, stop and diagnose before any sweeps — the rest of the
      plan assumes the baseline is honest.

## (c) The three sweeps the original omits (validation plan §4)

These are the contributions of *this* package. Each is a small extension of
the reproduced training loop above.

**Sweep 1 — Correlation sweep (Thm 3 MSE floor).** Vary the *empirical*
correlation between the two reward channels by reweighting / subsampling
prompts into bins with measured `corr(r^(1), r^(2))` in {-0.6, -0.3, 0, 0.3, 0.6, 0.9}.
For each bin, train AN and NA at fixed `m, w` and log gradient-noise MSE
per step (use the multi-seed estimator in §(d)).
- [ ] Predicted: `MSE_NA ∝ w^T C w` (linear in measured ρ); `MSE_AN` curves
      cross above and below per Prop 1.
- [ ] Money number: slope of `MSE_NA` vs `w^T C w` should be ≈ `ζ_1 / m`.

**Sweep 2 — Conditioning phase boundary (Prop 4 / 4′).** Fix γ (contamination
strength: how much format-reward leaks on incorrect answers; controlled by
the format-reward shaping) and sweep `p_a` (correctness pass-rate) by
binning prompts by difficulty into ~6 bins. Compare {none, zero-fill,
subgroup} conditioning at each bin.
- [ ] Predicted: subgroup loses to *no* conditioning in the mid-range
      `p_a ∈ [0.29, 0.74]` (figT1). Zero-fill wins on ≈75% of `(p_a, γ)`
      plane (figT3); boundary tracks the bias-zero indifference curve
      `γ = α_c p_a / (1 - p_b)`.

**Sweep 3 — m* vs N at fixed correlation (Thm 3 group-size law).** For
budgets `N ∈ {32, 96, 192, 384}` and group sizes `m ∈ {2, 4, 8, 16, 32}`,
measure realized gradient-noise MSE; find `m*(N)`.
- [ ] Predicted: `m*` is flat in ρ (FALSIFIES the original `√(w^T C w)`
      claim, per figM1) and grows as `m* ∝ N^{1/3}` (fig2c).
- [ ] Negative-result statement is the contribution here — be ready to log
      whatever m\* you find, even if it doesn't match.

## (d) Money plot: predicted vs. realized estimator MSE

Single scatter, all sweeps overlaid: x-axis = theory prediction
(`w^T C w / m` for Sweep 1, U-statistic two-term formula for Sweep 3,
closed-form bias `p_b(1-p_a)[γ(1-p_b) - α_c p_a]` for Sweep 2);
y-axis = realized empirical MSE / bias from training. Each point is one
config × one seed; aim for ≥3 seeds per config so error bars are honest.

To measure gradient-noise MSE in a real training loop without an oracle
gradient: use the **multi-seed difference** estimator — run K≥4 independent
seeds per config, take the per-step gradient mean across seeds as a proxy
oracle, and compute the seed-wise MSE around it. This is what
`grpo_harness.py` already does for the synthetic checks; the only change
is that the gradient comes from a backward pass instead of a closed form.

- [ ] One scatter, log-log axes, y=x reference line.
- [ ] Color by sweep, marker by AN vs NA, error bars from seed spread.
- [ ] R² and slope of `log(realized) ~ log(predicted)`; both should be ≈1.

Acceptance: slope ∈ [0.9, 1.1] and R² ≥ 0.9 on Sweeps 1 and 2.
Sweep 3 is a negative-result claim; the acceptance is "m*(ρ) variance
across the ρ grid is within seed noise."
