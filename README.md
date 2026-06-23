# Conditioned Multi-Reward Advantage Estimation — research package

A literature gap, a problem statement with candidate theorems, fully written
proofs, and a simulation+LLM harness that verifies every claim.

## Paper skeleton (read in this order)

```
literature-review.md         Three strands of related work, the gap, positioning, citation checklist.
problem-statement.md         The gap (summary), canonical formal setup/notation, the 4 questions.
proposed-solutions.md        The 4 results (Prop 1, 2; Thm 3; Prop 4/4') with intuition + what each solves.
proofs.md                    [APPENDIX] Full proofs, cross-referenced to figures.
empirical-section.md         3-tier verification: synthetic harness, real GSM8K (2 scales), GRPO fine-tuning.
huggingface_assets.md        Released datasets (3) + fine-tuned models (3), with schemas + URLs.
```

## Supporting files

```
runpod_user_guide.md         How to reproduce the RunPod runs end-to-end.
validation_plan.md           LLM validation pipeline details + GPU budget + debugging guide.
gdpo_reproduction_plan.md    Anchor reproduction scaffold (DAPO fallback included).
gdpo_adapter.py              Framework-agnostic AN/NA + conditioning switch stub.
.env.example                 Template for RUNPOD_API_KEY + HF_TOKEN.
scripts/                     Synthetic harness (grpo_*.py) + LLM pipeline (llm_*.py) + RunPod orchestration (runpod_*.py).
figures/                     Synthetic-harness PNGs + LLM money plots + GSM8K/fintech/training outputs.
```

## Python package (`pip install multireward-grpo`)

The reusable training/generation/analysis/RunPod tooling is packaged so you can
train your own multi-reward GRPO models and verify Theorem 3 on your own
rollouts. The `scripts/` above remain the paper's experiment drivers.

```bash
pip install multireward-grpo            # core (numpy/scipy): advantage + analysis
pip install "multireward-grpo[llm]"     # + torch/transformers/peft: training & real generation
pip install "multireward-grpo[viz]"     # + matplotlib (money-plot figure)
pip install "multireward-grpo[runpod]"  # + requests (cloud GPU)
```

```python
from multireward_grpo import GRPOConfig, GRPOTrainer, compute_advantage
from multireward_grpo.examples import FintechRewardFunction, make_fintech_prompts

prompts = make_fintech_prompts(400, seed=0)
cfg = GRPOConfig(mode="na", weights=(1.0, 1.0, 0.5), n_steps=200)
GRPOTrainer(cfg, FintechRewardFunction(), prompts).train()
```

**Data contract:** `prompts` = list of strings / chat-message lists / dicts with
metadata; `reward_fn(completion, prompt) -> [r₁…r_R]` (channel 0 is the gate);
`weights` length `R`; `mode ∈ {na, an, single}`. Reward tensors for the analysis
tools use shape `(P, K, m, R)`. CPU smoke test: `multireward-grpo thm3-check`.
Full API + examples in `README_PYPI.md`. The package lives in `src/`; running the
research `scripts/` needs `uv sync --extra research`.

## Released artifacts (Hugging Face, `eagle0504` namespace)

Full schemas + load snippets in [`huggingface_assets.md`](huggingface_assets.md).

**Datasets**
- [multireward-grpo-gsm8k-rewards](https://huggingface.co/datasets/eagle0504/multireward-grpo-gsm8k-rewards) — 76,800 Qwen2.5-1.5B GSM8K rollouts (rewards + chains-of-thought)
- [multireward-grpo-gsm8k-rewards-qwen2.5-7b](https://huggingface.co/datasets/eagle0504/multireward-grpo-gsm8k-rewards-qwen2.5-7b) — 25,600 Qwen2.5-7B rollouts
- [multireward-grpo-fintech-customer-comms](https://huggingface.co/datasets/eagle0504/multireward-grpo-fintech-customer-comms) — 2,400 fintech conversations

**Models** (LoRA adapters for Qwen2.5-1.5B-Instruct)
- [multireward-grpo-fintech-na-qwen2.5-1.5b](https://huggingface.co/eagle0504/multireward-grpo-fintech-na-qwen2.5-1.5b) — NA (paper's recommendation)
- [multireward-grpo-fintech-an-qwen2.5-1.5b](https://huggingface.co/eagle0504/multireward-grpo-fintech-an-qwen2.5-1.5b) — AN baseline
- [multireward-grpo-fintech-single-qwen2.5-1.5b](https://huggingface.co/eagle0504/multireward-grpo-fintech-single-qwen2.5-1.5b) — single-reward ablation

> **Note:** raw generated `data/` is gitignored and not published to GitHub — all
> datasets/models are released via the Hugging Face links above.

## The gap (one line)

Multi-reward RLVR defaults to GRPO; GDPO (arXiv:2601.05242) showed scalarize-then-normalize
collapses reward combinations and that conditioning helps — but offers no theory. The
single-reward U-statistic theory (arXiv:2603.01162) and shrinkage-baseline work
(arXiv:2511.03710) exist but have not been extended to the multi-reward / conditioned setting.
This package develops and tests that extension.

## How to run

```bash
# environment (uv-based; pyproject.toml already in repo)
uv sync

# synthetic harness — each prints sim-vs-theory checks and writes PNGs
uv run scripts/grpo_harness.py    # Prop 1, 2; Thm 3 core + U-statistic; budget m*
uv run scripts/grpo_thm4.py       # Prop 4 / 4' conditioning
uv run scripts/grpo_selfnorm.py   # Thm 3 self-normalized lift
uv run scripts/grpo_mstar.py      # Thm 3 group-size vs correlation
uv run scripts/grpo_prop2.py      # Prop 2 resolution (enumeration)

# LLM validation pipeline
uv run scripts/llm_validate.py --mode mock   # CPU, ~20 sec — gating experiment
uv run scripts/llm_validate.py --mode gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
    --n-prompts 100 --K 8 --m-grid 4 8 16 32 --subsample-from-max   # requires GPU

# RunPod orchestration (spawn H100 pod, run, retrieve results, terminate)
# requires RUNPOD_API_KEY in .env and an SSH key at ~/.ssh/id_ed25519
uv run scripts/runpod_launch.py --n-prompts 50 --K 8 --m-grid 4 8 16 32
```
Runtime: synthetic harness is seconds to ~1 min each on a laptop; mock LLM
validation is ~20 sec on CPU; real GSM8K experiment is ~1.5 hr on a single
RTX 4090. See `validation_plan.md` for the full GPU budget and debug guide.

## Script -> figure -> claim map

| Script | Figure | Claim tested | Result |
|---|---|---|---|
| grpo_harness.py | fig1a_wCw_scaling | Thm 3: MSE $= (\tau^2/m)\, w^\top Cw$ at $\tau^2=1$ | verified, 3 digits |
| grpo_harness.py | fig1b_influence_law | Prop 1: AN influence $\propto w_\ell\sigma_\ell$, NA $\propto w_\ell$ | verified (het. scales) |
| grpo_harness.py | fig1c_coeff_vs_rho | Thm 3: $m\cdot$MSE traces $w^\top Cw$ | verified |
| grpo_harness.py | fig2a_bias | U-statistic self-centering bias $(m{-}1)/m$ | verified |
| grpo_harness.py | fig2b_variance | Hoeffding two-term variance $a/m+b/m^2$ | verified (inverse-variance-weighted fit; pure $a/m$ is rejected) |
| grpo_harness.py | fig2c_groupsize_law | budget-optimal $m^\star$ grows as $N^{1/3}$ (single reward) | verified by log-log fit over 6 budgets (exponent printed by harness) |
| grpo_thm4.py | figT1_crossover_pa | Prop 4': non-monotone subgroup MSE (degenerates structurally at low $p_a$ but wins on raw MSE there because the format channel is small); subgroup loses in mid-$p_a$; zero-fill best on average | verified |
| grpo_thm4.py | figT2_bias_law | Prop 4: bias $=p_b(1{-}p_a)[\gamma(1{-}p_b)-\alpha_c p_a]$, sign-changing | verified, $2\times10^{-3}$ |
| grpo_thm4.py | figT3_phase_diagram | Prop 4': condition vs not over $(p_a,\gamma)$; boundary = bias-zero curve | verified |
| grpo_selfnorm.py | figS1_selfnorm_mse | self-norm Thm 3: $m\cdot$MSE $=\mathbb{E}[w^\top\hat Cw]\to w^\top Cw$ | verified |
| grpo_selfnorm.py | figS2_selfnorm_vs_rho | self-norm structure preserved over $\rho$ | verified |
| grpo_selfnorm.py | figS3_selfnorm_bias | self-norm bias is $O(1/m)$ with distribution-dependent coefficient | Gaussian $\to -3\theta/4$ (exact $\Gamma$-formula); Bernoulli($\tfrac12$) $\to -\theta/2$; original $+\theta/4$ approx FALSIFIED |
| grpo_mstar.py | figM1_mstar_vs_rho | Thm 3 group size vs reward correlation | **FALSIFIED** $\sqrt{w^\top Cw}$; $m^\star$ flat in $\rho$ |
| grpo_prop2.py | figP2_resolution | Prop 2: NA product lattice $L^R$ vs AN sum lattice | verified (het. scales only) |
| llm_validate.py --mode mock | llm_money_scatter_mock_m{m}, llm_mse_vs_m_mock | Thm 3 + Prop 1 on synthetic LLM-shaped data | NA realized / pred = 0.94–1.02 across m=4–32; AN drifts to 0.62–0.75 (Prop 1) |
| llm_validate.py --mode gsm8k | llm_money_scatter_gsm8k_m{m}, llm_mse_vs_m_gsm8k | Same on real LLM generations | pending GPU run |

## Verification status (summary)

All claims are simulation-backed. Four first-draft claims were caught wrong and corrected:
1. **Prop 4 bias** — original dropped a cross-term; corrected form is sign-changing.
2. **Prop 4' variance penalty** — the claimed $1/p_a$ inflation does not appear. The real
   structural pathology is subgroup-baseline degeneracy at low pass-rate (figT1 gray curve), but
   the subgroup MSE profile is **non-monotone** in $p_a$: subgroup loses to unconditioned only in
   the mid-range $p_a\in[0.29, 0.74]$, not below a single threshold. Zero-fill conditioning is the
   recommended estimator (wins ~75% of the $(p_a,\gamma)$ plane).
3. **Thm 3 group-size scaling** — $m^\star \propto \sqrt{w^\top Cw}$ is false; $m^\star$ is set by
   prompt heterogeneity and bias, independent of reward correlation. Correlation sets the MSE *floor*.
4. **Self-norm bias coefficient** — original $+\theta/4$ independence approximation is false. The
   leading $1/m$ coefficient is distribution-dependent: Gaussian gives $-3\theta/4$ (exact via the
   $\Gamma$-formula); Bernoulli($\tfrac12$) gives $-\theta/2$ empirically.

Headline that survives: reward correlation $w^\top Cw$ governs the achievable MSE floor of
multi-reward GRPO (positively correlated objectives are fundamentally harder); decoupled
normalization restores weight-proportional influence and advantage resolution under heterogeneous
scales; the self-normalized estimator's sample-std bias is $O(1/m)$ with a
distribution-dependent leading coefficient ($-3\theta/4$ for Gaussian via the $\Gamma$-formula;
$-\theta/2$ empirically for Bernoulli($\tfrac12$)); conditioning removes a sign-changing
contamination bias, best implemented by zero-fill.

See `problem-statement.md` Section 3 for the per-claim figure/number references.
