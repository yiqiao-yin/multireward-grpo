# LLM Validation Plan — Thm 3 on Real Data

This document is the bridge between the synthetic harness (which verifies the
theorems to 3 significant figures) and the LLM experiment that produces the
headline empirical figures for the paper. The whole pipeline is built and
mock-tested; no GPU spend is needed until you flip a single CLI flag.

## What success looks like

The "money plot" for Thm 3 is a single figure: realized gradient-noise MSE
(measured across $K$ random seeds, per prompt) plotted against the Thm 3
prediction $(1/m) \, w^\top \hat C w$. If the theory holds, NA points hug
the $y = x$ line; AN points drift to a different line (Prop 1's contamination).

Acceptance bar — what we need to see in mock mode *before* spending GPU money:

- [x] Realized NA MSE / predicted MSE ∈ [0.9, 1.1] (mean across prompts, all $m$)
- [x] Realized AN MSE ≠ predicted (lower in absolute terms because AN
      normalizes away the multi-reward structure — Prop 1's story)
- [x] Cross-$m$ MSE-vs-$m$ overlay shows NA tracking the dashed prediction
      line across $m \in \{4, 8, 16, 32\}$

**All three are confirmed** by the mock run (`uv run scripts/llm_validate.py
--mode mock`). Mock-mode numbers from the gating experiment:

| $m$ | Predicted (pooled) | Realized NA / pred | Realized AN / pred |
|---|---|---|---|
| 4  | 0.324 | **1.017** | 0.752 |
| 8  | 0.180 | **1.017** | 0.685 |
| 16 | 0.094 | **0.935** | 0.620 |
| 32 | 0.047 | **0.964** | 0.644 |

NA ratio stays within ±7% of 1.0 across the full range. AN ratio sits at
0.62–0.75, with the gap to NA growing as reward correlations stabilize at
larger $m$ — exactly the Prop 1 contamination signature.

## Pipeline architecture

Four files in `scripts/`, all tested end-to-end:

```
scripts/
├── llm_rewards.py     reward channels (correctness, length, format)
├── llm_generation.py  Backend protocol + MockBackend + QwenBackend
├── llm_analysis.py    advantages, realized MSE, predicted MSE, plots
└── llm_validate.py    CLI entry point — --mode {mock,gsm8k}
```

The interface is tight by design: a Backend exposes one method,
`.generate_for_prompt(prompt, gold, m, seed) → Rollouts`, and the analysis
module takes a `(P, K, m, R)` reward tensor. Swapping out the backend
(MockBackend → QwenBackend → DAPOBackend → …) is a single line.

The directory layout mirrors the math:

```
proofs.md Thm 3 statement                 →  llm_analysis.analyze
proofs.md Thm 3 self-norm Step 1+2 identity →  llm_analysis.predicted_mse_per_prompt
problem-statement.md §2 NA estimator      →  llm_analysis.advantage_NA
problem-statement.md §2 AN estimator      →  llm_analysis.advantage_AN
problem-statement.md §3 Prop 1 influence  →  llm_analysis.influence_per_channel
```

## How to run

### Mock mode (no GPU, runs in ~20 sec on CPU)

```bash
uv run scripts/llm_validate.py --mode mock \
    --n-prompts 80 --K 32 --m-grid 4 8 16 32
```

Writes `figures/llm_money_scatter_mock_m{m}.png`,
`figures/llm_mse_vs_m_mock.png`, and `figures/llm_summary_mock.json`. This
is the gating experiment — if the table above stops matching after a code
change, the analysis pipeline is broken and no LLM experiment will fix it.

### GSM8K mode (single consumer GPU)

```bash
# install once (pyproject.toml ships an "llm" optional group)
uv sync --extra llm

# run
uv run scripts/llm_validate.py --mode gsm8k \
    --n-prompts 100 --K 8 --m-grid 4 8 16 32 \
    --model Qwen/Qwen2.5-1.5B-Instruct
```

Writes the same files with `_gsm8k_` in the names. K=8 keeps the per-seed
overhead reasonable; bump to K=16 or 32 if you need tighter error bars and
have the GPU budget.

## GPU budget (real numbers)

Estimates assume batched generation. Two GPU options compared:

| Configuration | Total generations | RTX 4090 (~$0.40/hr) | RunPod H100 PCIe (~$2.39/hr) |
|---|---|---|---|
| 100 prompts × 8 seeds × {4, 8, 16, 32} | 48 000 | ~1.5 hr ≈ **$0.60** | ~25 min ≈ **$1.60** |
| 300 prompts × 16 seeds × {4, 8, 16, 32} | 290 000 | ~9 hr ≈ **$3.60** | ~2.5 hr ≈ **$6.00** |
| Same as above, but Qwen2.5-7B | 290 000 | (won't fit at fp16) | ~5 hr ≈ **$12** |

Generation cost is *cheap*. The expensive thing is training, and we are not
training in this experiment — only measuring estimator MSE on a frozen
policy. **The full Thm 3 money plot can be produced on RunPod H100 for
under $15.**

## What the GSM8K experiment will look like

Per-prompt reward correlation $\hat C_p$ on real data will be more
heterogeneous than in mock mode (math problems vary widely in
difficulty/length structure), so the per-prompt scatter should look
considerably less compressed than the mock version. The headline cross-$m$
overlay, however, should look essentially identical to the mock figure — a
straight line for NA tracking $(1/m) w^\top \bar C w$, with AN drifting
below it as Prop 1 predicts.

**Predicted reward correlation pattern on Qwen2.5-1.5B over GSM8K:**

- correctness × length: weakly negative (–0.2 to –0.4). Easy problems → short
  correct answers; hard problems → long incorrect rambles.
- correctness × format: strongly positive (+0.5 to +0.7). When the model
  knows the answer it tends to use the `\boxed{}` convention.
- length × format: mildly negative. Long outputs often miss the boxed
  closer.

If the realized $\hat C$ is wildly off this pattern, that's a data-quality
signal: e.g., the format reward regex is broken, or the tokenizer is not
seeing `\boxed{}` correctly.

## Failure modes and how to debug

| Symptom | Likely cause | Where to look |
|---|---|---|
| Realized NA / pred drifts > 1.2 | $K$ too small → seed variance dominates | bump `--K`; check `realized_na.std()` |
| AN realized > NA realized | Reward σ's are very similar | check pooled Ĉ — Prop 2's equal-σ regime |
| pooled Ĉ has nan / inf | A reward channel has zero within-group variance | check `score_generation` outputs |
| Format reward always 0 | Tokenizer eating `\boxed{}` | print first few generations, inspect |
| GSM8K accuracy < 5% | Model too small / prompt template wrong | check `_format_prompt` in QwenBackend |

The `figures/llm_summary_*.json` file contains every aggregate number for
post-hoc inspection; print it with `jq . figures/llm_summary_mock.json | less`.

## After the headline plot lands

In priority order, the follow-ups that turn this into a full empirical
section of the paper:

1. **Prop 4 bias law on real data.** Sweep the contamination knob $\gamma$
   in `RewardConfig` (already implemented in `llm_rewards.score_generation`)
   and re-measure the cross-objective bias. Predicted curve from §3 should
   sign-change at $\gamma^\star = \alpha_c p_a / (1 - p_b)$.

2. **Prop 4′ phase boundary on real data.** Bin GSM8K problems by
   difficulty (use base-model pass-rate as $p_a$), sweep $\gamma$, plot the
   $(p_a, \gamma)$ phase diagram with zero-fill / subgroup / none.

3. **Fintech anchor (new dataset).** Re-run the same money plot on the
   synthetic-fintech corpus once it's generated. Sets up the cross-domain
   generalization story.

4. **Compare against GDPO's reported numbers.** Once the GDPO repo is
   located (see `gdpo_reproduction_plan.md`), the same `llm_validate.py
   --mode <gdpo-anchor>` should reproduce their AN-vs-NA gap to within
   their reported seed noise. That's the literature-anchor sanity check.

Each of these is the same pipeline with a different backend or sweep —
no new code paths to design.

## Reproducibility checklist (camera-ready)

- [ ] All seeds documented (currently: `score_seed=12345`, generation seed
      = `k`, prompt seed = `hash(prompt)`)
- [ ] Hardware listed (model name, GPU, batch size, generation params)
- [ ] Reward function code linked to a specific PyPI version
- [ ] Dataset URL pinned to a HuggingFace commit hash
- [ ] Random-state checkpoint saved between phases
- [ ] `figures/llm_summary_*.json` shipped as supplementary material
