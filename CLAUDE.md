# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is a **research package**, not an application. It develops a finite-sample,
correlation-aware *theory* of decoupled and conditioned multi-reward GRPO
advantage estimators, and ships a verification harness that checks every
theoretical claim against simulation and real-LLM data. The deliverable is the
paper (the `*.md` files) backed by reproducible code (`scripts/`) and released
HuggingFace artifacts.

The prose and the code are tightly coupled: each script verifies a specific
named claim, writes a specific figure, and the `README.md` "Script → figure →
claim map" table is the source of truth for that mapping. **When you change a
script's numerical output, update the claim's status in `README.md` and the
relevant prose file** (`empirical-section.md`, `proposed-solutions.md`,
`proofs.md`). Several first-draft claims were *falsified* by the harness and the
prose was corrected — that falsification record is a feature, not a bug; preserve
it.

### Document map (read in this order)
- `literature-review.md` — related work, the gap, citation positioning
- `problem-statement.md` — **canonical setup/notation referenced by all other files**; defines AN/NA, conditioning, the 4 questions Q1–Q4
- `proposed-solutions.md` — the results: Prop 1 (background, = MO-GRPO), Prop 2, Thm 3, Prop 4/4′
- `proofs.md` — full proofs (appendix)
- `empirical-section.md` — the 3-tier verification story
- `huggingface_assets.md` — released datasets (3) + fine-tuned models (3) under HF user `eagle0504`

## Core domain concepts (needed to read any script)

- **AN — Aggregate-then-Normalize**: scalarize rewards `s = wᵀr`, then group-normalize. The GRPO baseline. Suffers Prop 1 (high-variance channel dominates) and Prop 2 (resolution collapse).
- **NA — Normalize-then-Aggregate**: per-channel standardize, then weighted-sum. The "decoupled" estimator from MO-GRPO/GDPO — **the object analyzed**. Restores weight-proportional influence.
- These two orderings are implemented canonically in `gdpo_adapter.py::compute_multireward_advantage(rewards, w, mode)` and again, per-pipeline, as `advantage_AN` / `advantage_NA` in `scripts/llm_analysis.py` and `compute_advantages` in `scripts/grpo_train.py`. Keep all three consistent.
- **Conditioning** ("reward b counts only if a passes"): three impls compared — `none`, `zero_fill` (recommended, preserves U-statistic structure), `subgroup` (degenerates when <2 rollouts pass the gate). See `gdpo_adapter.py::apply_conditioning`.
- **Headline result (Thm 3)**: MSE `= (τ²/m)·wᵀCw + O(m⁻²)` — reward correlation `C` sets the achievable MSE floor.
- **"Money plot"**: predicted-vs-realized gradient-noise MSE scatter, computed *without* an oracle gradient by using the seed-mean as a proxy oracle (`measure_gradient_mse` / `realized_mse`).

## Two layers: the pip package and the research harness

This repo is **both** a published PyPI package (`multireward-grpo`, import
`multireward_grpo`, src-layout under `src/`) **and** the original research
harness (`scripts/`). The package modularizes the reusable training/generation/
analysis/RunPod pieces so external developers can train their own models; the
`scripts/` are the paper's experiment drivers and are NOT installed by the
package. The two share the AN/NA/conditioning definitions — keep
`src/multireward_grpo/advantage.py`, `gdpo_adapter.py`, `scripts/llm_analysis.py`,
and `scripts/grpo_train.py` consistent (see the AN/NA note above).

Package modules (`src/multireward_grpo/`): `advantage.py` (AN/NA + conditioning),
`rewards.py` (reward framework + GSM8K math channels), `generation.py`
(Mock/Qwen backends), `train.py` (generalized `GRPOTrainer` — bring your own
prompts + reward_fn + weights + mode), `analysis.py` (Thm 3 `analyze`),
`runpod.py` (`RunPodClient`), `examples/` (fintech + gsm8k), `cli.py`
(`multireward-grpo` console script: `version`, `thm3-check`, `train`).

## Environment & common commands

Uses **uv** (not pip/conda). Dev Python pinned to **3.12** via `.python-version`;
the published wheel allows `>=3.10`. The project is now a real package, so
`uv sync` builds+installs it into the venv.

```bash
uv sync                      # lean package deps only (numpy, scipy)
uv sync --extra research     # + matplotlib/datasets/hf/pandas/pyarrow/requests — REQUIRED to run scripts/
uv sync --extra llm          # adds torch (cu124 wheel index)/transformers/peft — GPU host only
uv run scripts/<name>.py     # always run scripts through uv (needs --extra research synced)
uv build                     # build sdist + wheel into dist/ (excludes data/)
```

Note: the synthetic/LLM harness scripts need the `research` extra (matplotlib et
al. are no longer base deps — they moved to extras to keep the published library
lean). The package core (advantage + analysis math) needs only numpy/scipy.

`research` is the union convenience extra for this repo. Library consumers get
finer-grained ones (`pyproject.toml`): `viz` (matplotlib, figures), `runpod`
(requests, `multireward_grpo.runpod`), `data` (datasets/hf-hub/pandas/pyarrow,
`examples.gsm8k` + model publishing), `llm` (torch/transformers/peft). When you
add an import to a package module, put its dependency in the matching extra —
never in the base `dependencies`, which must stay numpy+scipy only.

There is **no test suite, linter, or build step**. Verification = running the
harness scripts and checking their printed "sim-vs-theory" lines and PNG output.
The package's closest smoke test is `multireward-grpo thm3-check` (CPU).
The closest thing to a unit test is the mock-mode gate (see below).

### Synthetic harness (CPU, seconds–1 min each; writes PNGs to CWD)
```bash
uv run scripts/grpo_harness.py    # Prop 1, 2; Thm 3 core + U-statistic; budget m*  (fig1*, fig2*)
uv run scripts/grpo_thm4.py       # Prop 4 / 4' conditioning  (figT*)
uv run scripts/grpo_selfnorm.py   # Thm 3 self-normalized lift  (figS*)
uv run scripts/grpo_mstar.py      # group size vs correlation  (figM1 — a FALSIFIED claim)
uv run scripts/grpo_prop2.py      # Prop 2 resolution lattice  (figP2)
```

### LLM validation pipeline
```bash
# Mock mode = the GATING experiment. CPU, ~20s. If it doesn't reproduce Thm 3 to
# ~1% on synthetic Bernoulli rewards, there's a bug in the analysis pipeline —
# do NOT spend GPU money until mock passes.
uv run scripts/llm_validate.py --mode mock

# Real GSM8K (needs GPU + --extra llm). --subsample-from-max is ~4x cheaper.
uv run scripts/llm_validate.py --mode gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
    --n-prompts 100 --K 8 --m-grid 4 8 16 32 --subsample-from-max
```

### RunPod orchestration (spawns H100 pod, runs, retrieves, terminates)
Requires `RUNPOD_API_KEY` in `.env` (copy from `.env.example`) and an SSH key at
`~/.ssh/id_ed25519`. See `runpod_user_guide.md` and `validation_plan.md` for the
full GPU budget and debugging guide.
```bash
uv run scripts/runpod_launch.py --n-prompts 50 --K 8 --m-grid 4 8 16 32   # Thm 3 GSM8K run
uv run scripts/runpod_train.py --mode both --n-steps 200                  # Tier-3 GRPO fine-tune
uv run scripts/runpod_train_multiseed.py                                  # multi-seed training curves
```

## Architecture of the script layer

Scripts fall into four families (prefix = family):

- **`grpo_*` — synthetic harness.** Self-contained, NumPy-only. Each draws rewards with a known correlation matrix, computes AN/NA advantages, and overlays the closed-form theory on a PNG. `grpo_harness.py` is the main one (Prop 1, Prop 2, Thm 3, U-statistic bias/variance, budget-optimal `m*`). `grpo_train.py`/`grpo_eval.py` are the GPU fine-tuning + eval (Tier 3, fintech domain).

- **`llm_*` — real-LLM validation pipeline.** Three-stage, importable modules:
  1. `llm_generation.py` — `Backend` protocol with `MockBackend` (synthetic, CPU) and `QwenBackend` (real, GPU). `run_corpus` → `pack_for_analysis` produces a `(P, K, m, R)` tensor (prompts × seeds × rollouts × reward channels).
  2. `llm_rewards.py` — the R reward channels for GSM8K (correctness/length/format).
  3. `llm_analysis.py` — `analyze()` returns a `Thm3Result`; produces money-plot scatters and influence-law numbers.
  `llm_validate.py` is the entry point orchestrating all three; `llm_prop4_real.py` does the Prop 4 γ-sweep on saved rewards.

- **`runpod_*` — cloud GPU orchestration.** `runpod_lib.py` is the shared library (pod create/wait/ssh/rsync/delete via the RunPod REST API). `runpod_launch.py` (Thm 3), `runpod_train.py` / `runpod_train_multiseed.py` (Tier 3 training), `runpod_fintech.py` are thin task-specific drivers. These rsync the repo to a pod, run a `uv` command remotely, rsync results back, then terminate the pod. **Note `runpod_launch.py` duplicates much of `runpod_lib.py` rather than importing it** — keep them in sync if editing the pod lifecycle logic.

- **`hf_*` / `fintech_*` — asset publishing + the fintech domain.** `fintech_scenarios.py`/`fintech_rewards.py`/`fintech_generate.py` define the synthetic fintech-customer-comms domain (Tier 3); `hf_push_*.py` / `hf_add_parquet.py` publish datasets and LoRA adapters to the `eagle0504` HuggingFace account.

`plot_grpo_training.py` / `plot_grpo_training_multiseed.py` render the training
curves produced by the Tier-3 fine-tune (`grpo_train.py` / `runpod_train*.py`);
`hf_enrich_gsm8k_parquet.py` post-processes a released GSM8K parquet before
re-pushing it. These are post-hoc plotting/publishing utilities, not part of the
verification harness.

`gdpo_adapter.py` (repo root) is a **non-runnable stub** documenting the drop-in
contract for wiring AN/NA + conditioning into an external verl/TRL/GDPO training
loop — it is reference, not part of any pipeline.

## Conventions specific to this repo

- **`.txt` and `:Zone.Identifier` files**: `data/pdfs/*.txt` are extracted text of the source papers (read these instead of the PDFs). `*:Zone.Identifier` files are Windows/WSL download-provenance cruft — ignore them.
- **Figures**: synthetic-harness scripts write PNGs to the **current working directory** (`FIG = "."`), so run them from `scripts/` or `figures/` as intended; the LLM pipeline writes to `figures/` via `--out-dir`. `*_mock*` figures are committed CPU-reproducible outputs; bare-name figures come from GPU runs.
- **Reproducibility**: synthetic scripts use fixed seeds (`np.random.default_rng(7)` etc.). Changing a seed can shift the last printed digit — don't "fix" a claim by reseeding.
- **Falsified claims are documented on purpose** in `README.md` (§"Verification status") and the prose. If your edit changes a result, update both the table row and the narrative; never silently flip a "FALSIFIED" to "verified".
- **Git — two repos, different visibility.** This working copy is under version control (branch `main`, remote `github.com/yiqiao-yin/multireward-grpo`, **PUBLIC**). History is shallow — the whole package + harness landed in one commit (`f3df088`), so don't expect `git log` to explain design decisions; the prose files and this file are the record. Secrets live in `.env` (RunPod + HF tokens), gitignored; `.env.example` is the template.
- **`data/` is a separate, PRIVATE repo** nested inside this one (its remote is in `data/.git/config`; not named here because this file is public). The outer `.gitignore` ignores `data/` wholesale, so the public repo never sees it — no gitlink, no submodule. It holds the LaTeX manuscript, `figs/`, the Scientific Reports cover letter, editor correspondence, and third-party reference PDFs. **It must stay private**: the manuscript is under review, the correspondence is private, and `data/pdfs/` is other authors' copyrighted work. Never `git add` anything from `data/` in the outer repo, and never move paper content out of `data/` into a tracked path.
- **The manuscript lives at `data/overleaf/template/_extracted/templateArxiv.tex`** (~1070 lines) — that is the live copy, byte-identical to the `_v2.zip` snapshot and newer than `_FILLED.zip`. Edit `_extracted/`, never the zips. It states the same claims as the tracked `.md` prose files, so **a numerical result that changes must be updated in both** — the `.tex` is in the private repo and the `.md` in the public one, so nothing will flag the drift for you. Commit `data/` changes from inside `data/` (its own repo); commit code changes from the root.
- **Release surface**: the repo publishes to three places — GitHub (above), PyPI (`multireward-grpo`, version in `pyproject.toml`), and HuggingFace (`eagle0504` namespace, see `huggingface_assets.md`). A version bump means bumping `pyproject.toml` and rebuilding `dist/` with `uv build`; `README_PYPI.md` (not `README.md`) is what renders on the PyPI page, so keep it in sync separately.
