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
- `INTUITION.md` — plain-language companion: every result with one everyday analogy. Mostly notation-free; the "Formally:" blockquotes and the one section bridging the cart analogy to `wᵀCw` are the deliberate exceptions. Keep in sync when a result changes, same as the other prose files. **This file is MIRRORED at `data/INTUITION.md`** — two copies in two repos, kept byte-identical by hand. Nothing enforces it and it has silently drifted before, so after editing either copy, `cp` it to the other and commit both repos. Check with `diff INTUITION.md data/INTUITION.md`.
- `gdpo_reproduction_plan.md` — plan for reproducing against the GDPO anchor. **Its header flags that the GDPO citation (arXiv ID, authors, GitHub URL, GPU budget) is an unverified placeholder** carried from `problem-statement.md`/`README.md`, written without network access. Don't treat those as confirmed facts, and don't propagate them further without checking.

## Core domain concepts (needed to read any script)

- **AN — Aggregate-then-Normalize**: scalarize rewards `s = wᵀr`, then group-normalize. The GRPO baseline. Suffers Prop 1 (high-variance channel dominates) and Prop 2 (resolution collapse).
- **NA — Normalize-then-Aggregate**: per-channel standardize, then weighted-sum. The "decoupled" estimator from MO-GRPO/GDPO — **the object analyzed**. Restores weight-proportional influence.
- These two orderings exist in **five** places and there is no test tying them together — keep them consistent by hand:
  1. `src/multireward_grpo/advantage.py::compute_advantage(rewards, w, mode)` — the package's public entry point; `mode` is **lowercase** `"na"`/`"an"`/`"single"`.
  2. `gdpo_adapter.py::compute_multireward_advantage(rewards, w, mode)` — the external-trainer stub; `mode` is **uppercase** `"NA"`/`"AN"`. The case mismatch with (1) is real, not a typo — don't pass one's string to the other.
  3. `src/multireward_grpo/analysis.py::advantage_AN` / `advantage_NA`
  4. `scripts/llm_analysis.py::advantage_AN` / `advantage_NA` — (3) and (4) are near-verbatim duplicates of each other; a fix to either almost always belongs in both.
  5. `scripts/grpo_train.py::compute_advantages`
- **Conditioning** ("reward b counts only if a passes"): three impls compared — `none`, `zero_fill` (recommended, preserves U-statistic structure), `subgroup` (degenerates when <2 rollouts pass the gate). See `gdpo_adapter.py::apply_conditioning`.
- **Headline result (Thm 3)**: MSE `= (τ²/m)·wᵀCw + O(m⁻²)` — reward correlation `C` sets the achievable MSE floor.
- **"Money plot"**: predicted-vs-realized gradient-noise MSE scatter, computed *without* an oracle gradient by using the seed-mean as a proxy oracle (`measure_gradient_mse` / `realized_mse`).

## Two layers: the pip package and the research harness

This repo is **both** a published PyPI package (`multireward-grpo`, import
`multireward_grpo`, src-layout under `src/`) **and** the original research
harness (`scripts/`). The package modularizes the reusable training/generation/
analysis/RunPod pieces so external developers can train their own models; the
`scripts/` are the paper's experiment drivers and are NOT installed by the
package. The two share the AN/NA/conditioning definitions — see the five-copy
list in the AN/NA note above before editing any of them.

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
- **Git — two repos, different visibility.** This working copy is under version control (branch `main`, remote `github.com/yiqiao-yin/multireward-grpo`, **PUBLIC**). History is shallow — the whole package + harness landed in one commit (`f3df088`), with only a handful of doc/prose commits after it, so don't expect `git log` to explain design decisions; the prose files and this file are the record. Secrets live in `.env` (RunPod + HF tokens), gitignored; `.env.example` is the template.
- **`data/` is a separate, PRIVATE repo** nested inside this one (its remote is in `data/.git/config`; not named here because this file is public). The outer `.gitignore` ignores `data/` wholesale, so the public repo never sees it — no gitlink, no submodule. It holds the LaTeX manuscript, `figs/`, the Scientific Reports cover letter, editor correspondence, and third-party reference PDFs. **It must stay private**: the correspondence is private, and `data/pdfs/` is other authors' copyrighted work. Never `git add` anything from `data/` in the outer repo, and never move paper content out of `data/` into a tracked path.
- **STATUS: the paper is ACCEPTED and in production** (Scientific Reports, editor Claire Potter; both reviewers accepted at round 3). **Do not edit the manuscript.** The text that went to typesetting is frozen. If something in it is genuinely wrong, say so and ask — a production change is a request to the editorial office, not a file edit. Work that *is* still open lives in the public repo (prose `.md`, package, HF assets).
- **The manuscript lives at `data/overleaf/template/_extracted/templateArxiv.tex`** (2047 lines after revision 3) — that is the annotated master. Always edit `_extracted/`, never the zips: the zips are numbered snapshots built *from* `_extracted/` for Overleaf upload. `_v2.zip` is the as-submitted version the reviewers saw, worth keeping as the historical record; `_v6.zip` is revision 1; `_v7.zip` is revision 2; **`_v8.zip` is revision 3 and is the last of that lineage** — it matches `_extracted/` byte for byte. Never delete an older zip. If a new numbered zip is ever needed, build it from `_extracted/` by copying `_v8.zip`'s manifest rather than globbing the directory — that keeps the entry set and order stable across snapshots and skips the `*:Zone.Identifier` cruft, which would otherwise land in the archive. Then verify: the new zip should differ from its predecessor in exactly the files you edited, and every entry should hash-match `_extracted/`.
- **Two manuscript lineages — don't confuse them.** `_extracted/` (+ `_v1…_v8.zip`) is the **annotated** copy carrying `\hl` highlighting, revision tags and `revremark` blocks; it is the record of what changed in each review round. `_clean/` (+ `Conditioned_Multireward_final_clean.zip`, deliberately *outside* the `vN` numbering because it is a different lineage) is the **de-annotated, typesetting-ready** copy sent to production: 0 `\hl`, 0 revision tags, `soul` dropped, and the 14 `revremark` blocks relabelled as plain numbered `remark`s. The remarks were **renamed, not deleted** — 16 `\ref{rem:...}` point into them from the body and they carry load-bearing content (Thm 3's assumptions, the `Cw = w` condition, the Table 5 statistics, the limitations). Numbering stays 1–14 so plain-text "Remark N" mentions still resolve. The two `.tex` files differ *only* in markup: all 132 distinct result values and the full section/theorem/table/figure/citation structure are identical, and that equivalence is worth re-verifying if either is ever touched.
- **There is no LaTeX toolchain in this environment** (no `pdflatex`, `latexmk` or `tectonic`). You cannot compile the manuscript locally to check an edit. Verify structurally instead — balanced braces, matched `\begin`/`\end`, and the `\hl` constraint below — then say plainly that it is uncompiled and needs an Overleaf build.
- **Revision markup convention** (applies to `_extracted/` only — `_clean/` has none of this by design). Every change made in response to a reviewer is highlighted with `\hl{...}` (package `soul`, loaded after `xcolor`) and opens with a plain-text tag naming the source, e.g. `\hl{Reviewer 1, Item 3: assumptions made explicit.}` **`\hl` cannot contain math or macros** — no `$...$`, no `\ref`, no `\emph`, and no blank line inside the braces. Split the sentence and leave the math outside the highlight. Point-by-point replies live in `revremark` environments (`\newtheorem{revremark}`), which deliberately use their **own counter**, not the shared theorem counter: numbering them together would renumber Proposition 2 / Theorem 3 / Proposition 4, which the reviewers cite by number. Reviewers also cite Revision Remarks by number, so **do not add, remove or reorder a `revremark`** without checking every plain-text "Remark N" reference in the body.
- **A qualification belongs at the point of claim, not only in a Revision Remark — and the title is a point of claim.** This lesson cost two review rounds. Round 1 recorded qualifications in the remarks — one of which even asserted "we have moderated the surrounding text accordingly" — while the Abstract, body and Conclusion went on making the unqualified claim. Reviewer 1 checked the remarks against the body and found the gap. Revision 3 was the same failure one level up: the qualification about aggregate reward had reached the body but not the **title**, which is the one sentence every reader sees. When you moderate a claim, grep the **whole** manuscript for every occurrence — title and abstract included — and fix each one; the remark is the derivation, not the fix.
- **Submission documents live in `data/overleaf/`**: `reviewer_feedback.docx` (round 1), `_2.docx` (round 2), `_3.docx` (round 3 — editor's framing + dataset-provenance points; both reviewers accepted), `_4.docx` (a single paragraph covering the clean typesetting copy) with their derived `.pdf`s, `Cover_Letter_Scientific_Reports.docx`/`.pdf`, and `Author_Contributions_Statement.txt`. Incoming referee reports are in `data/feedback/`. The `.docx` files are generated by `data/tools/build_response.py`, `build_response_2.py`, `build_response_3.py`, `build_response_4.py` and `build_coverletter.py` (all on the dependency-free `mkdocx.py` OOXML writer) — edit the script and regenerate, do not hand-edit the `.docx`, or the next regeneration silently discards the edit. Keep one script per review round; do not fold a later round back into an earlier script. The `.pdf` beside each `.docx` is a **manual Word export** — `mkdocx.py` cannot produce it, so after regenerating, the PDF is stale until the author re-exports it. The response documents state the same claims as the tracked `.md` prose files, so **a numerical result that changes must be updated in both** — the `.tex` is in the private repo and the `.md` in the public one, so nothing will flag the drift for you. Commit `data/` changes from inside `data/` (its own repo); commit code changes from the root.
- **The paper's title has changed TWICE; the published one is the third.** Original (as submitted): *"When and Why Decoupling and Conditioning Beat Reweighting in Multi-Reward GRPO: A U-Statistic Treatment"*. Revision 2 removed the colon per the in-house editor: *"…Outperform Reweighting…"*. Revision 3 removed *"Outperform"* because the editor found it unsupported by the evidence — the experiments show differences in influence allocation and gradient-noise behaviour, not statistically significant gains in aggregate reward. **The final title is *"Decoupling and Conditioning Reshape Influence Allocation and the Gradient-Noise Floor in Multi-Reward GRPO under a Finite-Sample U-Statistic Analysis"*.** This is now carried consistently across the `.tex` files and all seven prose/code locations that name the paper: `pyproject.toml` description, `README_PYPI.md`, `problem-statement.md`, `INTUITION.md`, `src/multireward_grpo/__init__.py`, `src/multireward_grpo/advantage.py`, and `data/README.md`. **If the title ever changes again, those seven are the checklist** — nothing enforces it, and the drift went two revisions deep before being caught. The PyPI description is user-visible and needs a release to propagate. It is *correct* for `data/tools/build_response*.py` and `build_coverletter.py` to keep older titles — those are dated correspondence, and each round's letter cites the previous title explicitly. Do not "fix" those.
- **Code is deposited under a DOI; that requirement is satisfied.** Zenodo archived GitHub release `v0.1.0`: **version DOI `10.5281/zenodo.22267541`** (cite this one — it pins the exact code behind the results) and concept DOI `10.5281/zenodo.22267540` (always resolves to latest). Cited in the manuscript's Code Availability section. Pushing to `main` does **not** update the archive — only cutting a new GitHub release does, and that would mint a *new* version DOI, orphaning the number printed in the paper. So do not cut a release casually. Zenodo *metadata* (title, description, related identifiers) is editable at any time without changing the DOI; only the files and the DOI are frozen. When the paper publishes, the record's description should get the citation and the article DOI added as an `isSupplementTo` related identifier.
- **Release surface**: the repo publishes to three places — GitHub (above), PyPI (`multireward-grpo`, version in `pyproject.toml`), and HuggingFace (`eagle0504` namespace, see `huggingface_assets.md`). A version bump means bumping `pyproject.toml` and rebuilding `dist/` with `uv build`; `README_PYPI.md` (not `README.md`) is what renders on the PyPI page, so keep it in sync separately.
