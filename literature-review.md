# Literature Review

*Companion to `problem-statement.md`, `proposed-solutions.md`, `proofs.md`, and
`empirical-section.md`. This file surveys the three research strands our work
sits at the intersection of, states the gap precisely, and positions our four
results relative to prior art.*

> **Citation status.** All load-bearing citations below were verified against
> arXiv via web search on 2026-05-29 (IDs, titles, and abstracts confirmed).
> Verified entries are marked **[✓]**. A handful of classical references
> (Greensmith et al.; Hoeffding; Fisher) are standard and easy to confirm.

---

## 1. Overview: three strands, one gap

Our contribution lives where three lines of work meet:

```
   (A) GRPO / RLVR                (B) single-reward estimator theory
   "how to train LLMs            "how good is the GRPO baseline,
    with verifiable rewards"       as a statistical estimator?"
            \                         /
             \                       /
              \                     /
            (C) multi-objective RLHF / GRPO
   "decoupled normalization" (MO-GRPO, GDPO)
   — method exists, variance-domination noted
                     |
                     v
          THE GAP: a finite-sample,
       correlation-aware *theory* of the
     *decoupled* and *conditioned*
      multi-reward advantage estimators
      (MSE floor, resolution bound,
       conditioning-bias law)
```

- Strand **(A)** gives us the algorithm (GRPO) and the empirical phenomena we
  explain (decoupling and conditioning help).
- Strand **(B)** gives us the statistical tools (U-statistics, Hoeffding
  decomposition, optimal baselines) but only for a **single** reward.
- Strand **(C)** gives us the multi-reward setting. **The decoupled
  (normalize-then-aggregate) estimator already exists** — it is the method of
  MO-GRPO (Sept 2025) and GDPO (Jan 2026), which also identified, at least
  qualitatively, that aggregation lets the highest-variance channel dominate.
  What strand (C) lacks is a *finite-sample, correlation-aware analysis*: no
  reward-correlation-matrix MSE law, no rigorous resolution bound, no
  conditioning-bias law.

**We do not propose the decoupled estimator** — MO-GRPO / GDPO did — **and we
do not claim the influence law either**: MO-GRPO's Theorems 1–2 already give
the reward-advantage correlation for AN and NA *including the cross-channel
covariance terms* (see §3 below). Our genuinely new results are the
finite-sample MSE floor as a quadratic form in the reward correlation matrix
(**Thm 3**) and the sign-changing conditioning-bias law (**Prop 4**). We
additionally contribute the lattice resolution bound (**Prop 2**, formalizing
GDPO's "collapse"); and we re-state MO-GRPO's influence result in our notation
with general weights as **Prop 1**, used only as background.

---

## 2. Strand A — GRPO and RLVR

**Shao et al. 2024 — DeepSeekMath / GRPO.** Introduces Group Relative Policy
Optimization: rather than learning a value network, GRPO samples a *group* of
$m$ rollouts per prompt and uses the within-group mean (and std) of the reward
as the baseline. The advantage of rollout $j$ is the group-standardized reward
$(r_j - \bar r)/\hat\sigma$. This is the estimator whose statistical behavior
our paper analyzes. Origin of the "group baseline" we generalize.

**Guo et al. 2025 — DeepSeek-R1.** Scales RLVR (RL with verifiable rewards) to
frontier reasoning models, establishing that simple verifiable rewards
(answer-correct / not) suffice to elicit strong chain-of-thought. Motivates
the "verifiable reward channel" abstraction we use ($R$ channels, each a
checkable score). **[partially verified — DeepSeek-R1 is real; exact arXiv ID
should be confirmed].**

**Yu et al. 2025 — DAPO (arXiv:2503.14476).** A production-grade GRPO variant:
decoupled clipping, dynamic sampling, and a fix for zero-variance groups
(when all $m$ rollouts get identical reward, the baseline is degenerate). DAPO
is the *closest existing engineering* to our setting — it already ships
multi-component reward handling in verl — but it offers no finite-sample
theory of why its choices reduce variance. We provide that theory; DAPO is a
natural deployment target for our drop-in NA estimator.

**Liu et al. 2025 — Dr. GRPO / "Understanding R1-Zero-Like Training".**
Identifies length- and difficulty-normalization *biases* in GRPO: the standard
within-group normalization systematically rewards longer or easier responses.
This is the single-reward analog of the bias our Proposition 4 dissects in the
multi-reward / conditioned case. Directly relevant to our length-channel
analysis.

---

## 3. Strand B — single-reward estimator theory

**Greensmith, Bartlett & Baxter 2004.** Classical analysis of variance
reduction for policy-gradient estimators via baselines and control variates.
The optimal-baseline result and the variance-decomposition methodology
underpin our MSE analysis. The "score function is mean-zero" convention we use
in `proofs.md` comes from this line.

**Hoeffding 1948.** The U-statistic and its Hoeffding decomposition: any
symmetric estimator built from $m$ i.i.d. samples decomposes into orthogonal
projections whose variances scale as $1/m, 1/m^2, \dots$. This is the engine
behind finite-sample MSE results for group-based estimators.

**Zhou et al. 2026 — "Demystifying GRPO: Its Policy Gradient is a U-Statistic"
(arXiv:2603.01162). [✓]** The sharpest recent single-reward theory: frames the
GRPO gradient as a U-statistic, characterizes its finite-sample MSE, proves
oracle equivalence (GRPO is asymptotically equivalent to an oracle
value-function algorithm), and establishes a *universal scaling law* for the
optimal group size. **This is the single-reward tool we extend.** Our Theorem 3
is the multi-reward generalization: where they have a scalar variance, we have
the quadratic form $w^\top C w$ and a correlation-dependent MSE floor. We
credit them with the U-statistic framing and the single-reward MSE / oracle /
group-size results — Theorem 3 builds *on top of* their machinery.

> **Caution on group size (do not conflate).** Zhou et al.'s scaling law
> (their Theorem 7) states the optimal group size is **universal — independent
> of the training budget**. Our `fig2c` finding ($m^\star\propto N^{1/3}$,
> *grows* with the budget) is **not** the same law and must **not** be framed
> as "the multi-reward read of theirs." They answer "what intrinsic $m$ is
> optimal at a fixed prompt set"; our `fig2c` answers a *different* question —
> "how to split a fixed total rollout budget $N=Pm$ between prompts $P$ and
> group size $m$." These are distinct optimization problems with distinct
> answers; presenting ours as an extension of theirs would be wrong. We
> therefore demote our group-size result to a scoped budget-allocation remark
> (see `proposed-solutions.md` Solution 3) and keep Zhou's universal law as the
> single-reward statement we cite, not extend.

**Zeng, Zhou, Arora & Zanette 2025 — "Shrinking the Variance: Shrinkage
Baselines for RLVR" (arXiv:2511.03710). [✓]** The competing theory group. They
reduce GRPO gradient variance via Stein/James–Stein shrinkage baselines that
combine per-prompt and across-prompt means. This is an *alternative,
complementary* variance-reduction lever: they shrink the baseline (an
across-prompt mechanism); we analyze decoupled per-channel normalization (a
cross-channel mechanism). One could in principle shrink *and* decouple. Our
contribution is multi-reward-native and addresses the *cross-channel*
correlation structure that single-channel shrinkage does not model.

---

## 4. Strand C — multi-objective / multi-reward GRPO (the closest prior work)

This is the strand we are closest to, and where we must be most careful about
attribution. **The decoupled (normalize-then-aggregate) estimator and the
variance-domination observation are prior art.** Our contribution is the
finite-sample theory, not the method.

**MO-GRPO — Mitigating Reward Hacking of GRPO on Multi-Objective Problems
(arXiv:2509.22047, CyberAgent / NAIST, Sept 2025). [✓] — CLOSEST PRIOR WORK.**
It predates GDPO and, on a full read of its theory section, **already contains
our Proposition 1 — both the AN and NA influence laws, including the
cross-channel covariance terms.** Specifically:
  - **proposes the decoupled fix** (= our "NA"): *"MO-GRPO normalizes each
    reward dimension independently before aggregation,"* whereas GRPO
    aggregates then normalizes;
  - **Theorem 1 (AN):** $\mathrm{Corr}(R_i, A_g) = \big(\sigma_i^2 +
    \sum_{j\ne i}\mathrm{Cov}(R_i,R_j)\big)/(\sigma\,\sigma_i)$ — the advantage
    is dominated by the highest-variance channel. This is **algebraically our
    $\mathrm{infl}_\ell^{\mathrm{AN}}$** (at unit weights);
  - **Theorem 2 (NA) + Corollary 1:** $\mathrm{Corr}(R_i, A_g^{\mathrm{MO}})
    \to 1/\sqrt K$ (constant, variance-independent when channels are
    uncorrelated) — this is **our $\mathrm{infl}_\ell^{\mathrm{NA}}$**;
  - offers an equalization-style guarantee (all rewards contribute evenly,
    preference order preserved).

**Attribution consequence (important).** We do **not** claim the NA estimator,
**nor the influence law** — MO-GRPO's Theorems 1–2 already give it, cross-
channel terms and all. Our **Proposition 1 is therefore a restatement of
MO-GRPO Thm 1–2** in our notation with general weights $w_\ell$ (a trivial
extension of their unit-weight result); it is used only as background, not as
a contribution. Our genuinely new results — which MO-GRPO does **not** contain
(it has no finite-sample MSE, no U-statistic, no correlation-matrix gradient
variance, no resolution bound, no conditioning analysis) — are **Theorem 3**
(the correlation-floor MSE) and **Proposition 4** (the conditioning-bias law).

**GDPO — Group reward-Decoupled Normalization Policy Optimization
(arXiv:2601.05242, NVlabs, Jan 2026). [✓]** A second, independent realization
of decoupled normalization, with an explicit *"reward signal collapse"*
framing: aggregating then normalizing maps distinct reward combinations to
identical advantages, *"reducing the resolution of the training signal."* GDPO
is the empirical anchor for the **collapse phenomenon** that our Proposition 2
formalizes (the sum-lattice-vs-product-lattice counting bound). GDPO ships an
NVlabs reference implementation and reports gains over GRPO on tool-calling,
math, and coding. As with MO-GRPO: GDPO contributes the method and the
phenomenon; we contribute the rigorous resolution bound and the MSE theory.

**Stratified GRPO — Stratified Advantage Normalization (arXiv:2510.06214,
Oct 2025). [✓]** Normalizes advantages *within homogeneous strata* of
trajectories (for LLM search agents with structural heterogeneity); its
Theorem quantifies between-stratum bias. Adjacent to our conditioning analysis
— stratified/subgroup normalization is the same family as our subgroup-baseline
conditioning (Prop 4′) — but it targets *trajectory* heterogeneity, not
*cross-reward* gating, and has no conditioning-bias law.

**Blockwise Advantage Estimation (arXiv:2602.10231, Feb 2026). [✓]** Nearest
neighbor on the "conditional baseline" axis: assigns each objective its own
advantage applied to its text block, with an *Outcome-Conditioned Baseline*
that stratifies samples by a prefix-derived intermediate outcome. Crucial
distinction: their conditioning is **within-trajectory** (block $t$ conditioned
on a prefix outcome), whereas ours is **cross-reward** (channel $b$ conditioned
on channel $a$ in the same rollout). Their outcome-conditioned baseline is
structurally analogous to our *subgroup* baseline (Prop 4′) and reinforces our
finding that within-group stratified conditioning needs care to stay unbiased.

**Simultaneous Multi-objective Alignment across verifiable and non-verifiable
rewards (arXiv:2510.01167, Oct 2025). [✓]** Broader multi-objective alignment;
situates the multi-reward problem but does not analyze the GRPO advantage
estimator's finite-sample behavior. Cite at landscape level.

**"Why GRPO Needs Normalization: A Local-Curvature Perspective"
(arXiv:2601.23135, 2026). [✓]** Explains GRPO's normalization through an
adaptive-gradient / local-curvature lens — a complementary *why-normalize*
account to our variance/MSE account. Worth citing as an alternative
theoretical perspective on the same normalization step.

**Mroueh 2025 — GRPO as KL-regularized contrastive loss (arXiv:2503.06639).**
An "effective-loss" view of GRPO; situates the gradient estimator in a
loss-function framework, orthogonal to our finite-sample variance question.
**[verify ID before submission]**

**BASIS — Batchwise Advantage Estimation from Single-Rollout Information
Sharing (arXiv:2605.27293, 2026). [✓]** Shares advantage information across a
batch to stabilize single-rollout estimation. Adjacent variance-reduction idea
on the *sampling* axis; orthogonal to the cross-channel question. Cite at
landscape level.

### 4a. Closely adjacent / concurrent work (surfaced in the 2026-05-31 attribution audit)

These four were not in the first draft; a thorough web re-check surfaced them as
sufficiently close that honest scholarship requires citing and distinguishing
them. **None proves our theorems** (no finite-sample $w^\top C w$ MSE law, no
sign-changing conditioning-bias closed form), so they do not pre-empt our
contributions — but they touch the same mechanisms and must be credited.

**CANON — Conditional Advantage Estimation for RL in Large Reasoning Models
(Chen, Li, Jiang, Qian, Ren, Yang, Cheng, Liu, Shao; arXiv:2509.23962,
Sept 2025). [✓] — closest prior work on *conditioning*.** Regroups sampled
responses by a continuous target metric (length, entropy) into higher/lower
groups and does inter-group comparison to amplify that metric's effect. This is
*within-metric advantage shaping*; our **Prop 4** is a *cross-reward* gate
("$b$ counts only if $a$ passes") with a closed-form, sign-changing bias law and
the explicit threshold $\gamma^\star$. Prior, related, distinct — cite next to
Prop 4.

**RDPO — Multi-Objective and Mixed-Reward RL via Reward-Decorrelated Policy
Optimization (Bai, Liu, Zhuang, Zhou, Weng, Chen, Wang, Cai; arXiv:2605.13641,
2026). [✓] — concurrent, closest on *reward correlation*.** Identifies
correlated reward dimensions as a source of unstable scalar advantages and
removes the redundancy via Magnitude-Aware Quantile Normalization + Mahalanobis
whitening (applied to LongCat-Flash post-training). This is a decorrelation
*method*; our **Thm 3** is the finite-sample *theory* of the decoupled
estimator — it quantifies the MSE *cost* ($w^\top C w$ floor) of leaving the
channels correlated, which is exactly what RDPO's whitening removes by
construction. Concurrent and complementary — cite next to Thm 3.

**Your Group-Relative Advantage Is Biased (Yang, Chen, Wang, Lu, Chai, Yin,
Lin, Ma, Zhuang, Wang, Yang, Li, Ban; arXiv:2601.08521, Jan 2026). [✓]** Shows
the group-relative advantage is biased by *prompt difficulty* (underestimates
hard prompts, overestimates easy ones) and proposes a difficulty-reweighting
fix. This is a *prompt-difficulty* bias, complementary to (i) the *cross-reward*
conditioning bias of Prop 4 and (ii) the self-normalization $\hat\sigma$ bias of
Thm 3. Cite in the bias discussion.

**On the Hidden Objective Biases of Group-based RL (Fontana, Simoni, Rossolini,
Saracino, Mori; arXiv:2601.05002, Jan 2026). [✓]** Catalogs further structural
biases of GRPO (non-uniform group weighting → gradient biases on shared prefix
tokens; AdamW interactions making training insensitive to reward scale). Adjacent
"biases of group RL" landscape; cite alongside the above.

**Wider multi-objective RLHF (cite at survey level, do not re-run):**
reward-soup / model-merging approaches, multi-objective PPO, and
preference-based methods (DPO/IPO/SimPO; see also "Your GRPO Is Secretly DPO",
arXiv:2510.00977). These are *pair-based* or *weight-space* methods — a
different paradigm from group-based GRPO. The apt baselines for our claims are
**AN-GRPO** (which we run), **single-reward GRPO** (which we run), and the
**NA** decoupled estimator of MO-GRPO/GDPO (which we analyze).

---

## 5. The gap, stated precisely

Multi-reward RLVR has defaulted to GRPO. **MO-GRPO and GDPO already give the
decoupled (normalize-then-aggregate) method**, and MO-GRPO already observed —
qualitatively — that aggregation lets the highest-variance channel dominate.
But these are *empirical / method* contributions: MO-GRPO offers an
equalization guarantee, GDPO an empirical collapse demonstration. Neither gives
a **finite-sample, correlation-aware estimator analysis**. Meanwhile the
single-reward estimator theory (Zhou et al.'s U-statistic; Zeng et al.'s
shrinkage; the classical baseline literature) is sharp but **stops at one
reward**: no reward *correlation matrix*, no closed-form cross-channel
influence law, no resolution bound, no conditioning bias.

**The gap:** *no prior work gives the finite-sample, correlation-aware theory
of the decoupled and conditioned multi-reward advantage estimators* — i.e. the
MSE floor as a quadratic form $w^\top C w$, the lattice resolution bound, and
the conditioning-bias law. That theory is our contribution.

---

## 6. What is prior art vs what is ours (explicit)

| Component | Status | Owner |
|---|---|---|
| NA (normalize-then-aggregate) **method** | prior art | MO-GRPO (2509.22047), GDPO (2601.05242) |
| Influence law (AN dominated by high-variance channel; NA variance-independent), **incl. cross-channel covariance terms** | prior art | **MO-GRPO Thm 1 & 2** |
| "Reward signal collapse" phenomenon (qualitative) | prior art | GDPO |
| Single-reward U-statistic MSE / oracle / **universal (budget-independent) group-size law** | prior art | Zhou et al. (2603.01162) |
| **Prop 1** — influence law restated in our notation with general weights $w_\ell$ | **background, not a contribution** (= MO-GRPO Thm 1–2) | MO-GRPO |
| **Prop 2** — sum-lattice vs product-lattice *resolution bound* with $\mathbb Q$-independence condition | **ours** (formalizes GDPO's collapse; pending GDPO full-text check) | — |
| **Thm 3** — finite-sample MSE $=(\tau^2/m)\,w^\top C w + O(m^{-2})$; correlation floor | **ours, genuinely new** (multi-reward extension of Zhou's single-reward MSE) | — |
| group-size remark $m^\star\propto N^{1/3}$ (fixed-budget prompt/rollout split) | **ours, scoped remark** — distinct from Zhou's universal law, **not** an extension of it | — |
| **Prop 4 / 4′** — sign-changing conditioning-bias law + zero-fill recommendation | **ours, genuinely new** (no prior analog found) | — |

**Honest framing.** This is a *theory-of-an-existing-method* paper. We do not
claim the decoupled estimator, nor the influence law (both MO-GRPO's), nor the
collapse phenomenon (GDPO's). **The genuinely new contributions are Theorem 3
(the correlation-floor MSE) and Proposition 4 (the conditioning-bias law);**
Proposition 2 is a rigorous resolution bound formalizing GDPO's collapse; and
Proposition 1 is a restatement of MO-GRPO's Theorems 1–2 included only as
background.

**One-sentence positioning (for the intro):**
> "MO-GRPO and GDPO introduced decoupled multi-reward normalization and showed
> — MO-GRPO analytically, GDPO empirically — that aggregation lets the
> highest-variance objective dominate and collapses distinct reward
> combinations; the single-reward U-statistic and shrinkage lines give
> finite-sample estimator theory for one reward. We supply the missing
> finite-sample, **correlation-aware** theory for the *multi-reward* setting:
> a reward-correlation MSE floor $w^\top C w$ (Thm 3) and a sign-changing
> conditioning-bias law (Prop 4), together with a rigorous resolution bound
> (Prop 2), verified in simulation and on real LLM rollouts at two model
> scales."

---

## 7. Must-cite checklist (status as of 2026-05-31 web re-check)

All arXiv IDs and author lists below were re-verified against arXiv on
2026-05-31 (titles, IDs, and author lists confirmed; entries 1–17 plus the four
adjacent-work additions 18–21).

| # | Reference | arXiv | Status |
|---|---|---|---|
| 1 | Shao et al. 2024 — DeepSeekMath / GRPO | 2402.03300 | ✓ verified |
| 2 | Guo et al. 2025 — DeepSeek-R1 | 2501.12948 | ✓ verified |
| 3 | Yu et al. 2025 — DAPO | 2503.14476 | ✓ verified |
| 4 | Liu et al. 2025 — Dr. GRPO ("Understanding R1-Zero-Like Training") | 2503.20783 | ✓ verified |
| 5 | **MO-GRPO** — multi-objective GRPO (Ichihara et al.) | **2509.22047** | **✓ verified — CLOSEST PRIOR WORK; Thm 1–2 = our Prop 1** |
| 6 | **GDPO** — decoupled normalization (NVIDIA) | **2601.05242** | **✓ verified — empirical anchor** |
| 7 | **Zhou et al.** — GRPO U-statistic | **2603.01162** | **✓ verified — core single-reward tool** |
| 8 | **Zeng et al.** — shrinkage baselines | **2511.03710** | **✓ verified — competing lever** |
| 9 | **Blockwise Advantage Estimation** | **2602.10231** | **✓ verified — within-trajectory conditioning** |
| 10 | **Stratified GRPO** | **2510.06214** | **✓ verified — within-stratum normalization** |
| 11 | Simultaneous Multi-objective Alignment | 2510.01167 | ✓ verified (landscape) |
| 12 | "Why GRPO Needs Normalization" (local curvature) | 2601.23135 | ✓ verified (alt. perspective) |
| 13 | BASIS — batchwise advantage estimation | 2605.27293 | ✓ verified (landscape) |
| 14 | Mroueh 2025 — GRPO as contrastive loss | 2503.06639 | ✓ verified |
| 15 | Greensmith, Bartlett & Baxter 2004 | JMLR v5, 1471–1530 | ✓ verified (classical) |
| 16 | Hoeffding 1948 — U-statistics | Ann. Math. Stat. 19(3) | classical |
| 17 | Fisher 1921 / Anderson — sample-correlation bias | Metron 1; Wiley 2003 | classical (used in Thm 3 proof) |
| 18 | **CANON — Conditional Advantage Estimation** (Chen et al.) | **2509.23962** | **✓ verified — closest prior work on *conditioning* (cf. Prop 4)** |
| 19 | **RDPO — Reward-Decorrelated Policy Optimization** (Bai et al.) | **2605.13641** | **✓ verified — concurrent, closest on *reward correlation* (cf. Thm 3)** |
| 20 | **Your Group-Relative Advantage Is Biased** (Yang et al.) | **2601.08521** | **✓ verified — prompt-difficulty advantage bias (cf. Prop 4 / Thm 3)** |
| 21 | On the Hidden Objective Biases of Group-based RL (Fontana et al.) | 2601.05002 | ✓ verified (biases landscape) |
