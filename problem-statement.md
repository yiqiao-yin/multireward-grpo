# Problem Statement

**Title.** *Decoupling and Conditioning Reshape Influence Allocation and the
Gradient-Noise Floor in Multi-Reward GRPO under a Finite-Sample U-Statistic
Analysis.*

> **Document map.** This file defines the problem (gap, formal setup, the
> questions we answer). The companion files are:
> - `literature-review.md` — full related work + citation positioning
> - `proposed-solutions.md` — the results (Thm 3 & Prop 4 are ours; Prop 2 ours; Prop 1 = MO-GRPO restated as background) with intuition
> - `proofs.md` — full proofs (appendix)
> - `empirical-section.md` — synthetic + real-LLM + fine-tuning verification
> - `huggingface_assets.md` — released datasets & models

---

## 1. The gap (summary; full treatment in `literature-review.md`)

Multi-reward RLVR has defaulted to GRPO. Two recent works — **MO-GRPO**
(arXiv:2509.22047) and **GDPO** (arXiv:2601.05242) — already introduced the
**decoupled (normalize-then-aggregate) estimator** and showed, empirically,
that **how** the reward channels are combined materially changes training:

- **(i)** scalarize-then-normalize *collapses* distinct reward combinations
  into identical advantages (GDPO's "reward signal collapse");
- **(ii)** the advantage is dominated by the highest-variance channel, so
  reweighting alone fails (MO-GRPO's reward-hacking observation); decoupled
  per-channel normalization fixes it;
- **(iii)** *conditioning* an easier reward on a harder one ("length counts
  only if correct") restores control (GDPO).

These are **method + empirical** contributions: MO-GRPO offers an equalization
guarantee, GDPO an empirical demonstration. **Neither gives a finite-sample,
correlation-aware estimator analysis.** Meanwhile the single-reward statistical
theory (Zhou et al.'s U-statistic MSE / oracle / group-size law; Zeng et al.'s
shrinkage baselines) is sharp but **stops at one reward** — no reward
correlation matrix, no closed-form cross-channel influence law, no resolution
bound, no conditioning bias.

**The gap:** no prior work gives the finite-sample, correlation-aware *theory*
of the decoupled and conditioned multi-reward advantage estimators. This work
supplies it.

> **Attribution.** We do **not** propose the decoupled estimator (MO-GRPO /
> GDPO) **nor the influence law** (MO-GRPO's Theorems 1–2 already give it,
> cross-channel terms included; our Prop 1 just restates it with weights, as
> background). Our genuinely new contributions are a reward-correlation MSE
> floor (**Thm 3**) and a sign-changing conditioning-bias law (**Prop 4**); we
> also contribute the lattice resolution bound (**Prop 2**, formalizing GDPO's
> collapse). See `literature-review.md` §6 for the full prior-art-vs-ours
> table.

---

## 2. Setup and notation (canonical — referenced by all companion files)

For a prompt $x$, sample a group of $m$ rollouts $\{y_j\}_{j=1}^m$, each scored
on $R$ verifiable channels (i.i.d. given $x$):

$$
r_j = \left(r_j^{(1)}, \ldots, r_j^{(R)}\right) \in \mathbb{R}^R .
$$

Let

$$
\Sigma = \mathrm{Cov}(r_j \mid x), \qquad
D = \mathrm{diag}(\Sigma), \qquad
C = D^{-1/2}\Sigma D^{-1/2},
$$

with per-channel variance $\sigma_\ell^2 = D_{\ell\ell}$, correlation matrix
$C$, objective weights $w \in \mathbb{R}_{\ge 0}^R$, standardized reward
$z_j = D^{-1/2}(r_j - \mu)$ (so $\mathbb E[z_j]=0$, $\mathrm{Cov}(z_j)=C$), and
within-group sample mean / std $\bar r^{(\ell)}, \hat\sigma_\ell$.

**Two operation orderings.**

- **AN — Aggregate-then-Normalize (GRPO / scalarize baseline).**
  $\;s_j = w^\top r_j,\quad A_j^{\mathrm{AN}} = (s_j - \bar s)/\hat\sigma_s.$
- **NA — Normalize-then-Aggregate (the decoupled estimator of MO-GRPO / GDPO;
  the object we analyze).**
  $\;A_j^{\mathrm{NA}} = \sum_{\ell=1}^{R} w_\ell\,\dfrac{r_j^{(\ell)} - \bar r^{(\ell)}}{\hat\sigma_\ell + \delta}.$

**Gradient estimator and target.** $\hat g(\theta) = \frac{1}{m}\sum_j A_j\,
\nabla_\theta \log \pi_\theta(y_j \mid x)$. For the analysis we study the MSE
of the scalar contraction along a fixed score direction $v$, writing
$g_j = v^\top \nabla_\theta \log \pi_\theta(y_j\mid x)$ with
$\mathbb E[g_j\mid x]=0$, $\mathrm{Var}(g_j\mid x)=\tau^2$, and $g_j\perp r_j$
(the score is fixed by the sampled $y_j$ before the reward is evaluated). The
**target object throughout is the MSE of $\hat g$** relative to the population
gradient.

**Conditioning.** The gate "$b$ counts only if $a$ passes" is
$r_j^{(b),\mathrm{cond}} = r_j^{(b)}\,\mathbb 1[r_j^{(a)} = 1]$, with
$p_a = \Pr[r^{(a)}=1\mid x]$, $p_b = \Pr[r^{(b)}=1\mid x]$.

---

## 3. The questions we answer

The four questions below correspond one-to-one with the four results in
`proposed-solutions.md`. Stated as questions here so the problem is defined
independently of our answers.

**Q1 (influence).** Under AN vs NA, how much does each reward channel actually
influence the advantage — and can a practitioner control that influence by
choosing the weights $w$? *(Answered by **MO-GRPO's Theorems 1–2** — restated
as our Prop 1 for completeness: AN influence ∝ $w_\ell\sigma_\ell$,
uncontrollable by weight alone; NA influence ∝ $w_\ell$. This is prior art,
not our contribution.)*

**Q2 (resolution).** When rewards are discrete, does aggregation lose
information by collapsing distinct reward combinations, and does decoupling
recover it? *(Answered by Prop 2: AN realizes a sum lattice $R(L{-}1)+1$; NA the
product lattice $L^R$, iff scales are heterogeneous.)*

**Q3 (finite-sample MSE).** What is the finite-sample MSE of the decoupled
multi-reward gradient estimator, and how does the reward **correlation
structure** enter? *(Answered by Thm 3: $\mathrm{MSE} = (\tau^2/m)\,w^\top C w +
O(m^{-2})$ — reward correlation sets the MSE floor. A scoped, secondary remark
addresses the budget-allocation group size $m^\star\propto N^{1/3}$; this is
**distinct from** Zhou et al.'s universal, budget-independent group-size law
and is not framed as an extension of it.)*

**Q4 (conditioning).** What bias does conditioning remove, and when does it
help vs hurt? How should it be implemented so the finite-sample theory still
applies? *(Answered by Prop 4: a sign-changing bias law with threshold
$\gamma^\star=\alpha_c p_a/(1-p_b)$; and Prop 4′: implement via zero-fill, which
preserves the U-statistic structure.)*

---

## 4. Verification approach (full results in `empirical-section.md`)

Every claim is checked in three tiers: (1) a synthetic harness that reproduces
each closed form to 3 significant figures; (2) real LLM rollouts on GSM8K at
two model scales (Qwen2.5-1.5B and -7B); (3) a multi-reward GRPO fine-tuning
study on a synthetic fintech domain. Two first-draft claims were
falsified-and-corrected by the harness before write-up (the group-size scaling
and the bias form), so no claim rests on an untested derivation.

---

## 5. Scope and honest limitations

- **Prop 4** is proven under an explicit synthetic generative model (the one
  the experiments simulate). Lifting it to a *general* data-dependent gate is
  future work; the **zero-fill** estimator (Prop 4′) is the constructive
  workaround that preserves the Theorem-3 structure, so the operational
  recommendation does not depend on the open extension.
- **Thm 3's $O(m^{-2})$ remainder** is bounded but its exact constant has a
  closed form only for Gaussian rewards; the discrete-reward constant is a
  one-page supplementary computation. The headline floor uses only the proven
  leading term.
- **Fine-tuning evidence (Tier 3)** supports "multi-reward shaping is
  necessary" and "NA gives weight-proportional influence where AN distorts"
  — it does **not** claim NA achieves higher aggregate reward than AN (the two
  are comparable). See `empirical-section.md` §4 for the honest delineation.
