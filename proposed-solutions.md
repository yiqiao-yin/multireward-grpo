# Proposed Solutions

*The theory section. We state our four results as the answers to the questions
raised in `problem-statement.md`, give the intuition and what each resolves,
and point to `proofs.md` (appendix) for full rigor. Empirical verification is
in `empirical-section.md`.*

> **Attribution (read first).** The **decoupled estimator** below (NA =
> normalize-then-aggregate) is **not ours** — it is the method of MO-GRPO
> (arXiv:2509.22047) and GDPO (arXiv:2601.05242). Concretely:
> - **Thm 3** (correlation-floor MSE) and **Prop 4** (sign-changing
>   conditioning-bias law) are the **genuinely new results** — no prior analog.
> - **Prop 2** (lattice resolution bound) is our **rigorous formalization** of
>   GDPO's "collapse" phenomenon.
> - **Prop 1** (influence law) is **not a contribution** — it is a restatement
>   of **MO-GRPO's Theorems 1–2** (which already give the AN and NA
>   reward-advantage correlation, cross-channel terms included) in our notation
>   with general weights. We include it only as background that motivates
>   Thm 3.
>
> We lead the paper with Thm 3 and Prop 4. See `literature-review.md` §6 for
> the full prior-art-vs-ours table.

---

## Notation (recap from problem-statement §2)

For a prompt $x$, sample $m$ rollouts $\{y_j\}_{j=1}^m$, each scored on $R$
verifiable channels $r_j = (r_j^{(1)}, \dots, r_j^{(R)}) \in \mathbb{R}^R$,
i.i.d. given $x$. Let

$$
\Sigma = \mathrm{Cov}(r_j \mid x), \quad D = \mathrm{diag}(\Sigma), \quad
C = D^{-1/2}\Sigma D^{-1/2}\ \text{(reward correlation matrix)},
$$

with per-channel std $\sigma_\ell$, objective weights $w \in \mathbb{R}^R_{\ge 0}$,
and standardized reward $z_j = D^{-1/2}(r_j - \mu)$. The two estimator
orderings under study:

- **AN — Aggregate-then-Normalize (the GRPO / scalarize baseline):**
  $\;s_j = w^\top r_j,\quad A_j^{\mathrm{AN}} = (s_j - \bar s)/\hat\sigma_s.$
- **NA — Normalize-then-Aggregate (the decoupled estimator of MO-GRPO / GDPO;
  the object we analyze):**
  $\;A_j^{\mathrm{NA}} = \sum_\ell w_\ell\,(r_j^{(\ell)} - \bar r^{(\ell)})/(\hat\sigma_\ell + \delta).$

Gradient estimator $\hat g = \frac1m \sum_j A_j\,g_j$ with score $g_j = v^\top
\nabla_\theta \log \pi_\theta(y_j\mid x)$, $\mathbb E[g_j]=0$,
$\mathrm{Var}(g_j) = \tau^2$, $g_j \perp r_j$.

---

## Solution 1 — Influence law (background; this is MO-GRPO's Theorems 1–2, not a contribution)

> **Attribution.** This result is **MO-GRPO's** (their Theorem 1 for AN, Theorem
> 2 + Corollary 1 for NA), including the cross-channel covariance terms. We
> restate it in our notation with general weights $w_\ell$ (their result uses
> unit weights; the extension is trivial) **only as background** — it motivates
> Theorem 3 but is not claimed as a contribution. The genuinely new results are
> Thm 3 and Prop 4.

**Statement (restated from MO-GRPO Thm 1–2).** Define channel $\ell$'s
standardized influence on the advantage as
$\mathrm{infl}_\ell(A) := \mathrm{Cov}(A, z_\ell)$. Under the oracle estimators,

$$
\mathrm{infl}_\ell^{\mathrm{AN}} = \frac{\sum_k w_k\,\sigma_k\,C_{k\ell}}{\sqrt{w^\top\Sigma w}},
\qquad
\mathrm{infl}_\ell^{\mathrm{NA}} = \sum_k w_k\,C_{k\ell}.
$$

At $C = I$: $\mathrm{infl}_\ell^{\mathrm{AN}} = w_\ell\sigma_\ell/\sqrt{w^\top\Sigma w}$
(scales with the channel's **standard deviation**) and
$\mathrm{infl}_\ell^{\mathrm{NA}} = w_\ell$ (scales with the **weight** only).
Self-normalized versions inherit these to $O(m^{-1})$.

**What it solves.** It explains *why scalarize-then-normalize misbehaves*:
under AN, a high-variance channel — typically the harder objective — dominates
the advantage **regardless of its weight** $w_\ell$, and raising $w_\ell$ moves
its influence only sublinearly (because $w_\ell$ also enters the denominator
$\sqrt{w^\top\Sigma w}$). So you *cannot* fix the imbalance by reweighting —
the mechanism behind the multi-objective reward-hacking that MO-GRPO reported
and the "reweighting fails" behavior GDPO observed. NA removes the
$\sigma_\ell$ factor, restoring weight-proportional control.

**Intuition.** AN standardizes the *aggregate*, so each channel enters the
advantage in proportion to how much it moves the aggregate, i.e. its $\sigma$.
NA standardizes each channel *first*, equalizing their scales before the
weighted sum — so the weight is what you actually dialed.

*Proof:* `proofs.md` §"Proposition 1". Oracle case exact; finite-sample
correction via delta method.

---

## Solution 2 — Resolution bound (the rigorous *formalization* of GDPO's "reward signal collapse")

> **What's new here.** GDPO described "reward signal collapse" — distinct
> reward combinations mapping to identical advantages, "reducing the
> resolution of the training signal" — empirically. Our contribution is the
> exact counting bound (sum lattice vs product lattice) and the precise
> condition under which decoupling recovers full resolution ($\mathbb
> Q$-linear independence of the $w_\ell/\sigma_\ell$, i.e. heterogeneous
> scales). We formalize a known phenomenon.

**Claim.** With discrete rewards on $L$ levels per channel, group
normalization is affine-invariant, so the number of *distinct advantage
values* equals the number of distinct values of the underlying linear
functional. Then:

- **AN** realizes at most the **sum lattice**: $R(L-1)+1$ distinct values
  (equal weights). Different reward combinations with the same weighted sum
  collapse to the same advantage.
- **NA** realizes the full **product lattice** $L^R$ — *but only when the
  per-channel scales are heterogeneous*. The exact condition, with
  $a_\ell = w_\ell/\sigma_\ell$, is a **bounded-integer** one: no nonzero
  integer vector $n$ with $|n_\ell| \le L-1$ satisfies
  $\sum_\ell a_\ell n_\ell = 0$. Under equal scales, NA reduces to a scalar
  multiple of AN and the gain vanishes.

> **Correction (revision 1).** This was previously stated as "$L^R$ **iff**
> $\{w_\ell/\sigma_\ell\}$ are $\mathbb Q$-linearly independent." Rational
> independence is **sufficient but not necessary**: only integer relations with
> coefficients $|n_\ell| \le L-1$ are realizable as differences of grid points,
> so a rational relation needing larger coefficients never materializes.
> Counterexample: $R=2$, $L=2$, $a=(1,3)$ are rationally dependent
> ($3a_1 - a_2 = 0$) yet the four grid points map to $\{0,1,3,4\}$, all
> distinct, so $N_{\mathrm{NA}} = 4 = L^R$. Caught in peer review; see
> `proofs.md` §Prop 2.

**What it solves.** This is the formal version of GDPO's "scalarize-then-
normalize collapses distinct reward combinations into identical advantages."
The collapse is a *lattice* phenomenon: aggregation projects the $R$-dim reward
grid onto a 1-D sum, losing resolution. Decoupling preserves the full grid —
provided the channels have genuinely different scales, which is the realistic
accuracy-vs-length regime.

**Intuition.** "Correct + short" and "wrong + long" can have the same weighted
sum and thus the same AN advantage even though they are opposite behaviors.
NA keeps them distinct because their per-channel standardized values differ.

*Proof:* `proofs.md` §"Proposition 2". Tight on both bounds; the equal-scale
counterexample is necessary.

---

## Solution 3 — Finite-sample MSE law and the correlation floor (the headline; genuinely new)

> **What's new here.** This is one of the two genuinely new results (with
> Prop 4). No prior multi-reward work gives a finite-sample MSE as a quadratic
> form in the reward correlation matrix. It is the multi-reward extension of
> Zhou et al.'s single-reward U-statistic MSE — where they have a scalar
> variance, we have $w^\top C w$.

**Claim.** Under finite fourth-moment conditions, the decoupled
self-normalized gradient estimator satisfies the **exact** identity (for all
$m \ge 2$)

$$
\mathrm{MSE}(\hat g^{\mathrm{NA}}) = \frac{\tau^2}{m}\, w^\top \mathbb E[\hat C]\, w,
$$

and hence asymptotically

$$
\boxed{\;\mathrm{MSE}(\hat g^{\mathrm{NA}}) = \frac{\tau^2}{m}\, w^\top C\, w + O(m^{-2}).\;}
$$

**What it solves — the correlation floor.** The leading $1/m$ coefficient is
the quadratic form $w^\top C w$. Therefore:

- **positively correlated objectives are fundamentally harder to optimize** —
  they inflate $w^\top C w$ and raise the achievable gradient-MSE floor at a
  fixed sampling budget;
- **antagonistic (negatively correlated) objectives are easier** — they lower
  the floor.

This is the genuinely new content: a single-reward theory has only a scalar
variance; the multi-reward floor is governed by the *off-diagonal* structure
of $C$. It tells a practitioner, before training, which reward pairs will be
cheap or expensive to optimize jointly.

**Scoped remark (group size) — distinct from Zhou et al.'s law; not a headline
claim.** An earlier draft conjectured $m^\star \propto \sqrt{w^\top C w}$;
**this is false** (the harness falsified it) — reward correlation sets the MSE
*floor*, not the group size. In our *fixed-total-budget* setup ($N = Pm$, where
one trades the number of prompts $P$ against the group size $m$), the
budget-optimal split grows as $m^\star \propto N^{1/3}$, independent of reward
correlation (`fig2c`).

> **Do not conflate with Zhou et al.** Zhou et al.'s scaling law (their Thm 7)
> says the optimal group size is **universal — independent of the training
> budget**. Our $N^{1/3}$ result answers a *different* question (how to split a
> fixed rollout budget between prompts and group size), not "what intrinsic
> $m$ is optimal." We present this only as a scoped budget-allocation remark
> and do **not** claim it as a multi-reward extension of Zhou's universal law.
> The headline of Theorem 3 is the **correlation floor** $w^\top C w$, which is
> independent of this group-size discussion.

**Why NA and not AN.** The same machinery applied to AN gives a different
leading coefficient that depends on the scalarization; AN's gradient is
dominated by the high-$\sigma$ channel (Solution 1), so its effective MSE does
not track the intended $w^\top C w$. NA is the estimator for which the clean
floor holds.

*Proof:* `proofs.md` §"Theorem 3". The oracle case is a one-line exact
variance calculation; the self-normalized lift uses the exact sample-
correlation identity $u_{j,\ell}u_{j,k}\!\to\!\hat C_{\ell k}$ plus the
classical $\mathbb E[\hat C] = C + O(1/m)$ expansion — no Hoeffding decomposition
needed.

---

## Solution 4 — Conditioning bias law (explains "length counts only if correct"; genuinely new)

> **What's new here.** The second genuinely new result. GDPO observed that
> conditioning helps where reweighting fails; we give the closed-form,
> sign-changing bias law and the threshold $\gamma^\star$, plus the
> implementation (zero-fill) that keeps the Theorem-3 structure. No prior
> conditioning-bias analysis exists for this setting.

**Setup.** Model the gate "$b$ counts only if $a$ passes" as
$r_j^{(b),\mathrm{cond}} = r_j^{(b)}\,\mathbb 1[r_j^{(a)} = 1]$, with
$p_a = \Pr[r^{(a)}=1]$, $p_b = \Pr[r^{(b)}=1]$, and a synthetic gradient
$g = \alpha_c(r^{(a)}-p_a) + \alpha_f(r^{(b)}-p_b)r^{(a)} + \gamma(r^{(b)}-p_b)(1-r^{(a)}) + \varepsilon$,
where $\gamma$ is the **contamination**: how strongly the easier reward drives
the gradient on $a$-failures.

**Claim (sign-changing bias law).** The cross-objective bias that conditioning
removes is the closed form

$$
\boxed{\;\beta_{ab} = w_b\, p_b(1-p_a)\big[\gamma(1-p_b) - \alpha_c\,p_a\big].\;}
$$

It **changes sign** at $\gamma^\star = \alpha_c\,p_a/(1-p_b)$: conditioning
*helps* iff contamination dominates gate-coupling ($\gamma > \gamma^\star$) and
is *harmful* below the threshold.

**What it solves.** GDPO finds that conditioning ("length counts only if
correct") restores control where reweighting failed. We show this is not a
monotone "conditioning always helps" effect — it is a sign-changing law. When
contamination is high (the easier reward is being chased on wrong answers),
conditioning removes a real bias and helps; when contamination is low,
conditioning *over-clips* the legitimate signal and hurts. The threshold is
explicit in $(p_a, p_b, \alpha_c, \gamma)$.

**The earlier draft was wrong twice; both caught by the harness.** (i) The
original sketch dropped a cross-term (the conditioned reward $r^{(b)}r^{(a)}$ is
correlated with the gate $r^{(a)}$); the corrected form above carries it. (ii)
A claimed $\Theta(1/(mp_a))$ variance penalty does not appear (Solution 4′).

### Solution 4′ — Implementation: use zero-fill, not subgroup baselines

Two implementations of conditioning:

- **subgroup baseline** — center $r^{(b)}$ over only the passing rollouts
  ($r^{(a)}=1$). This is **data-dependent within the group**: the passing set
  is random and, when $mp_a \lesssim 2$, *empty*. It breaks the i.i.d. symmetry
  Theorem 3 relies on, and its MSE is **non-monotone** in $p_a$.
- **zero-fill** — set $r^{(b)}r^{(a)}$ and center it over *all* $m$ rollouts.
  This **preserves the U-statistic structure**, so Theorem 3 applies directly
  to the conditioned channel.

**Recommendation: zero-fill.** It avoids the degeneracy, attains $\approx
0.86\times$ the unconditioned MSE in simulation, and wins on $\approx 75\%$ of
the $(p_a, \gamma)$ plane, with the decision boundary tracking the bias-zero
indifference curve $\gamma = \alpha_c p_a/(1-p_b)$.

*Proof / scope:* `proofs.md` §"Proposition 4". Proven under the synthetic
generative model (which is exactly what the experiments simulate). Lifting to a
general data-dependent gate is explicit future work; zero-fill + Theorem 3 is
the constructive workaround that makes the deferral harmless.

---

## How the four solutions interlock

```
 Prop 1 (influence)  ─┐
                      ├─► explain GDPO finding (i)+(ii): scalarize collapses,
 Prop 2 (resolution) ─┘   decoupling fixes it, reweighting can't
                      
 Thm 3 (MSE floor)   ───► the quantitative headline: which reward pairs are
                          cheap/expensive to co-optimize (w^T C w)
                      
 Prop 4 (+4′)        ───► explain GDPO finding (iii): conditioning is a
                          sign-changing bias remover; implement via zero-fill,
                          which inherits Thm 3
```

All four are corroborated in `empirical-section.md`: Prop 1/2/Thm 3 in the
synthetic harness (3-digit agreement) and on real GSM8K rollouts at two model
scales; Prop 4 on real LLM rewards; and Prop 1's channel-allocation prediction
shows up in two GRPO-fine-tuned models (NA stays weight-balanced; AN distorts
toward the high-variance channel).

---

## Status of each solution

| Solution | Novelty | Proof status | Verified |
|---|---|---|---|
| 1 — Influence law | **not a contribution** — = MO-GRPO Thm 1–2 (we add weights) | Proven (oracle exact; self-norm $O(m^{-1})$) | Synthetic (`fig1b`); trained models (NA vs AN channel allocation) |
| 2 — Resolution bound | ours — formalizes GDPO's "collapse" (lattice bound; pending GDPO full-text check) | Proven (both directions tight) | Synthetic enumeration (`figP2`) |
| 3 — MSE floor (correlation) | **genuinely new** (extends Zhou et al. to multi-reward) | Proven (exact $m\ge2$ identity + standard expansion) | Synthetic 3-digit (`fig1a/1c/S1/S2`); real GSM8K, Qwen-1.5B + 7B |
| 3 — group-size remark $m^\star\propto N^{1/3}$ | scoped budget-allocation remark; **distinct from Zhou's universal law**, not an extension of it | Empirical | Synthetic (`figM1`, `fig2c`) |
| 4 — Conditioning bias law | **genuinely new** (no prior analog) | Proven under synthetic model | Synthetic (`figT2`, $2\times10^{-3}$); real GSM8K |
| 4′ — Zero-fill recommendation | ours (cf. Blockwise/Stratified subgroup baselines) | Operational + structural argument | Synthetic (`figT1/T3`) |
| 4 — general data-dependent gate | future work | — (zero-fill workaround) | — |

**Bottom line on novelty.** Genuinely new and the headline: **Thm 3**
(correlation-floor MSE) and **Prop 4** (+4′, conditioning-bias law). Our
formalization of a known phenomenon: **Prop 2** (GDPO's collapse). **Not
contributions** (prior art, included as background): the NA method (MO-GRPO /
GDPO) and the influence law **Prop 1** (= MO-GRPO Thm 1–2). The group-size
$N^{1/3}$ result is a scoped remark, explicitly *not* an extension of Zhou's
universal group-size law.
