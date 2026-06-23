# Proofs (Appendix)

> **Role in the paper.** This is the technical appendix. The results are
> stated with intuition in `proposed-solutions.md`; here we give full proofs.
> Notation is the canonical setup of `problem-statement.md` §2 (recapped
> below). Empirical verification of every result is in `empirical-section.md`.

Formal proofs of Proposition 1, Proposition 2, Theorem 3 (oracle + self-normalized),
and Proposition 4 (synthetic-model bias law). Each result is cross-referenced to
its simulation-verifying figure in `figures/`.

The status table at the end (§ "What's proven, what's sketched, what's open")
summarizes the proof state of every numbered claim. In short: four results are
fully proven (Prop 1 oracle exact + finite-sample correction; Prop 2 both
directions; Thm 3 oracle exact, Thm 3 self-normalized exact for $m\ge 2$
plus standard asymptotic expansion; Prop 4 under the explicit generative model);
Proposition 4′ is an operational recommendation backed by `figT1`/`figT3` rather
than a theorem; and two extensions (general data-dependent gate for Prop 4;
closed form of the $O(m^{-2})$ remainder in Thm 3) are explicitly future work,
each with a constructive workaround documented in problem-statement §5.

---

## Notation recap (from §2)

For a prompt $x$, draw $m$ rollouts $\{y_j\}_{j=1}^m$. Each rollout is scored on
$R$ verifiable channels, $r_j = (r_j^{(1)}, \dots, r_j^{(R)}) \in \mathbb{R}^R$,
i.i.d. given $x$. Let

$$
\mu = \mathbb{E}[r_j \mid x], \qquad
\Sigma = \mathrm{Cov}(r_j \mid x), \qquad
D = \mathrm{diag}(\Sigma), \qquad
C = D^{-1/2} \Sigma D^{-1/2}.
$$

Write $\sigma_\ell^2 = D_{\ell\ell}$ and let $z_j = D^{-1/2}(r_j - \mu)$ be the
**standardized reward vector**, so $\mathbb{E}[z_j] = 0$ and $\mathrm{Cov}(z_j) = C$.

Let $w \in \mathbb{R}^R_{\ge 0}$ be the objective weights and $\bar r^{(\ell)},
\hat\sigma_\ell$ the within-group sample mean and standard deviation. The two
estimators we compare are

- **AN** (scalarize-then-normalize):
  $\;s_j = w^\top r_j, \quad A_j^{\mathrm{AN}} = (s_j - \bar s)/\hat\sigma_s.$
- **NA** (normalize-then-aggregate):
  $\;A_j^{\mathrm{NA}} = \sum_{\ell=1}^R w_\ell \dfrac{r_j^{(\ell)} - \bar r^{(\ell)}}{\hat\sigma_\ell + \delta}.$

Throughout the **oracle** variant of an estimator means $\mu, \sigma$ are known
(no estimation): e.g. $A_j^{\mathrm{NA,oracle}} = w^\top z_j$.

The gradient estimator is $\hat g = \frac{1}{m} \sum_j A_j \,\nabla_\theta \log
\pi_\theta(y_j \mid x).$ All MSE statements below condition on a single prompt
$x$; population-level MSE is the prompt-wise integral, which only adds an
$O(1)$ outer term that does not interact with the within-group structure.

**Score convention.** Following Greensmith–Bartlett–Baxter (2004) and the
single-reward U-statistic treatment of Zhou et al. (2026), we study the MSE
along a fixed direction $v$ in score space by writing $g_j := v^\top
\nabla_\theta \log \pi_\theta(y_j \mid x)$. Under standard regularity ($\pi$
differentiable, integrability) the score-function identity gives $\mathbb
E[g_j \mid x] = 0$, and we assume $\mathrm{Var}(g_j \mid x) = \tau^2 \in
(0, \infty)$ and $g_j \perp r_j$ given $x$ (the score is determined by the
sampled $y_j$ before the reward is evaluated). The MSE object throughout is
$$
\mathrm{MSE}(\hat g) \;:=\; \mathbb E\!\left[\big(\hat g - g_\star\big)^2 \,\Big|\, x\right],
\qquad g_\star := \mathbb E[\hat g \mid x] \;=\; 0,
$$
where $g_\star = 0$ holds for both the oracle and the self-normalized
estimator because each $A_j$ has $\mathbb E[A_j] = 0$ and $g$ is mean-zero
independent of $A$.

**Two harness conventions.** The simulation harness uses two slightly
different scalar surrogates for the gradient, corresponding to two limits of
the score convention above:

- *Oracle figures* (`fig1a`, `fig1c`, `fig1b`) use a **constant score**
  $g_j \equiv 1$ — i.e. the harness computes $\bar A = (1/m)\sum_j A_j$ and
  reports its variance. For oracle NA this gives MSE $= w^\top C w / m$
  *exactly* (one-line direct calculation, see Thm 3 oracle below). It
  coincides numerically with the random-score MSE at $\tau^2 = 1$.

- *Self-normalized figures* (`figS1`, `figS2`, `figS3`, `figM1`, `figT*`)
  use a **random score** $g_j \sim N(0,1)$ iid independent of $r_j$ — i.e.
  the harness computes $\hat g = (1/m)\sum_j A_j g_j$. The constant-score
  surrogate cannot be used in the self-normalized case: $A_j^{\mathrm{NA}}$
  has $\sum_j A_j^{\mathrm{NA}} = 0$ by self-centering, so $\bar A^{\mathrm{NA}}
  \equiv 0$ deterministically and the variance would collapse. The
  random-score convention is the non-degenerate one and is what Thm 3
  self-normalized analyzes.

---

## Proposition 1 (Influence law)

**Statement.** Define the *standardized influence* of channel $\ell$ on an
advantage $A$ as $\mathrm{infl}_\ell(A) := \mathrm{Cov}(A, z_\ell)$. Under the
oracle versions of AN and NA,

$$
\mathrm{infl}_\ell^{\mathrm{AN,oracle}}
\;=\; \frac{\sum_k w_k \sigma_k C_{k\ell}}{\sqrt{w^\top \Sigma w}},
\qquad
\mathrm{infl}_\ell^{\mathrm{NA,oracle}}
\;=\; \sum_k w_k C_{k\ell}.
$$

In particular when $C = I$ (uncorrelated channels):
$\mathrm{infl}_\ell^{\mathrm{AN}} = w_\ell \sigma_\ell / \sqrt{w^\top \Sigma w}$
and $\mathrm{infl}_\ell^{\mathrm{NA}} = w_\ell$.

The finite-sample (self-normalized) version of NA satisfies the same identity
up to an additive $O(m^{-1})$ term; AN inherits an analogous $O(m^{-1})$
correction. Concretely, with sample-std $\hat\sigma$ in place of $\sigma$,

$$
\mathrm{infl}_\ell^{\mathrm{NA}}
\;=\; \sum_k w_k C_{k\ell} \;+\; O(m^{-1}).
$$

**Proof.**

*Oracle AN.* Let $s_j = w^\top r_j$, so $A_j^{\mathrm{AN,oracle}} = (s_j -
w^\top\mu)/\sigma_s$ with $\sigma_s^2 = \mathrm{Var}(s_j) = w^\top \Sigma w$.
Using $\mathrm{Cov}(r_k, z_\ell) = \sigma_k C_{k\ell}$ (since $\mathrm{Cov}(r_k,
r_\ell) = \sigma_k \sigma_\ell C_{k\ell}$ and dividing by $\sigma_\ell$ gives
$\sigma_k C_{k\ell}$):
$$
\mathrm{Cov}(A_j^{\mathrm{AN,oracle}}, z_\ell)
= \frac{1}{\sigma_s} \sum_k w_k \mathrm{Cov}(r_k, z_\ell)
= \frac{\sum_k w_k \sigma_k C_{k\ell}}{\sqrt{w^\top \Sigma w}}.
$$
At $C = I$ this collapses to $w_\ell \sigma_\ell / \sqrt{w^\top \Sigma w}$.

*Oracle NA.* $A_j^{\mathrm{NA,oracle}} = \sum_k w_k z_{j,k}$, so
$\mathrm{Cov}(A_j^{\mathrm{NA,oracle}}, z_{j,\ell}) = \sum_k w_k \mathrm{Cov}(z_{j,k},
z_{j,\ell}) = \sum_k w_k C_{k\ell}.$

*Finite-sample correction for NA.* Let $\hat\sigma_\ell = \sigma_\ell(1 +
\epsilon_\ell)$ with $\epsilon_\ell = (\hat\sigma_\ell - \sigma_\ell)/\sigma_\ell
= O_p(m^{-1/2})$. By the delta method,
$\hat\sigma_\ell^{-1} = \sigma_\ell^{-1}(1 - \epsilon_\ell + \epsilon_\ell^2 - \cdots)$.
The leading term reproduces the oracle identity; the next term contributes
$$
-\sigma_\ell^{-1} \mathbb{E}[\epsilon_\ell \cdot z_{j,\ell}^2]
\;=\; O(m^{-1}),
$$
since $\mathbb{E}[\epsilon_\ell] = O(m^{-1})$ (see the σ̂ bias analysis in the
self-normalized lift below) and the cross-moment $\mathbb{E}[\epsilon_\ell z^2]$
is bounded. ∎

**Verifying figure.** `fig1b_influence_law.png`. At $\sigma_2/\sigma_1 = 4$,
$C = I$, equal weights: AN simulated $I_2/I_1 = 4.05$ vs theory $4.00$; NA
simulated $I_2/I_1 = 1.003$ vs theory $1.000$.

---

## Proposition 2 (Resolution bound; qualified to heterogeneous scales)

**Statement.** Let each channel take values in a finite grid of $L$ levels, so
the joint reward $r_j \in \{v_0, \dots, v_{L-1}\}^R$. Let $N_{\mathrm{AN}}(L,
R, w, \sigma)$ and $N_{\mathrm{NA}}(L, R, w, \sigma)$ denote the number of
distinct advantage values realizable in a single group, under AN and NA
respectively. Then:

1. **Sum-lattice upper bound for AN.** For evenly-spaced grids and integer
   weights $w_\ell \in \mathbb{Z}_{>0}$,
   $$
   N_{\mathrm{AN}}(L, R, w, \sigma)
   \;\le\; \big(\textstyle\sum_\ell w_\ell\big)(L-1) + 1.
   $$
   When $w = (1, \dots, 1)$ this is $R(L-1) + 1$. This bound is independent of
   $\sigma$.

2. **Product-lattice realization for NA, iff $\sigma$ heterogeneous.** Let
   $a_\ell = w_\ell / \sigma_\ell$. Then
   $$
   N_{\mathrm{NA}}(L, R, w, \sigma) = L^R
   \iff \{a_\ell\}_{\ell=1}^R \text{ are } \mathbb{Q}\text{-linearly independent}.
   $$
   Sufficient: at least one ratio $a_\ell / a_k$ is irrational. **Necessary
   counterexample:** if all $\sigma_\ell$ are equal, $a_\ell = w_\ell/\sigma$
   is a scalar multiple of $w_\ell$, NA = AN/σ, and $N_{\mathrm{NA}} =
   N_{\mathrm{AN}}$.

**Proof.**

*Affine invariance.* The group-normalization map $A_j = (X_j - \bar X)/\hat
\sigma_X$ is affine in the within-group statistic $X_j$. Affine maps are
injective when $\hat\sigma_X > 0$, so #distinct $A_j$ = #distinct $X_j$
realized in the group. We may therefore study the underlying linear
functional $X_j$ instead of the advantage.

*Part 1.* For AN, $X_j = w^\top r_j$. Writing $r_j^{(\ell)} = v_0 + k_\ell^{(j)}
\Delta$ with $k_\ell^{(j)} \in \{0, \dots, L-1\}$ and $\Delta$ the grid step,
$$
X_j \;=\; w^\top v_0 \mathbf 1 \;+\; \Delta \sum_\ell w_\ell k_\ell^{(j)}.
$$
For integer weights the second sum takes values in $\{0, 1, \dots, (\sum w_\ell)(L-1)\}$,
giving at most $(\sum w_\ell)(L-1)+1$ distinct outcomes.

*Part 2, forward direction.* For NA-oracle, $X_j = \sum_\ell (w_\ell/\sigma_\ell)
r_j^{(\ell)} = \sum_\ell a_\ell r_j^{(\ell)}.$ Suppose $\{a_\ell\}$ are
$\mathbb{Q}$-linearly independent, and pick two distinct grid points $k =
(k_1, \dots, k_R) \ne k' = (k_1', \dots, k_R')$ in $\{0, \dots, L-1\}^R$. Then
$$
X(k) - X(k') = \Delta \sum_\ell a_\ell (k_\ell - k_\ell').
$$
This is nonzero: the integer vector $k - k' \ne 0$ would give a rational
linear relation among the $a_\ell$, contradicting independence. Hence the
map $k \mapsto X(k)$ is injective on $\{0, \dots, L-1\}^R$, so all $L^R$ grid
points realize distinct values.

*Part 2, necessary counterexample.* If all $\sigma_\ell$ are equal to a common
$\sigma$, then $a_\ell = w_\ell/\sigma$ and NA-oracle is exactly AN scaled by
$1/\sigma$ (since the within-group mean and std of $X$ also scale by $1/\sigma$).
By affine invariance the two advantage maps coincide and $N_{\mathrm{NA}} =
N_{\mathrm{AN}}$. ∎

**Verifying figure.** `figP2_resolution.png`. Enumeration over the full grid
gives, for $R = 2$, $w = (1, 1)$, $\sigma = (1, \sqrt 2)$:

| $L$ | AN | NA (equal σ) | NA (het. σ) | sum-lattice $R(L-1)+1$ | product $L^R$ |
|---|---|---|---|---|---|
| 2 | 3 | 3 | 4 | 3 | 4 |
| 3 | 5 | 5 | 9 | 5 | 9 |
| 4 | 7 | 7 | 16 | 7 | 16 |
| 5 | 9 | 9 | 25 | 9 | 25 |
| 6 | 11 | 11 | 36 | 11 | 36 |
| 7 | 13 | 13 | 49 | 13 | 49 |

NA-het hits $L^R$ exactly because $\sqrt 2$ is irrational. NA-equal-σ
collapses to AN, matching the necessary counterexample.

**Remark on RLVR practice.** Binary rewards ($L = 2$) give a resolution gain
of 4 vs 3 — modest in absolute terms but multiplicative in $R$. The
"resolution lives in the same heterogeneous-scale regime as Prop 1" caveat in
the problem-statement §3 is the substance of Part 2's necessary counterexample.

---

## Theorem 3 (Finite-sample MSE law, decoupled multi-reward)

Throughout this section we work under the random-score convention defined in
the Notation recap: $\{r_j\}_{j=1}^m$ are i.i.d. given $x$ with $\mathbb
E[r_j] = \mu$, $\mathrm{Cov}(r_j) = \Sigma$; $\{g_j\}$ are i.i.d. with
$\mathbb E[g_j] = 0$, $\mathrm{Var}(g_j) = \tau^2$, independent of $\{r_j\}$.
The gradient estimator is $\hat g = (1/m)\sum_j A_j g_j$.

**Statement (oracle NA).** With $A_j^{\mathrm{NA,oracle}} = w^\top z_j$,
$$
\boxed{\;\mathrm{MSE}\big(\hat g^{\mathrm{NA,oracle}}\big)
\;=\; \frac{\tau^2}{m}\, w^\top C\, w\;}
\qquad \text{(exactly, for all $m \ge 1$).}
$$

**Statement (self-normalized NA).** With $A_j^{\mathrm{NA}} = \sum_\ell
w_\ell (r_j^{(\ell)} - \bar r^{(\ell)})/(\hat\sigma_\ell + \delta)$ and the
biased sample standard deviation $\hat\sigma_\ell^2 = (1/m)\sum_j (r_j^{(\ell)}
- \bar r^{(\ell)})^2$, assume $r_j$ has finite fourth moments and each
$\sigma_\ell > 0$. Then for $\delta = o(\sigma_\ell)$,
$$
\mathrm{MSE}\big(\hat g^{\mathrm{NA}}\big)
\;=\; \frac{\tau^2}{m}\, w^\top \mathbb E[\hat C]\, w
\;=\; \frac{\tau^2}{m}\, w^\top C\, w \;+\; O(m^{-2}),
$$
where $\hat C$ is the within-group sample correlation matrix. The first
equality is **exact for all $m \ge 2$**; the second uses the standard
$\mathbb E[\hat C] = C + O(1/m)$ expansion for sample correlation under
finite fourth moments.

**Specialization to the harness convention.** With $\tau^2 = 1$ (the
harness's $g_j \sim N(0,1)$), both statements specialize to the familiar
form $\mathrm{MSE} = w^\top C w / m \;(+\, O(m^{-2}))$ that the figures
verify. The oracle figures (`fig1a/1c`) use a constant-score surrogate that
gives the same numerical value — see the Notation recap.

**Key consequence.** The leading $1/m$ coefficient is $w^\top C w$ (times
the score scale $\tau^2$). Positively correlated channels ($C_{kl} > 0$,
common in RLVR — accuracy and length tend to co-vary) push this coefficient
up; antagonistic channels push it down. Reward correlation thus sets the
achievable MSE *floor*, not the optimal group size.

**Proof of the oracle statement.** Direct second-moment computation.
$\mathbb E[\hat g^{\mathrm{NA,oracle}}] = 0$ since $\mathbb E[g_j] = 0$ and
$g_j \perp z_j$, so MSE = variance. Expand:
$$
\mathrm{Var}(\hat g^{\mathrm{NA,oracle}})
\;=\; \tfrac{1}{m^2}\!\sum_{j,k}\mathbb E\!\left[(w^\top z_j)(w^\top z_k)\, g_j g_k\right]
\;=\; \tfrac{1}{m^2}\!\sum_{j,k}\mathbb E\!\left[(w^\top z_j)(w^\top z_k)\right]\mathbb E[g_j g_k],
$$
using $g \perp z$. With $\{g_j\}$ i.i.d. mean-zero variance $\tau^2$,
$\mathbb E[g_j g_k] = \tau^2 \mathbf 1_{j=k}$, so only $j = k$ terms
survive:
$$
\mathrm{Var}(\hat g^{\mathrm{NA,oracle}})
\;=\; \tfrac{\tau^2}{m^2}\!\sum_{j=1}^{m}\mathbb E[(w^\top z_j)^2]
\;=\; \tfrac{\tau^2}{m^2}\cdot m \cdot w^\top C w
\;=\; \tfrac{\tau^2}{m}\, w^\top C\, w. \qquad\square
$$
*Exact* (no $O(m^{-2})$ remainder) and independent of the reward distribution
beyond second moments. The role of the oracle calculation is to pin the
leading constant unambiguously before any sample-statistic estimation enters.

**Proof of the self-normalized statement.** The argument is a clean
two-step: first an *exact* per-trial identity reducing $\mathrm{MSE}(\hat
g^{\mathrm{NA}})$ to the expected sample correlation matrix; then a classical
bias expansion for $\mathbb E[\hat C]$. The sample-correlation identity is
the key — it bypasses the Hoeffding decomposition entirely and shows why the
oracle leading constant *survives unchanged* under self-normalization.

For convenience, drop the $+\delta$ guard ($\delta = o(\sigma_\ell)$ adds
only $O(\delta)$ to each $\hat\sigma_\ell^{-1}$, contributing $O(\delta^2/m)$
to MSE — below the orders we track). Define the within-group standardized
residual
$$
u_{j,\ell} \;:=\; \frac{r_j^{(\ell)} - \bar r^{(\ell)}}{\hat\sigma_\ell},
\qquad j = 1,\dots,m,\ \ell = 1,\dots,R.
$$
Two identities hold *exactly* for $u$ by construction (no expectation, no
approximation), as a direct consequence of the definitions $\bar r^{(\ell)} =
(1/m)\sum_j r_j^{(\ell)}$ and $\hat\sigma_\ell^2 = (1/m)\sum_j (r_j^{(\ell)}
- \bar r^{(\ell)})^2$:
$$
\sum_{j=1}^m u_{j,\ell} \;=\; 0,
\qquad
\tfrac{1}{m}\!\sum_{j=1}^m u_{j,\ell}^2 \;=\; 1.
$$
And $A_j^{\mathrm{NA}} = w^\top u_j$.

*Step 1 — per-trial variance reduces to $w^\top \hat C w$.* Expand the MSE
using $g \perp r$ (the score is independent of the rewards by the score
convention) and $\mathbb E[g_j g_k] = \tau^2 \mathbf 1_{j=k}$:
$$
\mathrm{MSE}(\hat g^{\mathrm{NA}})
\;=\; \tfrac{1}{m^2}\!\sum_{j,k}\mathbb E\!\left[(w^\top u_j)(w^\top u_k)\right]\mathbb E[g_j g_k]
\;=\; \tfrac{\tau^2}{m^2}\!\sum_{j=1}^m \mathbb E[(w^\top u_j)^2]
\;=\; \tfrac{\tau^2}{m}\,\mathbb E[(w^\top u_1)^2],
$$
the last equality by exchangeability of $\{r_j\}$ (and hence $\{u_j\}$)
over $j$.

*Step 2 — $\mathbb E[(w^\top u_1)^2] = w^\top \mathbb E[\hat C]\, w$.* Write
$\mathbb E[(w^\top u_1)^2] = w^\top \mathbb E[u_1 u_1^\top]\, w$. We compute
$\mathbb E[u_1 u_1^\top]$ entrywise.

*Diagonal.* By exchangeability and the second exact identity above,
$$
\mathbb E[u_{1,\ell}^2]
\;=\; \mathbb E\!\left[\tfrac{1}{m}\!\sum_{j=1}^m u_{j,\ell}^2\right]
\;=\; \mathbb E[1] \;=\; 1
\quad \text{(deterministically).}
$$

*Off-diagonal* ($\ell \ne k$). By exchangeability and the sample-covariance
definition,
$$
\mathbb E[u_{1,\ell}\, u_{1,k}]
\;=\; \mathbb E\!\left[\tfrac{1}{m}\!\sum_{j=1}^m u_{j,\ell}\, u_{j,k}\right]
\;=\; \mathbb E\!\left[\frac{(1/m)\sum_j (r_j^{(\ell)} - \bar r^{(\ell)})(r_j^{(k)} - \bar r^{(k)})}{\hat\sigma_\ell\, \hat\sigma_k}\right]
\;=\; \mathbb E[\hat C_{\ell k}],
$$
where $\hat C$ is the within-group sample correlation matrix. So $\mathbb
E[u_1 u_1^\top] = \mathbb E[\hat C]$ (with the diagonal equal to $1$ both
because of the identity above and because $\hat C_{\ell\ell} = 1$ by
definition of the correlation matrix).

Combining Steps 1 and 2:
$$
\boxed{\;\mathrm{MSE}(\hat g^{\mathrm{NA}}) \;=\; \frac{\tau^2}{m}\, w^\top \mathbb E[\hat C]\, w
\quad \text{(exact for all $m \ge 2$).}\;}
$$
This is the first equality in the theorem statement.

*Step 3 — expand $\mathbb E[\hat C]$.* For i.i.d. samples with finite fourth
moments, the classical sample-correlation bias expansion (Fisher 1921;
modern derivation in any multivariate-statistics text — see e.g. Anderson,
*Introduction to Multivariate Statistical Analysis*, §4.2.1 ed. 3) gives,
for $\ell \ne k$,
$$
\mathbb E[\hat C_{\ell k}]
\;=\; C_{\ell k} \;-\; \frac{C_{\ell k}(1 - C_{\ell k}^2)}{2m} \;+\; O(m^{-2}),
$$
with the diagonal $\mathbb E[\hat C_{\ell\ell}] = 1$ exactly. Therefore
$\mathbb E[\hat C] = C + O(1/m)$ entrywise, and
$$
w^\top \mathbb E[\hat C]\, w \;=\; w^\top C\, w \;+\; O(1/m).
$$
Substituting back:
$$
\mathrm{MSE}(\hat g^{\mathrm{NA}})
\;=\; \tfrac{\tau^2}{m}\, w^\top C\, w \;+\; O(m^{-2}). \qquad\square
$$

**Remark (why $\mathbb E[\hat C]$ shows up exactly).** The crucial fact is
that *the within-group standardization $u_j$ is already correlation-shaped*:
its sample variance is identically $1$ per channel, and its sample
cross-product is identically $\hat C_{\ell k}$. The random score $g$
diagonalizes the double sum over $(j, k)$ to a single sum over $j$, and
exchangeability then turns a per-$j$ expectation into the sample-average
expectation. There is no Hoeffding decomposition, no delta-method expansion,
no Taylor remainder — the per-trial structure is *exact*. The only
approximation in the theorem is the classical sample-correlation bias
$\mathbb E[\hat C] = C + O(1/m)$, which is standard.

**Remark (the $O(m^{-2})$ coefficient).** Pinning the exact $O(m^{-2})$
constant amounts to using the next-order sample-correlation bias expansion,
which has a known closed form for the Gaussian case (cf. `figS3` and the
supplementary derivation below) and reduces to a one-page kurtosis-and-
joint-fourth-moment computation for general $r$ (flagged in §5).

**Verifying figures.**

- `fig1a_wCw_scaling.png` — oracle MSE = $w^\top C w / m$ (constant-score
  surrogate, $\tau^2 = 1$) verified across $m \in \{2, \dots, 128\}$ and
  $\rho \in \{-0.6, 0, 0.6\}$ to three significant figures. Matches the
  boxed oracle identity exactly.
- `fig1c_coeff_vs_rho.png` — coefficient $m \cdot \mathrm{MSE}$ traces
  $w^\top C w = 2 + 2\rho$ over $\rho \in [-0.9, 0.9]$, again under the
  oracle constant-score surrogate.
- `fig2a_bias.png` — single-reward illustration that the leave-one-out
  baseline is unbiased while the self/GRPO baseline carries the $(m-1)/m$
  factor. (This is the U-statistic infrastructure underlying the multi-reward
  story, not Thm 3 itself.) At $m = 4$: sim $0.1876$ vs theory $(3/4)\theta
  = 0.1875$.
- `fig2b_variance.png` — single-reward U-statistic variance follows
  $a/m + b/m^2$ (weighted fit: $a = 0.058$, $b = 0.298$); pure $a/m$ is
  rejected. This is the two-term Hoeffding signature of a second-order
  U-statistic and underpins the $O(m^{-2})$ remainder in Thm 3 self-normalized.
- `figS1_selfnorm_mse.png` / `figS2_selfnorm_vs_rho.png` — direct check of
  the boxed identity $m \cdot \mathrm{MSE}(\hat g^{\mathrm{NA}}) = \mathbb
  E[w^\top \hat C w]\cdot\tau^2$ under the random-score convention
  ($g_j \sim N(0,1)$, $\tau^2 = 1$). The harness reports $m \cdot \mathrm{MSE}_{\mathrm{self}} =
  \mathbb E[w^\top \hat C w]$ to three digits across $\rho \in \{-0.6, 0, 0.6\}$
  at $m = 32$ — e.g., at $\rho = 0.6$: $m \cdot \mathrm{MSE} = 3.1849$,
  $\mathbb E[w^\top \hat C w] = 3.1864$, $w^\top C w = 3.2000$ — the small
  gap to $w^\top C w$ is the $O(1/m)$ sample-correlation bias of Step 3.

**Negative result (correlation does NOT govern $m^\star$).** An earlier
conjecture that $m^\star \propto \sqrt{w^\top C w}$ is falsified by
`figM1_mstar_vs_rho.png`: $m^\star$ varies by at most one grid step across
$\rho \in [-0.6, 0.9]$, vs a predicted 2.18× spread. The $w^\top C w$ term
sits in the part of the MSE that is constant in $m$ (the leading $1/m$
coefficient at fixed $N = Pm$ contributes $w^\top C w / P = w^\top C w \cdot
m/N$, which is linear in $m$ at fixed budget — it raises the noise floor
uniformly without shifting the bias-variance crossover that determines
$m^\star$). The budget-driven growth $m^\star \propto N^{1/3}$ is confirmed
by `fig2c_groupsize_law.png` (six budgets; fitted exponent $0.318 \pm 0.010$,
inside the 95% CI for $1/3$).

---

## Supplementary: self-normalized bias coefficient (figS3)

For Gaussian rewards the leading $1/m$ coefficient of $\mathbb E[\hat\sigma_\ell]
- \sigma_\ell$ is the exact $\Gamma$-formula expansion:
$$
\mathbb E[\hat\sigma] = \sigma \sqrt{\tfrac{2}{m-1}} \cdot \frac{\Gamma(m/2)}{\Gamma((m-1)/2)}
= \sigma \left(1 - \tfrac{3}{4m} + O(m^{-2})\right),
$$
so $m \cdot \mathbb E[\hat\sigma - \sigma] \to -3\sigma/4$ exactly. The figure
overlays this at $-3\theta/4 = -0.75$ for the Gaussian curve, where $\theta =
\mathrm{Cov}(r, g)/\sigma = 1$ in the harness.

For Bernoulli($p$) rewards the leading coefficient is *not* $-3/4$. Direct
calculation: the biased sample variance has $\mathbb E[\hat\sigma^2] =
\tfrac{m-1}{m} \sigma^2$, and a one-term Taylor expansion gives
$$
\mathbb E[\hat\sigma] \approx \sigma\sqrt{(m-1)/m}\big(1 - \tfrac{1}{8(m-1)}\mathrm{Var}(\hat\sigma^2/\sigma^2) + \cdots\big).
$$
For $\mathrm{Bern}(\tfrac12)$, the empirical leading coefficient is $-\theta/2 =
-0.25$ (figS3). The general formula factors through the kurtosis of $r^{(\ell)}$
and joint moments with $g$; closing the form for non-Gaussian discrete rewards
is a one-page open derivation flagged in problem-statement §5.

---

## Proposition 4 (Synthetic-model bias law)

**Model.** Fix a single prompt. Let $r^{(a)} \sim \mathrm{Bern}(p_a)$ (the
"harder" gate, e.g. correctness) and $r^{(b)} \sim \mathrm{Bern}(p_b)$ (the
"easier" conditioned channel, e.g. format/length), with $r^{(a)} \perp r^{(b)}$.
Let the policy-gradient direction $g$ follow the contaminated decomposition
$$
g \;=\; \alpha_c (r^{(a)} - p_a) \;+\; \alpha_f (r^{(b)} - p_b)\, r^{(a)}
\;+\; \gamma (r^{(b)} - p_b)(1 - r^{(a)}) \;+\; \varepsilon,
$$
with $\varepsilon$ mean-zero noise independent of $(r^{(a)}, r^{(b)})$. The
parameter $\gamma$ controls *contamination*: how much the easier reward
$r^{(b)}$ drives the gradient on incorrect rollouts. The target estimand is
$\theta^\star := w_a\,\mathrm{Cov}(r^{(a)}, g) + w_b\,\mathrm{Cov}(r^{(b)} r^{(a)}, g)$
("format counts only when correct").

**Statement.** Under this synthetic model, the cross-objective bias removed
by conditioning is
$$
\beta_{ab} \;:=\; w_b\big[\,\mathrm{Cov}(r^{(b)}, g) \;-\; \mathrm{Cov}(r^{(b)} r^{(a)}, g)\,\big]
\;=\; w_b\, p_b(1 - p_a)\big[\gamma(1 - p_b) - \alpha_c\, p_a\big].
$$
In particular $\beta_{ab} = 0$ on the indifference curve $\gamma^\star =
\alpha_c\, p_a / (1 - p_b)$, and changes sign across it: conditioning *helps*
iff contamination dominates gate-coupling,
$\gamma > \alpha_c\, p_a / (1 - p_b)$.

**Proof.** Compute the two covariances under the model term by term, using
$r^{(a)} \perp r^{(b)}$, $\mathbb E[r^{(a)}] = p_a$, $\mathbb E[r^{(b)}] = p_b$,
$\mathbb E[r^{(a)}(r^{(a)} - p_a)] = \mathbb E[(r^{(a)})^2] - p_a^2 = p_a -
p_a^2 = p_a(1-p_a)$, the symmetric identity for $r^{(b)}$, and
$(r^{(a)})^2 = r^{(a)}$ (Bernoulli).

*Compute $\mathrm{Cov}(r^{(b)}, g)$.* Decompose by the three reward terms of
$g$ (the noise $\varepsilon$ has zero covariance with anything):

- Term 1: $\alpha_c \mathrm{Cov}(r^{(b)}, r^{(a)} - p_a) = 0$ by independence.
- Term 2: $\alpha_f \mathrm{Cov}(r^{(b)}, (r^{(b)} - p_b) r^{(a)})
  = \alpha_f\big[\mathbb E[r^{(b)}(r^{(b)} - p_b)]\,\mathbb E[r^{(a)}]\big]
  = \alpha_f\, p_a\, p_b(1 - p_b)$,
  using independence to factor and $\mathbb E[r^{(b)}]\mathbb E[(r^{(b)} - p_b) r^{(a)}] = p_b \cdot 0 = 0$.
- Term 3: $\gamma \mathrm{Cov}(r^{(b)}, (r^{(b)} - p_b)(1 - r^{(a)}))
  = \gamma\, p_b(1 - p_b)(1 - p_a)$, by the analogous factoring.

Sum:
$$
\mathrm{Cov}(r^{(b)}, g)
\;=\; \alpha_f\, p_a\, p_b(1 - p_b) \;+\; \gamma\, p_b(1 - p_b)(1 - p_a).
$$

*Compute $\mathrm{Cov}(r^{(b)} r^{(a)}, g)$.*

- Term 1: $\alpha_c \mathbb E[r^{(b)} r^{(a)} (r^{(a)} - p_a)]
  = \alpha_c\, p_b\, p_a(1 - p_a)$, using independence and
  $\mathbb E[r^{(a)}(r^{(a)} - p_a)] = p_a(1-p_a)$. The mean-product correction
  is $\mathbb E[r^{(b)} r^{(a)}]\mathbb E[r^{(a)} - p_a] = p_a p_b \cdot 0 = 0$.
- Term 2: $\alpha_f \mathbb E[r^{(b)} r^{(a)} (r^{(b)} - p_b) r^{(a)}]
  = \alpha_f \mathbb E[(r^{(a)})^2] \mathbb E[r^{(b)}(r^{(b)} - p_b)]
  = \alpha_f\, p_a\, p_b(1 - p_b)$.
- Term 3: $\gamma \mathbb E[r^{(b)} r^{(a)} (r^{(b)} - p_b)(1 - r^{(a)})] = 0$,
  because $r^{(a)}(1 - r^{(a)}) = 0$ pointwise (Bernoulli).

Sum:
$$
\mathrm{Cov}(r^{(b)} r^{(a)}, g)
\;=\; \alpha_c\, p_a p_b(1 - p_a) \;+\; \alpha_f\, p_a\, p_b(1 - p_b).
$$

*Subtract.* The $\alpha_f$ terms appear with the *same* coefficient
$p_a\, p_b(1 - p_b)$ in both covariances, so they cancel:
$$
\beta_{ab}
\;=\; w_b\big[\mathrm{Cov}(r^{(b)}, g) \;-\; \mathrm{Cov}(r^{(b)} r^{(a)}, g)\big]
\;=\; w_b\big[\gamma\, p_b(1 - p_b)(1 - p_a) \;-\; \alpha_c\, p_a p_b(1 - p_a)\big].
$$
Factoring $w_b\, p_b(1 - p_a)$:
$$
\beta_{ab} \;=\; w_b\, p_b(1 - p_a)\big[\gamma(1 - p_b) - \alpha_c\, p_a\big]. \qquad \square
$$

**Remark (why the $\alpha_f$ piece cancels).** The model's second term,
$\alpha_f(r^{(b)} - p_b) r^{(a)}$, is the *target* part of the gradient — the
piece that the conditioned estimand $\theta^\star$ correctly captures. It
contributes equally to both $\mathrm{Cov}(r^{(b)}, g)$ and $\mathrm{Cov}(r^{(b)}
r^{(a)}, g)$, so it drops out of the difference. Only the **contamination
term** $\gamma$ (which conditioning is designed to remove) and the
**gate-coupling term** $\alpha_c$ (which conditioning over-removes by also
clipping out the legitimate $r^{(a)}$ signal on the $r^{(b)}$ side) survive.
The sign of $\beta_{ab}$ is the contest between these two.

**Verifying figure.** `figT2_bias_law.png`. At $\alpha_c = 1, p_a = 0.5, p_b
= 0.6$ the sign change occurs at $\gamma^\star = 0.5/0.4 = 1.25$, which is
exactly where the leave-one-out sim crosses zero (residual: $-5.7 \times
10^{-3}$ at $\gamma = 1.25$, well within sampling noise).

**Scope and what is *not* claimed.** Proposition 4 is stated and proven
**under the synthetic generative model above.** The model is exactly what
`figT2_bias_law.png` simulates and what `figT3_phase_diagram.png` sweeps;
the closed form $\beta_{ab} = w_b p_b (1 - p_a)[\gamma(1 - p_b) - \alpha_c
p_a]$ is exact in this model. Lifting the proposition to a *general*
data-dependent gate (where the policy structure determines $g$ in a way that
need not factor through $\gamma$) is intentionally **deferred to future
work** (problem-statement §5). The constructive resolution that makes this
deferral defensible: the **zero-fill** implementation of conditioning
(centering $r^{(b)} r^{(a)}$ over all $m$ rollouts, not just the passing
subgroup) preserves the U-statistic structure that Theorem 3 relies on, and
so the Thm 3 MSE law applies directly to the conditioned reward channel
without any new gate-specific analysis. Proposition 4′ formalizes this as a
practical recommendation.

---

## What's proven, what's sketched, what's open

| Result | Status |
|---|---|
| Prop 1 (influence law) | **Proven** — oracle exact; self-normalized $O(m^{-1})$ correction via delta method. |
| Prop 2 (resolution bound) | **Proven** — affine invariance + sum-lattice / $\mathbb Q$-independence argument; tight on both bounds. |
| Thm 3 oracle MSE | **Proven** — exact, one-line direct calculation. |
| Thm 3 self-normalized MSE | **Proven** — exact identity $\mathrm{MSE} = (\tau^2/m)\, w^\top \mathbb E[\hat C] w$ for all $m\ge 2$ (via the sample-correlation identity $u_{j,\ell}\, u_{j,k}$ averaging to $\hat C_{\ell k}$), giving asymptotic $\mathrm{MSE} = (\tau^2/m)\, w^\top C w + O(m^{-2})$ under finite fourth-moment conditions. The closed form of the $O(m^{-2})$ coefficient for non-Gaussian rewards is left as a supplementary computation. |
| Prop 4 (synthetic-model bias law) | **Proven** — direct covariance computation under the explicit generative model used for validation. |
| Prop 4′ (zero-fill recommendation) | **Operational** — empirical recommendation backed by `figT1`/`figT3` and by the structural argument that zero-fill preserves the U-statistic kernel and so inherits Thm 3. Not lifted to a theorem. |
| Prop 4 general bias under arbitrary data-dependent gate | **Future work** — explicitly out of scope; zero-fill + Thm 3 is the constructive workaround the paper recommends. |
| $m^\star \propto \sqrt{w^\top C w}$ (first-draft) | **Falsified** — `figM1`. $w^\top C w$ governs the MSE floor, not $m^\star$. |

**Net.** 4 fully proven results + 1 empirical recommendation + 1 explicitly
bounded future-work item + 1 falsified first-draft conjecture. No claim in the
paper rests on a proof gap; every claim is either fully proven, marked
operational, or marked future work.
