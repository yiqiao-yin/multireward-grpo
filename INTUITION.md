# INTUITION.md — the results, explained without the math

This file is the plain-language companion to the paper *"When and Why
Decoupling and Conditioning Beat Reweighting in Multi-Reward GRPO: A U-Statistic
Treatment."* Every result gets an everyday analogy. No proofs, no notation you
have to look up.

It is written for three audiences: someone who wants the ideas before the
algebra, someone explaining this work to a non-technical colleague, and the
future maintainer who needs to remember *why* a result mattered.

The formal statements live in `proposed-solutions.md`, the proofs in
`proofs.md`, and both appear in the manuscript. **This file is intuition only.
Where it and the paper disagree, the paper is right.**

---

## The setup, in one paragraph

We are training a language model with reinforcement learning. For each question,
the model produces several attempts — say eight. Each attempt is scored on
several things at once: *was the answer correct?*, *was the length reasonable?*,
*was the format right?* Those separate scores get blended into a single number
that tells the model "do more of this" or "do less of that." That blended number
is the **steering signal**.

Everything below is about one question: **how should the separate scores be
blended, and what does the blending cost you?**

There are two obvious orderings, and the whole paper hangs off the difference:

- **AN (Aggregate-then-Normalize)** — add the raw scores up first, then compare
  attempts against each other. This is what standard GRPO does.
- **NA (Normalize-then-Aggregate)** — put each score on a common scale first,
  *then* add them up. This is the "decoupled" method proposed by MO-GRPO and
  GDPO, and it is the object this paper analyzes.

---

## Proposition 1 — the influence law: *why raw addition breaks*

> **Formally:** under AN, a channel's influence on the advantage scales with its
> standard deviation σ. Under NA, the σ cancels.
>
> **Status:** background. This result is MO-GRPO's, restated here with general
> weights. We do not claim it.

### The analogy: adding money in different currencies

You have three sources of value and you want them to count equally. One is
denominated in Japanese yen, one in US dollars, one in British pounds.

If you just **add the numbers** — 5000 + 40 + 30 — the yen dominates completely.
Not because yen are worth more, but because yen *come in bigger numbers*. You
declared the three equally important and the arithmetic ignored you.

That is AN. Reward channels have different natural spreads: a binary
correctness flag jumps between 0 and 1, while a length score might drift within
a narrow band. Add them raw and the jumpiest channel runs the show, no matter
what importance you assigned.

NA converts everything to a common currency first — expressing each score as
"how unusual is this, for this channel?" — and only then applies your weights.
Now equal weights genuinely mean equal say.

### The part people get wrong

The tempting fix is *"just turn down the weight on the loud channel."* You can
do that in principle — but you'd need to know the exchange rate, the exchange
rate depends on the model, and it **drifts while you train**. As the model gets
better at correctness, that channel's spread changes and your carefully tuned
weights are wrong again.

So the honest claim is not "reweighting is impossible." It is: *reweighting
cannot fix this in a stable, set-it-once way.* Converting to a common scale can.

---

## Proposition 2 — the resolution bound: *why totals lose information*

> **Formally:** AN can realize at most `R(L−1)+1` distinct advantage values (a
> *sum* lattice); NA can realize up to `L^R` (a *product* lattice), provided the
> channel scales don't collide.
>
> **Status:** ours — it formalizes GDPO's qualitative "reward signal collapse."

### The analogy: grading a test on the total only

Two students each score 7 out of 10.

- Student A aced the math section and bombed the essay.
- Student B bombed the math and aced the essay.

Report only the total, and they are **identical**. You have thrown away the one
thing that distinguishes them — and if you were trying to teach them, you'd give
both the same feedback despite them needing opposite advice.

That is AN. "Correct but too short" and "wrong but nicely formatted" can add up
to the same total, so the model gets the same nudge for opposite behaviors.
Keeping the sub-scores separate keeps them distinguishable.

### The catch, and the correction peer review forced

Separating only helps if the sub-scores are on **genuinely different scales**.
If math and essay are both out of 5 and weighted equally, the total collapses
them just as badly. Different scales are what make the combinations land on
distinct values.

The original paper claimed a clean "if and only if" here, and **a reviewer
caught that it was wrong.** The corrected version is nicer, and there's a coin
analogy for it:

Suppose you have coins worth **1** and **3**, and you may use **at most one of
each**. The reachable totals are 0, 1, 3, 4 — all distinct, no collisions. But
if you were allowed **three** 1-coins, then 1+1+1 would collide with a single 3.

Whether collisions happen depends on *how many of each coin you're allowed* —
not just on the coin values. With only a few score levels per channel, you can't
build the big combinations that would cause a collision, so scales that look
like they should collide often don't. That bounded-count condition is the exact
answer; the original "irrational ratio" condition was merely *sufficient*.

---

## Theorem 3 — the correlation floor: *the headline result*

> **Formally:** `MSE = (τ²/m)·wᵀCw + O(m⁻²)`, where `m` is the number of
> attempts and `C` is the correlation matrix between reward channels.
>
> **Status:** ours, and new.

### The analogy: a team hauling a cart on ropes

Picture a heavy cart being pulled by a team. Each rope is one of your
objectives. Everyone pulls steadily *on average*, but every grip wobbles.

Two things decide how much the cart lurches:

**1. How many people are pulling.** More haulers, smoother ride. That's the
`1/m` — eight attempts instead of four halves the jitter. Straightforward, and
you pay for it in compute.

**2. Whether the wobbles are synchronized.** This is the surprising one. If
everyone's grip slips at the same instant, the cart lurches hard. If one
person's wobble pulls left exactly when another's pulls right, they cancel and
the cart glides. *Same crew, same average force, completely different ride.*

That synchronization is the correlation matrix `C`. The quantity `wᵀCw` is just
the arithmetic of how much the wobbles reinforce one another.

### The counterintuitive part

Most people assume objectives that **agree** should be easier to train. It is
the opposite: **objectives that move together cost you more.**

Agreement carries no new information. If "correct" and "well-formatted" always
rise and fall together, the second tells you nothing the first didn't — yet you
still pay full price for its noise. You've added a rope without adding pulling
power, while doubling the odds of a synchronized lurch.

Conversely, objectives that **fight** each other are cheap. On GSM8K we measure
correctness and length at −0.64: easy problems get short right answers, hard
ones get long wrong ones. Those wobbles genuinely cancel, and the steering comes
out smoother than if the objectives were unrelated.

### For a finance-literate audience

Skip the cart. `wᵀCw` is **literally the portfolio variance formula** — not an
analogy, the identical equation. Reward channels are assets, `w` is your
allocation, `C` is the correlation matrix. It's Markowitz.

The lesson transfers exactly: **a portfolio of five tech stocks is not
diversified.** Correlated holdings don't reduce volatility no matter how many
you hold. Bolt on five reward channels that all measure roughly "is this answer
good," and you've built a concentrated position, not a diversified one.

### Why this is useful before you spend anything

The correlation `C` is a property of the *scores*, not of the gradients. So you
can estimate it from a batch of sample outputs from the untrained model — pure
generation, no training — and compare candidate reward designs in advance.

You get **relative** cost, not absolute: if design A yields `wᵀCw = 3.0` and
design B yields `1.5`, then B needs about half as many attempts for equally
steady steering. You cannot get "40 GPU-hours" out of this, because the formula
also contains a term you can't know without differentiating the model — but that
term is the same for every design you're comparing, so it cancels in the ratio.

---

## Proposition 4 — the conditioning law: *when "only count X if Y" helps*

> **Formally:** the bias removed by gating is
> `β = w_b·p_b(1−p_a)·[γ(1−p_b) − α_c·p_a]`, which **changes sign** at a
> threshold `γ★`.
>
> **Status:** ours, and new.

A natural instinct with multiple objectives is to **gate** one behind another:
*"only count politeness if the answer was actually compliant."* The finding is
that this is **not** an always-good move. It helps in one regime and hurts in
the other.

### The analogy: tipping a waiter for friendliness

You want your restaurant's waiters to be accurate *and* friendly. You decide to
tip for friendliness **only when the order arrives correct**.

**When this is a good rule.** If waiters have learned that being charming lets
them get away with wrong orders, then friendliness is actively *masking*
failure. Gating it kills that strategy — charm no longer pays when the order is
wrong. You've removed a real problem.

**When this is a bad rule.** If waiters are simply friendly by disposition, and
wrong orders come from the kitchen rather than from them, gating punishes
genuinely good behavior for a failure that wasn't theirs. You've thrown away
real signal and taught them nothing.

The threshold `γ★` is the exact tipping point between those two worlds. Below
it, gating hurts. Above it, gating helps. Which is why *"only count X if Y"*
should be a measured decision, not a reflex.

---

## Proposition 4′ — how to implement gating: *zero-fill, not subgroups*

Two ways to actually build the gate, and they behave very differently.

### The analogy: ranking a class

You want to rank students on essay quality, but only among those who passed the
prerequisite exam.

**The subgroup approach** ranks only the students who passed. The problem: if
only *one* student passed, there is no ranking — a class of one has no spread.
If *none* passed, there's nothing at all. And this happens constantly: in our
own GSM8K runs, at four attempts per question, **72.8% of groups had at least
one score that was identical across every attempt.** The subgroup is empty or
degenerate far more often than you'd guess.

**The zero-fill approach** gives a zero to everyone who failed the prerequisite
and ranks the whole class. You always have a full class to rank against, the
comparison never collapses, and — the technical payoff — the statistical
structure is preserved, so Theorem 3 still applies to the gated channel with no
extra work.

**Recommendation: zero-fill.** It avoids the degeneracy and inherits the theory
for free.

---

## Two claims we tested and killed

The paper's harness falsified two of our own first-draft conjectures before
submission. We kept the record rather than quietly deleting them, because a
plausible-but-wrong claim is worth knowing about.

### "Correlated rewards mean you need bigger groups" — false

It sounds right: if the signal is noisier, take more attempts. It isn't right.
Correlation sets **how bumpy the ride is**, not **how many people you hire**.
Those turn out to be independent decisions.

What *does* set the group size is your total budget. If you can afford a fixed
number of attempts overall, you're choosing between *many questions tried a few
times each* and *few questions tried many times each* — and the optimal group
size grows slowly (roughly the cube root) with total budget, regardless of
reward correlation.

### "The self-normalization bias has a universal coefficient" — false

Estimating the spread from a small group introduces a small bias. We assumed the
size of that bias was universal. It isn't — it depends on the *shape* of the
reward distribution. Bell-curve-shaped rewards and coin-flip-shaped rewards give
different answers. There is still no general formula, and we say so.

---

## What the theory does *not* say

Worth stating plainly, because these are the natural over-readings:

**It's about steadiness, not direction.** Theorem 3 tells you how much the
steering wheel shakes, not whether you're steering somewhere worthwhile. A
perfectly smooth signal can point in a useless direction. Low `wᵀCw` means a
low-noise estimator, *not* fast learning or a better model.

**Correlation is measured, not fixed.** `C` is a snapshot of the model you
measured it on. As training changes the model's behavior, `C` drifts. Treat it
as a planning estimate, most trustworthy early.

**Decoupling isn't a free win on every metric.** In our fine-tuning experiment,
NA and AN reach *statistically indistinguishable* total reward. What differs —
and differs significantly — is **how the model allocates effort across
objectives**. NA honors the trade-off you specified; AN quietly over-serves
whichever channel happens to be jumpiest. That's the claim, and it's narrower
than "NA wins."

---

## Where to go next

| You want | Read |
|---|---|
| Formal statements with intuition | `proposed-solutions.md` |
| Full proofs | `proofs.md` |
| Notation and the four questions | `problem-statement.md` |
| What was verified, and what was falsified | `empirical-section.md`, and the verification table in `README.md` |
| Related work and positioning | `literature-review.md` |

> **Note:** this file is mirrored in two repositories — the public code
> repository and the private manuscript repository. Edit one, copy to the other.
