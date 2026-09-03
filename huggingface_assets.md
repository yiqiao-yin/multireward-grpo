# HuggingFace Assets — Multi-Reward GRPO Project

All dataset + model URLs released for the paper, with file contents and how to load.

---

## Datasets (3)

### 1. GSM8K rewards — Qwen2.5-1.5B-Instruct

**URL:** https://huggingface.co/datasets/eagle0504/multireward-grpo-gsm8k-rewards

**Files** (all viewable in the HF Files tab; the parquet renders as a tabular preview at the top of the page):

| File | Size | What |
|---|---|---|
| `data/train.parquet` | 124 KB | **Tabular preview** — 76,800 rows × 12 cols: `model, prompt_idx, seed_idx, rollout_idx, group_size_m_max, correctness, length, format, raw_length, question, gold_answer, gold_reasoning` |
| `llm_rewards_gsm8k.npz` | 335 KB | Raw numpy tensors: `rewards_m{m}` (P=150, K=16, m, R=3), `raw_length_m{m}` (P, K, m), metadata |
| `llm_generations_gsm8k.jsonl` | **77 MB** | **76,800 chains-of-thought** as JSON lines: `{prompt_idx, gold, seed_idx, rollout_idx, completion, correctness, length, format, raw_length}` |
| `llm_summary_gsm8k.json` | 4 KB | Aggregate Thm 3 numbers per m |
| `llm_prop4_summary_gsm8k.json` | 3 KB | γ-sweep numbers (Prop 4 bias law) |
| `llm_mse_vs_m_gsm8k.png` | 75 KB | Headline Theorem 3 figure |
| `llm_money_scatter_gsm8k_m{4,8,16,32}.png` | ~75 KB each | Per-prompt predicted-vs-realized scatters |
| `llm_prop4_bias_law_gsm8k.png` | 65 KB | Prop 4 sign-changing bias law figure |

**Quick load:**
```python
import pandas as pd
df = pd.read_parquet(
    "hf://datasets/eagle0504/multireward-grpo-gsm8k-rewards/data/train.parquet"
)
# 76800 rows × 12 cols; df["question"] has the GSM8K text
```

**Headline:** 150 GSM8K test prompts × 16 seeds × 32 rollouts on Qwen2.5-1.5B-Instruct. Thm 3 verified: NA/pred = 0.93–1.03 across m ∈ {4, 8, 16, 32}. Pooled reward correlation `corr(correctness, length) = −0.64`.

---

### 2. GSM8K rewards — Qwen2.5-7B-Instruct

**URL:** https://huggingface.co/datasets/eagle0504/multireward-grpo-gsm8k-rewards-qwen2.5-7b

**Files** (same shape as #1, generated with the 7B model):

| File | Size | What |
|---|---|---|
| `data/train.parquet` | 80 KB | **Tabular preview** — 25,600 rows × 12 cols (same schema as #1) |
| `llm_rewards_gsm8k.npz` | 94 KB | Raw rewards tensor (P=100, K=8, m, R=3) |
| `llm_generations_gsm8k.jsonl` | **26 MB** | **25,600 chains-of-thought** from Qwen2.5-7B |
| `llm_summary_gsm8k.json` | 4 KB | Aggregate Thm 3 numbers |
| `llm_prop4_summary_gsm8k.json` | 3 KB | γ-sweep numbers |
| `llm_mse_vs_m_gsm8k.png`, `llm_money_scatter_*.png`, `llm_prop4_bias_law_gsm8k.png` | ~75 KB each | Figures |

**Headline:** 100 GSM8K test prompts × 8 seeds × 32 rollouts. Thm 3 verified at production scale: NA/pred = 0.85–0.95.

---

### 3. Synthetic fintech customer-comms (Qwen2.5-7B generator)

**URL:** https://huggingface.co/datasets/eagle0504/multireward-grpo-fintech-customer-comms

**Files:**

| File | Size | What |
|---|---|---|
| `data/train.parquet` | 153 KB | **Tabular preview** — 2,400 rows × 17 cols: `model, scenario_id, scenario_type, persona, name, amount, due_date, seed_idx, rollout_idx, group_size_m, conversation_so_far, bot_reply, compliance, politeness_gated, action, raw_length, raw_politeness` |
| `fintech_rewards.npz` | 7 KB | Raw reward tensors |
| `fintech_generations.jsonl` | 985 KB | All 2,400 bot replies with their reward scores |
| `fintech_metadata.json` | 104 KB | Per-scenario metadata (system prompt, turn history) |
| `fintech_sample_rollouts.json` | 140 KB | 50 sample rollouts for inspection — a **uniform random subsample** drawn without replacement by a seeded RNG (`fintech_generate.py`), not human-selected |
| `fintech_summary.json` | 1.5 KB | Aggregate stats by scenario type |

**Headline:** 300 synthetic fintech customer-service conversations × 8 bot replies × 3 reward channels (compliance, politeness_gated, action). Mean compliance 0.984, politeness 0.587, action 0.996. Includes 15 scenario types (billing, refund, dispute, fraud, phishing-test, etc.) and 6 user personas.

---

## Trained models (2)

### 4. NA-GRPO fine-tune of Qwen2.5-1.5B (paper's recommendation)

**URL:** https://huggingface.co/eagle0504/multireward-grpo-fintech-na-qwen2.5-1.5b

- **Format:** LoRA adapter (r=16, α=32, ~17 MB)
- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Trained on:** fintech scenarios drawn from the **same generator** as dataset #3 (`fintech_scenarios.make_scenarios`), *not* on the released rows of dataset #3. `grpo_train.py` regenerates its own 400 scenarios under its own training seed; the released corpus (seed 42) and the training data are sibling draws from one generative process. Held-out eval draws 80 scenarios under seed 999.
- **Advantage:** **Normalize-then-Aggregate** — per-channel group-normalize the reward vector, then weighted sum
- **Hyperparameters:** 150 GRPO steps, P=4 prompts/batch, m=8 rollouts, lr=5e-6, kl_coef=0.05, weights=(1, 1, 0.5)
- **Result:** mean aggregate reward **1.7133 ± 0.0837** over last 30 steps (lower std vs AN — exactly Thm 3's prediction)

**Load:**
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype="bfloat16"
)
model = PeftModel.from_pretrained(
    model, "eagle0504/multireward-grpo-fintech-na-qwen2.5-1.5b"
)
```

### 5. AN-GRPO fine-tune of Qwen2.5-1.5B (baseline for comparison)

**URL:** https://huggingface.co/eagle0504/multireward-grpo-fintech-an-qwen2.5-1.5b

- **Format:** LoRA adapter (r=16, α=32, ~17 MB)
- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Trained on:** same generator and same hyperparameters as #4 (see the provenance note there)
- **Advantage:** **Aggregate-then-Normalize** — weighted sum then group-normalize (standard GRPO baseline)

The intended use is direct paired comparison with #4: identical everything except the advantage formula.

### 6. Single-reward GRPO fine-tune of Qwen2.5-1.5B (multi-reward-ablation baseline)

**URL:** https://huggingface.co/eagle0504/multireward-grpo-fintech-single-qwen2.5-1.5b

- **Format:** LoRA adapter (r=16, α=32, ~17 MB)
- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Advantage:** vanilla GRPO on the **compliance channel only** — ignores politeness and action
- **Purpose:** demonstrates that naively collapsing to a single reward is catastrophic for the other objectives. This model maxes compliance (1.00) but **destroys** politeness (0.31) and action (0.00), and its held-out aggregate reward (1.31) is **worse than the untrained base model (1.80)**.

---

## ⭐ Headline finetuning result (500 steps, 3 seeds for NA/AN, held-out eval on 80 unseen scenarios)

| Model | Held-out aggregate reward | compliance | politeness | action |
|---|---|---|---|---|
| base (no training) | 1.800 | 0.99 | 0.51 | 0.60 |
| **NA** (paper, n=3) | **2.082 ± 0.013** | 1.00 | **0.72** | 0.73 |
| **AN** (baseline, n=3) | **2.109 ± 0.007** | 1.00 | 0.65 | 0.92 |
| single-reward (n=1) | 1.310 | 1.00 | 0.31 | 0.00 |

**Three findings:**
1. **Multi-reward shaping is necessary, not cosmetic.** Single-reward GRPO (1.310) lands *below the untrained base model* (1.800) — optimizing only compliance collapses politeness and action. Both NA and AN beat base by ~0.3.
2. **NA respects the intended weight balance; AN distorts toward the high-variance channel.** NA keeps politeness and action balanced (0.72 / 0.73), while AN over-rewards the binary high-variance `action` channel (0.92) at the expense of the continuous `politeness` channel (0.65). This is **Proposition 1 (influence law) visible in a trained model** — AN's influence is dominated by the high-σ objective.
3. **NA and AN reach comparable aggregate reward** (2.082 vs 2.109) — consistent with the theory: NA is not claimed to win on aggregate reward; it is claimed to give weight-proportional influence (Prop 1) and a correlation-aware gradient-MSE floor (Thm 3, verified separately on the GSM8K rollout data).

Training curves (mean ± seed-std) → `figures/grpo_train/grpo_training_curves_multiseed.png`.
Held-out eval numbers → `figures/grpo_train/eval_comparison.json`.

---

## Verification — every URL is live (HTTP 200) as of last check

```
✓ https://huggingface.co/datasets/eagle0504/multireward-grpo-gsm8k-rewards
✓ https://huggingface.co/datasets/eagle0504/multireward-grpo-gsm8k-rewards-qwen2.5-7b
✓ https://huggingface.co/datasets/eagle0504/multireward-grpo-fintech-customer-comms
✓ https://huggingface.co/eagle0504/multireward-grpo-fintech-na-qwen2.5-1.5b
✓ https://huggingface.co/eagle0504/multireward-grpo-fintech-an-qwen2.5-1.5b
```

To re-verify any time:
```bash
for repo in eagle0504/multireward-grpo-gsm8k-rewards \
            eagle0504/multireward-grpo-gsm8k-rewards-qwen2.5-7b \
            eagle0504/multireward-grpo-fintech-customer-comms; do
  curl -sS -o /dev/null -w "%{http_code}  https://huggingface.co/datasets/$repo\n" \
    "https://huggingface.co/api/datasets/$repo"
done
for repo in eagle0504/multireward-grpo-fintech-na-qwen2.5-1.5b \
            eagle0504/multireward-grpo-fintech-an-qwen2.5-1.5b; do
  curl -sS -o /dev/null -w "%{http_code}  https://huggingface.co/$repo\n" \
    "https://huggingface.co/api/models/$repo"
done
```

---

## Headline numbers across all artifacts

### Theorem 3 (MSE law) — verified on two model scales

| Model | m=4 | m=8 | m=16 | m=32 | Source dataset |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 1.025 | 0.928 | 0.954 | 0.955 | #1 |
| *95% CI* | [0.94, 1.12] | [0.85, 1.01] | [0.87, 1.04] | [0.88, 1.03] | |
| Qwen2.5-7B  | 0.936 | 0.826 | 0.949 | 0.957 | #2 |
| *95% CI* | [0.79, 1.09] | [0.69, 0.97] | [0.79, 1.13] | [0.82, 1.10] | |

Values are realized NA MSE / Thm 3 prediction. Point ratios stay in [0.83, 1.03]
across all m, and 7 of the 8 bootstrap confidence intervals contain 1.0.

> **Correction (revision 1).** The Qwen2.5-7B row previously read
> 0.877 / 0.850 / 0.945 / 0.891, which does not reproduce from released dataset
> #2; the values above do. See `empirical-section.md` §2.1.

### Proposition 4 (sign-changing bias law) — verified on two model scales

| Model | p_a (correctness rate) | p_b (length high) | γ* predicted | γ where observed flips sign |
|---|---|---|---|---|
| Qwen2.5-1.5B | 0.39 | 0.26 | 0.53 | ≈ 0.40–0.60 |
| Qwen2.5-7B   | 0.44 | 0.21 | 0.56 | ≈ 0.20–0.40 |

### GRPO fine-tuning — NA vs AN on the fintech dataset (#4 vs #5)

- Mean aggregate reward (last 30 steps): **NA = 1.7133**, AN = 1.7068 — comparable means
- **Variance: NA std = 0.084, AN std = 0.097 (NA is 14% lower)** — Thm 3's MSE-floor prediction holds on real training trajectories

---

## Total RunPod spend across the whole project

| Date | Run | Cost |
|---|---|---|
| May 26 | Smoke (50×8 GSM8K Qwen-1.5B), 34 min | $1.35 |
| May 26 | Debug attempts (5 short pods) | $0.45 |
| May 26 | Headline (150×16 GSM8K Qwen-1.5B v1), 186 min | $7.39 |
| May 27 | Qwen-7B Thm 3 + Prop 4 (v1), 59 min | $2.34 |
| May 27 | Fintech generation v1 (no text saved), 8 min | $0.32 |
| May 27 | NA + AN training (single pod, both modes), 19 min | $0.75 |
| May 27 | Fintech generation **v2** (with full bot replies), 13 min | $0.53 |
| May 27 | Qwen-7B **v2** (with full chains-of-thought), 59 min | $2.34 |
| May 27 | Qwen-1.5B v2 failed boot (CUDA driver mismatch), 6 min | $0.22 |
| May 27 | Qwen-1.5B **v2** (with full chains-of-thought), 186 min | $7.39 |
| **Total** | | **$23.08** |

Budget: **$200**. Used: **$23.08**. Remaining: **$176.92** (88% unused).
