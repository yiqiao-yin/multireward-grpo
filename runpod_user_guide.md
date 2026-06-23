# RunPod User Guide — Multi-Reward GRPO LLM Validation

A practical guide for running our LLM validation experiments on RunPod-rented
H100 GPUs. Detailed enough that future-me (or future-you) can hand this file
back and we can reproduce today's run end-to-end without re-deriving any of
the bugs we already debugged.

---

## What this guide is for

The theory in `problem-statement.md` + `proofs.md` makes predictions about the
MSE and bias of multi-reward GRPO advantage estimators. To verify those
predictions on real LLM rollouts (not just synthetic Bernoulli), we need a
machine with a GPU. We don't have one locally — Claude Code runs in a CPU-only
WSL2 environment — so we rent an H100 from RunPod for the few hours it takes
to run the experiment, then terminate.

The whole workflow is automated by `scripts/runpod_launch.py`. This guide
explains what it does, how to invoke it, what to expect, and what the bugs
we hit today taught us.

---

## Architecture

```
your laptop / WSL                  RunPod data center
┌──────────────────────────┐       ┌─────────────────────────────────┐
│ Claude Code (CPU only)   │ HTTPS │ H100 PCIe pod                   │
│ ┌──────────────────────┐ │ ───►  │ ┌─────────────────────────────┐ │
│ │ runpod_launch.py     │ │  API  │ │ bash (started by            │ │
│ │   1. POST /pods      │ │       │ │  RunPod's dockerStartCmd)   │ │
│ │   2. poll RUNNING    │ │       │ │   ↓ apt-get openssh + rsync │ │
│ │   3. wait_for_ssh    │ │  SSH  │ │   ↓ start sshd              │ │
│ │   4. rsync code      │ │ ───►  │ │   ↓ (waits for ssh client)  │ │
│ │   5. ssh bash -s     │ │       │ │                             │ │
│ │       <stdin script> │ │       │ │ ssh client (us) connects    │ │
│ │   6. rsync results   │ │       │ │   ↓ uv install              │ │
│ │   7. POST /stop      │ │       │ │   ↓ uv sync --extra llm     │ │
│ │   8. DELETE /pods/id │ │       │ │   ↓ huggingface model dl    │ │
│ └──────────────────────┘ │       │ │   ↓ llm_validate.py         │ │
│                          │       │ │   ↓ torch + CUDA            │ │
│ ~/.ssh/id_ed25519        │       │ │   ↓ H100 (80 GB VRAM)       │ │
│ /tmp/id_ed25519_runpod   │       │ │       ↓                     │ │
│   (0600 copy)            │       │ │     ←── results back via    │ │
│ .env (RUNPOD_API_KEY)    │       │ │         rsync over SSH      │ │
└──────────────────────────┘       │ └─────────────────────────────┘ │
                                   └─────────────────────────────────┘
```

The orchestrator is synchronous: it makes a REST call, then opens one SSH
connection that stays open for the entire experiment, and blocks until the
remote command finishes. When the SSH session returns, it pulls results back
over the same connection and tears down the pod.

---

## Prerequisites (one-time setup)

### 1. RunPod account

- Sign up at https://runpod.io.
- Add a payment method and put some credit on the account ($20–50 is plenty
  for our experiment scale).
- Generate an API key at https://console.runpod.io/user/settings → API Keys.
  Give it pod read/write permissions.

### 2. SSH key

We use the public key to authorize root login on the pod. The private key
stays on your machine.

```bash
# generate if you don't have one
ssh-keygen -t ed25519 -C "your-email@example.com" -f ~/.ssh/id_ed25519
```

**WSL2 caveat we hit on day 1.** If your `~/.ssh/` lives on a Windows-mounted
filesystem (`/mnt/c/...`, `\\wsl$\...`), `chmod 600` is a no-op and SSH will
refuse to use the key because permissions are still `0777`. The orchestrator
handles this automatically by copying the key to `/tmp/id_ed25519_runpod`
(real Linux tmpfs) on first use and chmod-ing the copy to `0600`. You don't
need to do anything — but if you ever see "permissions are too open" errors
on the key, that's what's happening.

### 3. Local dependencies

```bash
# from the repo root
uv sync                # installs numpy, matplotlib, scipy, pandas, requests
# torch + transformers etc. are NOT needed locally; they only go on the pod
```

The orchestrator itself only needs `requests` for the REST calls plus stdlib
(`subprocess`, `pathlib`, etc.).

### 4. The `.env` file

Create `.env` in the repo root (it's gitignored). See `.env.example` for the
exact format. Minimum required:

```bash
RUNPOD_API_KEY=rpa_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

---

## Daily workflow — running the standard headline experiment

This is what we did today. The single command does everything:

```bash
# This is the canonical "scaled" headline run that produced the GSM8K Thm 3
# figure and the Prop 4 raw rewards npz on May 26 2026 ($1.80 spend).
uv run scripts/runpod_launch.py \
    --n-prompts 150 \
    --K 16 \
    --m-grid 4 8 16 32 \
    --wall-clock-cap 18000
```

What this command does, step by step:

1. **Loads** `.env`, reads `RUNPOD_API_KEY`, reads `~/.ssh/id_ed25519.pub`.
2. **Picks** an H100 GPU type from a preference list:
   `["NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL"]`,
   trying cheapest first.
3. **POSTs `/v1/pods`** with:
   - `imageName: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
   - `gpuTypeIds: [<chosen>]`, `gpuCount: 1`
   - `cloudType: SECURE`, `computeType: GPU`
   - `ports: ["22/tcp"]`
   - `containerDiskInGb: 60`, `volumeInGb: 20`, `volumeMountPath: /workspace`
   - `env.PUBLIC_KEY: <contents of ~/.ssh/id_ed25519.pub>`
   - `dockerStartCmd: [bash, -c, "apt-get install openssh-server rsync; ...; /usr/sbin/sshd -D"]`
4. **Polls** `GET /v1/pods/{id}` every 8 s until `desiredStatus == "RUNNING"`
   and the `22/tcp` port mapping appears (typically 30–90 s).
5. **Probes SSH** every 8 s for up to 6 min (`wait_for_ssh`). Apt-installing
   openssh-server + rsync takes 30–60 s on a fresh image; we wait for `ssh
   root@... 'echo ok'` to return 0.
6. **rsyncs** the repo to `/workspace/conditioned-multireward/` on the pod
   (excludes `.venv`, `.git`, `figures`, `data`, `.env`, `*.lock`, `.claude`,
   Windows `*Zone.Identifier*` files). Uses `--no-owner --no-group` to avoid
   spurious chown errors on RunPod's overlayfs.
7. **Opens** an SSH session and pipes the remote bootstrap script via stdin
   (`ssh ... bash -s`). The bootstrap:
   - installs `uv` if missing
   - runs `uv sync --extra llm` (downloads torch, transformers, datasets,
     accelerate, sentencepiece, protobuf — about 2–3 GB)
   - runs `uv run scripts/llm_validate.py --mode gsm8k --n-prompts {n} ...`
   - the validation script downloads the Qwen2.5-1.5B model (~3 GB on first
     run, cached for subsequent runs) and generates rollouts on the H100
8. **Waits** until the remote command exits. With `--subsample-from-max` the
   work is `n_prompts × K_seeds` model.generate() batches at m_max=32, each
   taking ~5 s on the H100. 150 × 16 = 2400 batches → ~3 hours.
9. **rsyncs results back** to `figures/gsm8k_runpod/` on your machine:
   `llm_money_scatter_gsm8k_m{4,8,16,32}.png`, `llm_mse_vs_m_gsm8k.png`,
   `llm_summary_gsm8k.json`, **`llm_rewards_gsm8k.npz`** (the raw rewards
   tensor for Prop 4 post-processing).
10. **POSTs** `/v1/pods/{id}/stop` then **DELETEs** `/v1/pods/{id}`.
11. **Prints** estimated cost = `elapsed_seconds / 3600 × $2.39`.

Hard safety:
- Wall-clock cap (`--wall-clock-cap`, default 5400 s). After this, the script
  raises and the `finally` block terminates the pod.
- `try/finally` around the whole flow: any unhandled exception triggers pod
  cleanup. The pod is never left running.

---

## After the run — local Prop 4 post-processing

The remote experiment produces a raw-rewards `.npz` containing
`(correctness, length, format)` triples per rollout plus the uncontaminated
`raw_length` per rollout. The Prop 4 γ sweep is then a cheap local script:

```bash
uv run scripts/llm_prop4_real.py \
    --rewards-npz figures/gsm8k_runpod/llm_rewards_gsm8k.npz \
    --label gsm8k \
    --n-replicates 64
```

This produces `figures/llm_prop4_bias_law_gsm8k.png` and a JSON summary.
Takes seconds — no GPU, no RunPod.

---

## Cost expectations

| Experiment | Config | Pod time | Cost @ $2.39/hr H100 PCIe |
|---|---|---|---|
| Smoke run | 50 prompts × K=8 | ~30–40 min | **~$1.50** |
| Headline | 150 prompts × K=16 | ~3 hours | **~$7–10** |
| Bigger model | Qwen2.5-7B at same scale | ~5–6 hours | **~$15–25** |

Cumulative spend on this project as of May 26 2026: **$1.80** (one smoke run
+ five debug attempts that auto-terminated cleanly).

---

## Specific commands we used today (May 26 2026)

### First successful smoke run (Thm 3 only)

```bash
uv run scripts/runpod_launch.py \
    --n-prompts 50 \
    --K 8 \
    --m-grid 4 8 16 32 \
    --wall-clock-cap 5400
```
**Result:** Pod `leqbir4wefp3ac`, 34 min, $1.35. NA/pred ratio 0.74–0.97
across m=4–32, AN/pred ratio 1.12–1.37. Real GSM8K reward correlation
`corr(correctness, length) = -0.658`. Theorem 3 verified.

### Scaled headline run (Thm 3 + raw rewards for Prop 4)

```bash
uv run scripts/runpod_launch.py \
    --n-prompts 150 \
    --K 16 \
    --m-grid 4 8 16 32 \
    --wall-clock-cap 18000
```
**Result:** [in progress as of this writing]

### Local Prop 4 sweep

```bash
uv run scripts/llm_prop4_real.py \
    --rewards-npz figures/gsm8k_runpod/llm_rewards_gsm8k.npz \
    --label gsm8k \
    --n-replicates 64
```

---

## Mock-mode pre-flight (no GPU)

Before any RunPod run, validate the pipeline locally:

```bash
uv run scripts/llm_validate.py --mode mock \
    --n-prompts 80 --K 32 --m-grid 4 8 16 32 --subsample-from-max
```

Acceptance: realized NA / predicted MSE ratio in [0.9, 1.1] mean across
prompts at all m. If this fails, do **not** spend GPU money — fix the
analysis first.

Then:
```bash
uv run scripts/llm_prop4_real.py \
    --rewards-npz figures/llm_rewards_mock.npz --label mock_test \
    --n-replicates 16
```

Acceptance: observed β tracks predicted β across γ ∈ [0, 3] within ~15%,
sign change near γ\* ≈ 1.0.

---

## Common pitfalls (what we debugged today, baked into the script)

These are all handled by the current `scripts/runpod_launch.py`. They are
documented here so a future reader understands why the code looks the way
it does, and so debugging a regression has a starting checklist.

| # | Symptom | Root cause | Fix in code |
|---|---|---|---|
| 1 | "Permissions are too open" / "Permission denied" on SSH | WSL's `~/.ssh/` is on a Windows-mounted FS where `chmod` is a no-op; SSH refuses keys with looser-than-0600 perms | `_ensure_usable_ssh_key()` copies the key to `/tmp/id_ed25519_runpod` and chmods the copy |
| 2 | Pod is RUNNING with port 22 mapped, but SSH probes time out | RunPod's `runpod/pytorch:*` image doesn't start sshd by default when launched via REST `/v1/pods` (only when launched via a web Template) | Pass `dockerStartCmd` explicitly running `apt-get install openssh-server` + `/usr/sbin/sshd -D` |
| 3 | Remote bash prints `[remote] uname:` then exits with `bash: line 1: ...\nfoo: command not found` | `json.dumps(multi_line_command)` escapes `\n` as the literal 2-char sequence, not a newline | Pipe the script via stdin: `ssh ... bash -s` with `input=command.encode()` instead of inline |
| 4 | `bash: line 1: rsync: command not found` | The PyTorch image ships without rsync | `dockerStartCmd` also installs rsync alongside openssh-server |
| 5 | `rsync: [generator] chown ... failed: Operation not permitted` → rc=23 → script aborts | RunPod's overlayfs / network volume on `/workspace` doesn't allow `chown` even as root | Use `--no-owner --no-group` (i.e. `-rltDvz` instead of `-avz`) — files transfer correctly without the ownership step |
| 6 | Orchestrator output file is empty even though Python is running | `print()` in non-tty subprocess is fully buffered | `sys.stdout.reconfigure(line_buffering=True)` at the top of `runpod_launch.py` |
| 7 | Pod is auto-terminated before you can debug | The orchestrator's `try/finally` will run cleanup on *any* exit including KeyboardInterrupt | For a manual recovery, kill the orchestrator with `kill -9 <pid>` (SIGKILL skips the `finally`); then drive the live pod by hand and call `DELETE /v1/pods/{id}` yourself when done |

---

## What the headline experiment validates

| Figure / file | Verifies | Predicted signature |
|---|---|---|
| `figures/gsm8k_runpod/llm_mse_vs_m_gsm8k.png` | Theorem 3 on real LLM rollouts | NA tracks the dashed `(1/m) w^T Ĉ w` line; AN drifts on a different line (Prop 1) |
| `figures/gsm8k_runpod/llm_money_scatter_gsm8k_m{m}.png` | Per-prompt predicted-vs-realized | NA log-log fit slope ≈ 1, R² ≈ 0.99 |
| `figures/gsm8k_runpod/llm_summary_gsm8k.json` | All aggregate numbers + pooled Ĉ | NA/pred mean ratio ∈ [0.85, 1.15]; pooled corr(c, length) negative on GSM8K |
| `figures/llm_prop4_bias_law_gsm8k.png` | Proposition 4 bias law on real rewards | Observed β tracks closed-form prediction; sign change at γ\* = α_c p_a / (1−p_b) |

---

## What to do if you give me this file in a future session

If you start a new conversation with me and want to reproduce or extend this
work, the prompt template that works is roughly:

```
Please read runpod_user_guide.md and follow it to:

(a) [or replicate the headline experiment]
    Spawn an H100 pod and run the scaled headline experiment
    (150 prompts × K=16 × m=4,8,16,32 on Qwen2.5-1.5B-Instruct).
    Wall-clock cap 5 hours. Budget $10.

(b) [or run a variant]
    Same as (a) but with Qwen2.5-7B-Instruct, n_prompts=100, K=8.
    Budget $25.

(c) [or just the post-processing]
    Run scripts/llm_prop4_real.py against the existing
    figures/gsm8k_runpod/llm_rewards_gsm8k.npz from May 26 2026.

I authorize you to spawn one pod and spend up to $N. If anything in
runpod_user_guide.md is out of date, flag it before proceeding.
```

When I see this, I will:
1. Read this file and confirm the orchestrator is still in `scripts/runpod_launch.py`.
2. Read `.env` to confirm `RUNPOD_API_KEY` is set.
3. Make a dry-run: `uv run scripts/runpod_launch.py --dry-run`.
4. Execute the requested run with hard wall-clock cap and budget ceiling.
5. Wait via the Monitor tool, report progress.
6. On completion, verify the expected output files exist, report cost,
   confirm the pod is terminated.
7. If post-processing is needed (Prop 4), run `llm_prop4_real.py` locally.

I will **not** spawn a pod without your explicit authorization in the same
turn. The "go" needs to be unambiguous (e.g., "yes, proceed with run (a)").

---

## File reference (where things live)

```
conditioned-multireward/
├── .env                              ← your API key (gitignored)
├── .env.example                      ← template
├── runpod_user_guide.md              ← this file
├── pyproject.toml                    ← `uv sync --extra llm` on the pod
├── scripts/
│   ├── runpod_launch.py              ← the orchestrator (lifecycle + SSH + rsync)
│   ├── llm_validate.py               ← the experiment driver (runs on the pod)
│   ├── llm_generation.py             ← QwenBackend + Rollouts dataclass + save_corpus_npz
│   ├── llm_rewards.py                ← correctness / length / format reward fns
│   ├── llm_analysis.py               ← MSE estimators, advantage formulas, money plot
│   └── llm_prop4_real.py             ← local γ-sweep post-processor (no GPU)
└── figures/
    └── gsm8k_runpod/                 ← outputs from real-LLM runs land here
        ├── llm_money_scatter_*_m{m}.png
        ├── llm_mse_vs_m_*.png
        ├── llm_summary_*.json
        └── llm_rewards_*.npz         ← raw rewards for Prop 4 post-processing
```

---

## Quick reference: REST endpoints used

All against base `https://rest.runpod.io/v1`, auth via `Authorization: Bearer
$RUNPOD_API_KEY`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/pods` | Create a pod. Body keys we use: `name`, `imageName`, `gpuTypeIds`, `gpuCount`, `cloudType`, `computeType`, `ports`, `containerDiskInGb`, `volumeInGb`, `volumeMountPath`, `env`, `dockerStartCmd` |
| GET | `/pods` | List pods (returns array; we use `[]` empty check) |
| GET | `/pods/{id}` | Get one pod; check `desiredStatus`, `publicIp`, `portMappings` |
| POST | `/pods/{id}/stop` | Stop a pod (stops billing for compute, keeps the volume) |
| DELETE | `/pods/{id}` | Delete the pod entirely (we always do stop + delete) |

The full OpenAPI spec is at `https://rest.runpod.io/v1/openapi.json` if you
need a field we haven't used.

---

## Quick reference: CLI commands

```bash
# preflight (CPU, ~30 sec)
uv run scripts/llm_validate.py --mode mock --n-prompts 80 --K 32 \
    --m-grid 4 8 16 32 --subsample-from-max

# dry-run orchestrator (no API call, no spend)
uv run scripts/runpod_launch.py --dry-run

# smoke run on RunPod (~30–40 min, ~$1.50)
uv run scripts/runpod_launch.py --n-prompts 50 --K 8 \
    --m-grid 4 8 16 32 --wall-clock-cap 5400

# headline run on RunPod (~3 hours, ~$7–10)
uv run scripts/runpod_launch.py --n-prompts 150 --K 16 \
    --m-grid 4 8 16 32 --wall-clock-cap 18000

# Prop 4 post-processing (CPU, ~10 sec)
uv run scripts/llm_prop4_real.py \
    --rewards-npz figures/gsm8k_runpod/llm_rewards_gsm8k.npz \
    --label gsm8k --n-replicates 64

# manual pod inspection (if something is wedged)
set -a; source .env; set +a
curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods" | python3 -m json.tool

# manual pod termination (escape hatch)
curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/<POD_ID>/stop"
curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/<POD_ID>"
```
