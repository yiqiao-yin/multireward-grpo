"""
Push trained NA and AN model checkpoints to HF Hub from the local machine.

Use after training pod returns with checkpoints rsync'd to
figures/grpo_train/{mode}/checkpoint_step*. Same effect as the in-pod
hf_push_model.py script, just run locally because the pod's bash session
didn't have HF_TOKEN exported (pre-fix runs).

Usage:
    set -a; source .env; set +a
    uv run scripts/hf_push_models_local.py --modes na an
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from huggingface_hub import HfApi, create_repo, whoami

REPO_ROOT = Path(__file__).resolve().parent.parent

# import the same model card template + helper as hf_push_model.py
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import hf_push_model as hpm  # noqa: E402


def push_one(mode: str, token: str) -> str:
    os.environ["HF_TOKEN"] = token
    os.environ["HF_MODE"] = mode

    api = HfApi(token=token)
    me = whoami(token=token)
    user = me["name"]
    repo_id = f"{user}/multireward-grpo-fintech-{mode}-qwen2.5-1.5b"
    print(f"target: huggingface.co/{repo_id}", flush=True)

    ckpt = hpm.find_latest_checkpoint(mode)
    print(f"  checkpoint: {ckpt}", flush=True)

    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True,
                token=token, private=False)

    card = hpm.MODEL_CARD_TEMPLATE.format(
        mode=mode, mode_upper=mode.upper(),
        mode_long=hpm.MODE_DESCRIPTIONS[mode],
    )
    (ckpt / "README.md").write_text(card)

    api.upload_folder(
        folder_path=str(ckpt),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"GRPO ({mode.upper()} advantage) — final checkpoint",
        ignore_patterns=["*.tmp"],
    )
    metrics_path = ckpt.parent / "metrics.json"
    if metrics_path.exists():
        api.upload_file(
            path_or_fileobj=str(metrics_path),
            path_in_repo="metrics.json",
            repo_id=repo_id,
            repo_type="model",
        )
    url = f"https://huggingface.co/{repo_id}"
    print(f"  ✓ {url}", flush=True)
    return url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["na", "an"])
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        # try loading from .env
        env_file = REPO_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip("'\"")
                    break
        if not token:
            sys.exit("HF_TOKEN not in env or .env")

    urls = []
    for mode in args.modes:
        urls.append(push_one(mode, token))

    print("\n=== DONE ===")
    for url in urls:
        print(f"  {url}")


if __name__ == "__main__":
    main()
