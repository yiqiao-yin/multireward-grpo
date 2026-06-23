"""
RunPod orchestrator for the GSM8K Thm 3 validation.

Lifecycle:
  1. Read API key from .env, SSH public key from ~/.ssh/id_ed25519.pub
  2. Resolve an H100 PCIe GPU type id (Secure Cloud).
  3. POST /v1/pods with PyTorch base image + PUBLIC_KEY env + port 22 mapped.
  4. Poll status until RUNNING and the 22/tcp public port is published.
  5. rsync the repo to the pod (scripts/ + pyproject.toml).
  6. SSH and run: install uv → sync deps → run llm_validate.py.
  7. rsync results back into figures/gsm8k_runpod/.
  8. POST /v1/pods/{id}/stop  then  DELETE /v1/pods/{id}.

Hard safety:
  - Wall-clock cap (default 5400 s = 90 min). On expiry, terminate.
  - try/finally ensures pod is stopped and deleted on any exception.
  - Cost accounting printed at the end (start + end timestamps × hourly rate).
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import subprocess
import json
import signal
from pathlib import Path
from typing import Optional

import requests


API_BASE = "https://rest.runpod.io/v1"
REPO_ROOT = Path(__file__).resolve().parent.parent

# Unbuffered stdout — Python buffers print() in non-tty mode and we want every
# event to appear in the tail-able output file immediately.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def load_env() -> dict:
    env = {}
    dotenv = REPO_ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


# ---------------------------------------------------------------------------
# tiny REST wrapper
# ---------------------------------------------------------------------------
class RunPod:
    def __init__(self, api_key: str):
        self.s = requests.Session()
        self.s.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def get(self, path, **kw):
        r = self.s.get(f"{API_BASE}{path}", timeout=30, **kw)
        if not r.ok:
            raise RuntimeError(f"GET {path}: {r.status_code} {r.text[:400]}")
        return r.json()

    def post(self, path, **kw):
        r = self.s.post(f"{API_BASE}{path}", timeout=60, **kw)
        if not r.ok:
            raise RuntimeError(f"POST {path}: {r.status_code} {r.text[:400]}")
        return r.json() if r.text else {}

    def delete(self, path, **kw):
        r = self.s.delete(f"{API_BASE}{path}", timeout=30, **kw)
        if not r.ok and r.status_code != 404:
            raise RuntimeError(f"DELETE {path}: {r.status_code} {r.text[:400]}")
        return {}


# ---------------------------------------------------------------------------
# pod lifecycle
# ---------------------------------------------------------------------------
H100_TYPE_PREFERENCES = [
    # Cheapest first; we try in order and pick the first that the API accepts.
    "NVIDIA H100 PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 NVL",
]


def pick_image() -> str:
    """Pick a base image that ships PyTorch + CUDA + SSH."""
    # Known-good RunPod official PyTorch image with SSH pre-wired
    return "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def create_pod(rp: RunPod, name: str, public_key: str, gpu_type: str,
               container_disk_gb: int = 60, volume_gb: int = 20) -> dict:
    # We override the container start command so we have a guaranteed,
    # idempotent sshd-up state. The official runpod/pytorch image has
    # openssh-server installed but does not start sshd by default when the
    # image is launched directly via /v1/pods (vs through a web template).
    # This start command:
    #   - installs openssh-server if it's somehow missing
    #   - appends $PUBLIC_KEY to /root/.ssh/authorized_keys
    #   - enables root login and starts sshd
    #   - sleeps forever so the container stays up
    docker_start_cmd = [
        "bash", "-c",
        (
            "set -e; "
            # rsync is required for rsync_to/from_pod; the pytorch image
            # doesn't ship it. apt-get is idempotent if already installed.
            "apt-get update -qq && apt-get install -y -qq openssh-server rsync; "
            "mkdir -p /root/.ssh && chmod 700 /root/.ssh; "
            "echo \"$PUBLIC_KEY\" >> /root/.ssh/authorized_keys; "
            "chmod 600 /root/.ssh/authorized_keys; "
            "sed -i 's/^#\\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config; "
            "sed -i 's/^#\\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config; "
            "mkdir -p /run/sshd; "
            "/usr/sbin/sshd -D"
        ),
    ]
    body = {
        "name": name,
        "imageName": pick_image(),
        "gpuTypeIds": [gpu_type],
        "gpuCount": 1,
        "cloudType": "SECURE",
        "computeType": "GPU",
        "ports": ["22/tcp"],
        "containerDiskInGb": container_disk_gb,
        "volumeInGb": volume_gb,
        "volumeMountPath": "/workspace",
        "env": {
            "PUBLIC_KEY": public_key,
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "dockerStartCmd": docker_start_cmd,
    }
    return rp.post("/pods", json=body)


def wait_for_ready(rp: RunPod, pod_id: str, timeout_s: int = 600) -> dict:
    """Poll until pod is RUNNING and port 22 is published."""
    start = time.time()
    last_status = None
    while time.time() - start < timeout_s:
        p = rp.get(f"/pods/{pod_id}")
        status = p.get("desiredStatus")
        if status != last_status:
            print(f"  pod status: {status}")
            last_status = status
        # Look for port 22 in portMappings
        port_map = p.get("portMappings") or {}
        ssh_port = port_map.get("22")
        public_ip = p.get("publicIp")
        if status == "RUNNING" and ssh_port and public_ip:
            return p
        time.sleep(8)
    raise TimeoutError(f"pod {pod_id} did not become RUNNING within {timeout_s}s")


def stop_pod(rp: RunPod, pod_id: str) -> None:
    print(f"  stopping pod {pod_id} ...")
    try:
        rp.post(f"/pods/{pod_id}/stop")
    except Exception as e:
        print(f"    stop failed (ignored): {e}")


def delete_pod(rp: RunPod, pod_id: str) -> None:
    print(f"  deleting pod {pod_id} ...")
    try:
        rp.delete(f"/pods/{pod_id}")
    except Exception as e:
        print(f"    delete failed (ignored): {e}")


# ---------------------------------------------------------------------------
# SSH / rsync wrappers
# ---------------------------------------------------------------------------
def _ensure_usable_ssh_key() -> str:
    """Return path to an SSH key with 0600 perms (SSH refuses 0644+).

    On WSL2 the user's ~/.ssh may live on a Windows-mounted filesystem where
    chmod is a no-op; we copy the key to /tmp (real Linux fs) and lock down
    the perms there. Idempotent on Unix-perm-respecting filesystems.
    """
    src = Path.home() / ".ssh" / "id_ed25519"
    if not src.exists():
        raise FileNotFoundError(f"missing SSH private key at {src}")
    src_stat = src.stat()
    if (src_stat.st_mode & 0o077) == 0:
        return str(src)
    dst = Path("/tmp") / "id_ed25519_runpod"
    dst.write_bytes(src.read_bytes())
    dst.chmod(0o600)
    return str(dst)


SSH_KEY = _ensure_usable_ssh_key()
SSH_OPTS = (
    f"-o StrictHostKeyChecking=no "
    f"-o UserKnownHostsFile=/dev/null "
    f"-o ServerAliveInterval=30 "
    f"-o ConnectTimeout=20 "
    f"-i {SSH_KEY}"
)


def ssh_run(host: str, port: int, command: str, timeout_s: int) -> int:
    """Run a (possibly multi-line) script on the remote via stdin pipe.

    Using `ssh ... bash -s` and piping the script as stdin avoids ALL shell
    quoting concerns — newlines, quotes, and escapes all pass through cleanly.
    """
    n_lines = command.count("\n") + 1
    print(f"  $ ssh ... -p {port} root@{host} bash -s  (<{n_lines}-line script via stdin>)")
    cmd = f"ssh {SSH_OPTS} -p {port} root@{host} bash -s"
    proc = subprocess.run(
        cmd, shell=True, input=command.encode(),
        timeout=timeout_s,
    )
    return proc.returncode


# RunPod overlayfs / network volumes don't permit chown of files in
# /workspace, so `-a` (which implies -ogp) produces spurious "Operation not
# permitted" warnings and a non-zero exit code (23 = partial transfer).
# We drop owner/group preservation; files still copy correctly.
RSYNC_FLAGS = "-rltDvz --no-owner --no-group"


def rsync_to_pod(host: str, port: int, src: str, dest: str) -> int:
    cmd = (
        f"rsync {RSYNC_FLAGS} --delete "
        f"-e 'ssh {SSH_OPTS} -p {port}' "
        f"--exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' "
        f"--exclude '.git' --exclude 'figures' --exclude 'data' "
        f"--exclude '.env' --exclude '*.lock' --exclude '.claude' "
        f"--exclude '*Zone.Identifier*' "
        f"{src} root@{host}:{dest}"
    )
    print(f"  $ rsync ... root@{host}:{dest}")
    return subprocess.call(cmd, shell=True)


def rsync_from_pod(host: str, port: int, src: str, dest: str) -> int:
    cmd = (
        f"rsync {RSYNC_FLAGS} "
        f"-e 'ssh {SSH_OPTS} -p {port}' "
        f"root@{host}:{src} {dest}"
    )
    print(f"  $ rsync from root@{host}:{src} -> {dest}")
    return subprocess.call(cmd, shell=True)


# ---------------------------------------------------------------------------
# wait for ssh to actually be accepting (port-open != sshd-up)
# ---------------------------------------------------------------------------
def wait_for_ssh(host: str, port: int, timeout_s: int = 360) -> None:
    """Wait for sshd to actually be accepting connections.

    Our dockerStartCmd runs apt update + install + sshd-start on first launch,
    which can take 60-120s on a fresh image. We poll every 8s for up to 6 min.
    """
    start = time.time()
    attempt = 0
    while time.time() - start < timeout_s:
        attempt += 1
        rc = subprocess.call(
            f"ssh {SSH_OPTS} -p {port} root@{host} 'echo ok' >/dev/null 2>&1",
            shell=True, timeout=20,
        )
        if rc == 0:
            print(f"  ssh ready after {int(time.time()-start)}s (attempt {attempt})")
            return
        if attempt == 1 or attempt % 5 == 0:
            print(f"  ssh probe {attempt}: not yet (elapsed {int(time.time()-start)}s)")
        time.sleep(8)
    raise TimeoutError(f"ssh did not accept connections within {timeout_s}s")


# ---------------------------------------------------------------------------
# the remote experiment script
# ---------------------------------------------------------------------------
REMOTE_BOOTSTRAP = """\
set -euo pipefail
echo '[remote] uname:'; uname -a
echo '[remote] nvidia-smi:'; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd /workspace/conditioned-multireward

# install uv if missing
if ! command -v uv >/dev/null 2>&1; then
  echo '[remote] installing uv ...'
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# sync deps (core + llm)
echo '[remote] uv sync --extra llm ...'
uv sync --extra llm

# run validation
echo '[remote] starting GSM8K validation ...'
mkdir -p figures
uv run scripts/llm_validate.py \\
    --mode gsm8k \\
    --n-prompts {n_prompts} \\
    --K {K} \\
    --m-grid {m_grid_str} \\
    --model {model} \\
    --subsample-from-max

echo '[remote] DONE'
ls -la figures/llm_*
"""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=50)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--m-grid", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--wall-clock-cap", type=int, default=5400,
                    help="seconds; force termination after this")
    ap.add_argument("--dry-run", action="store_true",
                    help="print actions without spawning anything")
    args = ap.parse_args()

    env = load_env()
    api_key = env.get("RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("RUNPOD_API_KEY not set in .env or environment")

    pub_key_path = Path.home() / ".ssh" / "id_ed25519.pub"
    if not pub_key_path.exists():
        sys.exit(f"missing public key at {pub_key_path}")
    public_key = pub_key_path.read_text().strip()

    rp = RunPod(api_key)

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"would create pod with image={pick_image()}")
        print(f"would inject PUBLIC_KEY = {public_key[:40]}...")
        print(f"would run: {args.n_prompts} prompts × {args.K} seeds × m_grid={args.m_grid}")
        return

    pod_id: Optional[str] = None
    public_ip: Optional[str] = None
    ssh_port: Optional[int] = None
    t_pod_started: Optional[float] = None
    HOURLY_RATE_USD = 2.39  # H100 PCIe Secure Cloud

    try:
        # ----------- create pod -----------
        gpu_choice = H100_TYPE_PREFERENCES[0]
        print(f"=== creating pod (gpuType={gpu_choice}) ===")
        # try each gpu preference in order
        created = None
        last_err = None
        for gt in H100_TYPE_PREFERENCES:
            try:
                created = create_pod(rp, "gsm8k-thm3", public_key, gt)
                gpu_choice = gt
                break
            except RuntimeError as e:
                last_err = e
                print(f"  gpuType={gt} rejected: {str(e)[:150]}")
        if created is None:
            raise RuntimeError(f"no H100 variant accepted: {last_err}")
        pod_id = created.get("id") or created.get("podId") or created.get("pod", {}).get("id")
        if not pod_id:
            raise RuntimeError(f"could not extract pod id from response: {json.dumps(created)[:400]}")
        t_pod_started = time.time()
        print(f"  pod created: {pod_id}  (billing starts NOW)")

        # ----------- wait for RUNNING + port -----------
        p = wait_for_ready(rp, pod_id, timeout_s=600)
        public_ip = p["publicIp"]
        ssh_port = int(p["portMappings"]["22"])
        print(f"  RUNNING on {public_ip}:{ssh_port}")

        # ----------- wait for sshd to actually accept -----------
        wait_for_ssh(public_ip, ssh_port, timeout_s=360)

        # ----------- rsync code to pod -----------
        rc = rsync_to_pod(public_ip, ssh_port, f"{REPO_ROOT}/", "/workspace/conditioned-multireward/")
        if rc != 0:
            raise RuntimeError(f"rsync to pod failed (rc={rc}); aborting before remote run")

        # ----------- run experiment -----------
        remote_cmd = REMOTE_BOOTSTRAP.format(
            n_prompts=args.n_prompts,
            K=args.K,
            m_grid_str=" ".join(map(str, args.m_grid)),
            model=args.model,
        )
        elapsed_so_far = time.time() - t_pod_started
        remaining = max(60, args.wall_clock_cap - int(elapsed_so_far) - 300)  # leave 5 min for cleanup
        rc = ssh_run(public_ip, ssh_port, remote_cmd, timeout_s=remaining)
        if rc != 0:
            print(f"  remote command exited with code {rc} — pulling whatever results exist anyway")

        # ----------- pull figures back -----------
        local_out = REPO_ROOT / "figures" / "gsm8k_runpod"
        local_out.mkdir(parents=True, exist_ok=True)
        rsync_from_pod(public_ip, ssh_port,
                      "/workspace/conditioned-multireward/figures/llm_*",
                      str(local_out) + "/")

    finally:
        if pod_id is not None and t_pod_started is not None:
            elapsed = time.time() - t_pod_started
            print(f"\n=== teardown (pod lived {elapsed:.0f}s = {elapsed/60:.1f} min) ===")
            stop_pod(rp, pod_id)
            time.sleep(3)
            delete_pod(rp, pod_id)
            cost = elapsed / 3600 * HOURLY_RATE_USD
            print(f"  estimated cost: ${cost:.2f} ({elapsed/60:.1f} min × ${HOURLY_RATE_USD}/hr)")

    print("\n=== DONE ===")
    print(f"results in: {REPO_ROOT / 'figures' / 'gsm8k_runpod'}")


if __name__ == "__main__":
    main()
