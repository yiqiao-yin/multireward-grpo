"""
Shared RunPod orchestration primitives — pod lifecycle, SSH, rsync, env load.

Used by:
  - scripts/runpod_launch.py   (gsm8k validation)
  - scripts/runpod_fintech.py  (fintech dataset generation + HF push)
  - scripts/runpod_train.py    (GRPO training + HF push)
"""
from __future__ import annotations
import os
import sys
import time
import subprocess
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


API_BASE = "https://rest.runpod.io/v1"
REPO_ROOT = Path(__file__).resolve().parent.parent

# H100 PCIe Secure Cloud as of May 2026 — adjust if RunPod prices change.
HOURLY_RATE_USD = 2.39
H100_TYPE_PREFERENCES = [
    "NVIDIA H100 PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 NVL",
]
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def load_env() -> dict:
    env = {}
    p = REPO_ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


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


@dataclass
class PodSpec:
    name: str
    gpu_count: int = 1
    container_disk_gb: int = 60
    volume_gb: int = 20
    extra_env: dict = None


def docker_start_cmd() -> list[str]:
    return [
        "bash", "-c",
        (
            "set -e; "
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


def create_pod(rp: RunPod, spec: PodSpec, public_key: str,
               hf_token: Optional[str], gpu_type: str) -> dict:
    env = {
        "PUBLIC_KEY": public_key,
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGINGFACE_HUB_TOKEN"] = hf_token
    if spec.extra_env:
        env.update(spec.extra_env)
    body = {
        "name": spec.name,
        "imageName": DEFAULT_IMAGE,
        "gpuTypeIds": [gpu_type],
        "gpuCount": spec.gpu_count,
        "cloudType": "SECURE",
        "computeType": "GPU",
        "ports": ["22/tcp"],
        "containerDiskInGb": spec.container_disk_gb,
        "volumeInGb": spec.volume_gb,
        "volumeMountPath": "/workspace",
        "env": env,
        "dockerStartCmd": docker_start_cmd(),
    }
    return rp.post("/pods", json=body)


def wait_for_ready(rp: RunPod, pod_id: str, timeout_s: int = 600) -> dict:
    start = time.time()
    last_status = None
    while time.time() - start < timeout_s:
        p = rp.get(f"/pods/{pod_id}")
        status = p.get("desiredStatus")
        if status != last_status:
            print(f"  pod status: {status}", flush=True)
            last_status = status
        port_map = p.get("portMappings") or {}
        ssh_port = port_map.get("22")
        public_ip = p.get("publicIp")
        if status == "RUNNING" and ssh_port and public_ip:
            return p
        time.sleep(8)
    raise TimeoutError(f"pod {pod_id} did not become RUNNING within {timeout_s}s")


def stop_pod(rp: RunPod, pod_id: str) -> None:
    print(f"  stopping pod {pod_id} ...", flush=True)
    try:
        rp.post(f"/pods/{pod_id}/stop")
    except Exception as e:
        print(f"    stop failed (ignored): {e}", flush=True)


def delete_pod(rp: RunPod, pod_id: str) -> None:
    print(f"  deleting pod {pod_id} ...", flush=True)
    try:
        rp.delete(f"/pods/{pod_id}")
    except Exception as e:
        print(f"    delete failed (ignored): {e}", flush=True)


def _ensure_usable_ssh_key() -> str:
    src = Path.home() / ".ssh" / "id_ed25519"
    if not src.exists():
        raise FileNotFoundError(f"missing SSH private key at {src}")
    if (src.stat().st_mode & 0o077) == 0:
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
RSYNC_FLAGS = "-rltDvz --no-owner --no-group"


def ssh_run(host: str, port: int, command: str, timeout_s: int,
            env_vars: Optional[dict] = None) -> int:
    """Run a script via SSH stdin. Env vars get exported at the top of the
    script (RunPod's pod-level env doesn't propagate into SSH sessions —
    container env != SSH-session env)."""
    parts = []
    if env_vars:
        for k, v in env_vars.items():
            if v is None:
                continue
            # single-quote the value, escape any single quotes inside
            esc = str(v).replace("'", "'\\''")
            parts.append(f"export {k}='{esc}'")
    parts.append(command)
    full = "\n".join(parts)
    n_lines = full.count("\n") + 1
    print(f"  $ ssh ... -p {port} root@{host} bash -s  (<{n_lines}-line script via stdin>)", flush=True)
    cmd = f"ssh {SSH_OPTS} -p {port} root@{host} bash -s"
    proc = subprocess.run(cmd, shell=True, input=full.encode(), timeout=timeout_s)
    return proc.returncode


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
    print(f"  $ rsync ... root@{host}:{dest}", flush=True)
    return subprocess.call(cmd, shell=True)


def rsync_from_pod(host: str, port: int, src: str, dest: str) -> int:
    cmd = (
        f"rsync {RSYNC_FLAGS} "
        f"-e 'ssh {SSH_OPTS} -p {port}' "
        f"root@{host}:{src} {dest}"
    )
    print(f"  $ rsync from root@{host}:{src} -> {dest}", flush=True)
    return subprocess.call(cmd, shell=True)


def wait_for_ssh(host: str, port: int, timeout_s: int = 360) -> None:
    start = time.time()
    attempt = 0
    while time.time() - start < timeout_s:
        attempt += 1
        rc = subprocess.call(
            f"ssh {SSH_OPTS} -p {port} root@{host} 'echo ok' >/dev/null 2>&1",
            shell=True, timeout=20,
        )
        if rc == 0:
            print(f"  ssh ready after {int(time.time()-start)}s (attempt {attempt})", flush=True)
            return
        if attempt == 1 or attempt % 5 == 0:
            print(f"  ssh probe {attempt}: not yet (elapsed {int(time.time()-start)}s)", flush=True)
        time.sleep(8)
    raise TimeoutError(f"ssh did not accept connections within {timeout_s}s")


def run_pod_experiment(
    *,
    name: str,
    remote_script: str,
    pull_glob: str,
    local_out: Path,
    wall_clock_cap: int,
    gpu_count: int = 1,
    container_disk_gb: int = 60,
    extra_env: Optional[dict] = None,
    needs_hf: bool = False,
):
    """End-to-end: spawn pod, ssh code in, run script, pull back, terminate.

    Returns a dict with {pod_id, cost_usd, elapsed_s, returncode}.
    """
    sys.stdout.reconfigure(line_buffering=True)
    env = load_env()
    api_key = env.get("RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("RUNPOD_API_KEY not set")
    hf_token = (env.get("HF_TOKEN") or os.environ.get("HF_TOKEN")) if needs_hf else None
    if needs_hf and not hf_token:
        sys.exit("HF_TOKEN not set but this experiment needs to push to HF")

    pub_key = (Path.home() / ".ssh" / "id_ed25519.pub").read_text().strip()
    rp = RunPod(api_key)

    pod_id = None
    t_started = None
    rc = -1
    try:
        spec = PodSpec(name=name, gpu_count=gpu_count,
                       container_disk_gb=container_disk_gb,
                       extra_env=extra_env)
        # try cheapest H100 variant first
        last_err = None
        created = None
        for gt in H100_TYPE_PREFERENCES:
            try:
                created = create_pod(rp, spec, pub_key, hf_token, gt)
                print(f"=== created pod (gpuType={gt}) ===", flush=True)
                break
            except RuntimeError as e:
                last_err = e
                print(f"  gpuType={gt} rejected: {str(e)[:150]}", flush=True)
        if created is None:
            raise RuntimeError(f"no H100 variant accepted: {last_err}")
        pod_id = (created.get("id") or created.get("podId")
                  or created.get("pod", {}).get("id"))
        t_started = time.time()
        print(f"  pod {pod_id}  (billing starts NOW)", flush=True)

        p = wait_for_ready(rp, pod_id, timeout_s=600)
        public_ip = p["publicIp"]
        ssh_port = int(p["portMappings"]["22"])
        print(f"  RUNNING on {public_ip}:{ssh_port}", flush=True)
        wait_for_ssh(public_ip, ssh_port, timeout_s=360)

        sync_rc = rsync_to_pod(public_ip, ssh_port,
                               f"{REPO_ROOT}/",
                               "/workspace/conditioned-multireward/")
        if sync_rc != 0:
            raise RuntimeError(f"rsync to pod failed (rc={sync_rc})")

        elapsed_so_far = time.time() - t_started
        remaining = max(60, wall_clock_cap - int(elapsed_so_far) - 300)
        # Pod env vars (passed to RunPod at create time) do NOT propagate
        # into SSH session env. We export them at the top of the bash script
        # so the remote scripts (hf_push_*.py) can read them.
        ssh_env = {}
        if needs_hf and hf_token:
            ssh_env["HF_TOKEN"] = hf_token
            ssh_env["HUGGINGFACE_HUB_TOKEN"] = hf_token
        rc = ssh_run(public_ip, ssh_port, remote_script,
                     timeout_s=remaining, env_vars=ssh_env)
        if rc != 0:
            print(f"  remote command exit code {rc} — pulling artifacts anyway", flush=True)

        local_out.mkdir(parents=True, exist_ok=True)
        rsync_from_pod(public_ip, ssh_port,
                       f"/workspace/conditioned-multireward/{pull_glob}",
                       str(local_out) + "/")

    finally:
        if pod_id is not None and t_started is not None:
            elapsed = time.time() - t_started
            print(f"\n=== teardown (pod lived {elapsed:.0f}s = {elapsed/60:.1f} min) ===", flush=True)
            stop_pod(rp, pod_id)
            time.sleep(3)
            delete_pod(rp, pod_id)
            cost = elapsed / 3600 * HOURLY_RATE_USD * gpu_count
            print(f"  estimated cost: ${cost:.2f} ({elapsed/60:.1f} min × ${HOURLY_RATE_USD}/hr × {gpu_count} GPU)", flush=True)
            return {"pod_id": pod_id, "cost_usd": cost,
                    "elapsed_s": elapsed, "returncode": rc}
    return {"pod_id": None, "cost_usd": 0.0, "elapsed_s": 0, "returncode": -1}
