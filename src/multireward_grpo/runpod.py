"""
Minimal RunPod client so anyone with a RunPod API key can run multi-reward GRPO
on a cloud H100 without hand-writing the pod lifecycle.

:class:`RunPodClient` wraps the RunPod REST API: spin up a GPU pod, wait for SSH,
run a remote bash command, then tear the pod down (billing stops on delete).
``requests`` is the only hard dependency; SSH/rsync use the system ``ssh``/
``rsync`` binaries and an SSH key you already have.

Example
-------
>>> from multireward_grpo.runpod import RunPodClient
>>> client = RunPodClient()  # reads RUNPOD_API_KEY from env or .env
>>> client.run_command(
...     "pip install -q multireward-grpo[llm] && "
...     "python -c 'import multireward_grpo as m; print(m.__version__)'",
...     wall_clock_cap=900,
... )
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

API_BASE = "https://rest.runpod.io/v1"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
H100_TYPE_PREFERENCES = (
    "NVIDIA H100 PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 NVL",
)
HOURLY_RATE_USD = 2.39  # H100 PCIe Secure Cloud (adjust if pricing changes)


def load_api_key(explicit: Optional[str] = None, env_path: str = ".env") -> str:
    """Resolve a RunPod API key from (1) ``explicit``, (2) ``RUNPOD_API_KEY`` env,
    (3) a ``.env`` file in the working directory."""
    if explicit:
        return explicit
    if key := os.environ.get("RUNPOD_API_KEY"):
        return key
    p = Path(env_path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line.startswith("RUNPOD_API_KEY=") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(
        "no RunPod API key found (pass api_key=, set RUNPOD_API_KEY, or add it to .env)"
    )


@dataclass
class PodSpec:
    name: str = "multireward-grpo"
    gpu_count: int = 1
    container_disk_gb: int = 60
    volume_gb: int = 20
    image: str = DEFAULT_IMAGE
    extra_env: dict = field(default_factory=dict)


def _docker_start_cmd() -> list[str]:
    return [
        "bash", "-c",
        (
            "set -e; apt-get update -qq && apt-get install -y -qq openssh-server rsync; "
            "mkdir -p /root/.ssh && chmod 700 /root/.ssh; "
            'echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys; '
            "chmod 600 /root/.ssh/authorized_keys; "
            "sed -i 's/^#\\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config; "
            "sed -i 's/^#\\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config; "
            "mkdir -p /run/sshd; /usr/sbin/sshd -D"
        ),
    ]


class RunPodClient:
    """Thin client over the RunPod REST API + SSH command execution."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        ssh_key_path: Optional[str] = None,
        verbose: bool = True,
    ):
        self.api_key = load_api_key(api_key)
        self.verbose = verbose
        self.ssh_key_path = ssh_key_path or str(Path.home() / ".ssh" / "id_ed25519")
        self._s = requests.Session()
        self._s.headers.update(
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        )

    # -- REST helpers --------------------------------------------------------
    def _get(self, path, **kw):
        r = self._s.get(f"{API_BASE}{path}", timeout=30, **kw)
        if not r.ok:
            raise RuntimeError(f"GET {path}: {r.status_code} {r.text[:400]}")
        return r.json()

    def _post(self, path, **kw):
        r = self._s.post(f"{API_BASE}{path}", timeout=60, **kw)
        if not r.ok:
            raise RuntimeError(f"POST {path}: {r.status_code} {r.text[:400]}")
        return r.json() if r.text else {}

    def _delete(self, path, **kw):
        r = self._s.delete(f"{API_BASE}{path}", timeout=30, **kw)
        if not r.ok and r.status_code != 404:
            raise RuntimeError(f"DELETE {path}: {r.status_code} {r.text[:400]}")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # -- pod lifecycle -------------------------------------------------------
    def create_pod(self, spec: PodSpec, gpu_types: Optional[tuple[str, ...]] = None,
                   hf_token: Optional[str] = None) -> str:
        """Create a GPU pod, trying each GPU type until one is accepted. Returns the pod id."""
        pub_key = Path(self.ssh_key_path + ".pub").read_text().strip()
        env = {"PUBLIC_KEY": pub_key, "HF_HUB_DISABLE_TELEMETRY": "1",
               "TOKENIZERS_PARALLELISM": "false", **spec.extra_env}
        if hf_token:
            env["HF_TOKEN"] = hf_token
            env["HUGGINGFACE_HUB_TOKEN"] = hf_token
        last_err = None
        for gt in (gpu_types or H100_TYPE_PREFERENCES):
            body = {
                "name": spec.name, "imageName": spec.image, "gpuTypeIds": [gt],
                "gpuCount": spec.gpu_count, "cloudType": "SECURE", "computeType": "GPU",
                "ports": ["22/tcp"], "containerDiskInGb": spec.container_disk_gb,
                "volumeInGb": spec.volume_gb, "volumeMountPath": "/workspace",
                "env": env, "dockerStartCmd": _docker_start_cmd(),
            }
            try:
                created = self._post("/pods", json=body)
                self._log(f"=== created pod (gpuType={gt}) ===")
                return created.get("id") or created.get("podId") or created.get("pod", {}).get("id")
            except RuntimeError as e:
                last_err = e
                self._log(f"  gpuType={gt} rejected: {str(e)[:150]}")
        raise RuntimeError(f"no GPU type accepted: {last_err}")

    def wait_for_ready(self, pod_id: str, timeout_s: int = 600) -> dict:
        start = time.time()
        while time.time() - start < timeout_s:
            p = self._get(f"/pods/{pod_id}")
            ports = p.get("portMappings") or {}
            if p.get("desiredStatus") == "RUNNING" and ports.get("22") and p.get("publicIp"):
                return p
            time.sleep(8)
        raise TimeoutError(f"pod {pod_id} not RUNNING within {timeout_s}s")

    def delete_pod(self, pod_id: str) -> None:
        self._log(f"  deleting pod {pod_id} ...")
        try:
            self._post(f"/pods/{pod_id}/stop")
            time.sleep(3)
        except Exception as e:
            self._log(f"    stop failed (ignored): {e}")
        try:
            self._delete(f"/pods/{pod_id}")
        except Exception as e:
            self._log(f"    delete failed (ignored): {e}")

    # -- SSH -----------------------------------------------------------------
    def _ssh_opts(self) -> str:
        return (
            "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            f"-o ServerAliveInterval=30 -o ConnectTimeout=20 -i {self.ssh_key_path}"
        )

    def wait_for_ssh(self, host: str, port: int, timeout_s: int = 360) -> None:
        start = time.time()
        while time.time() - start < timeout_s:
            rc = subprocess.call(
                f"ssh {self._ssh_opts()} -p {port} root@{host} 'echo ok' >/dev/null 2>&1",
                shell=True, timeout=20,
            )
            if rc == 0:
                self._log(f"  ssh ready after {int(time.time()-start)}s")
                return
            time.sleep(8)
        raise TimeoutError(f"ssh not ready within {timeout_s}s")

    def ssh_run(self, host: str, port: int, command: str, timeout_s: int,
                env_vars: Optional[dict] = None) -> int:
        parts = []
        for k, v in (env_vars or {}).items():
            if v is None:
                continue
            esc = str(v).replace("'", "'\\''")
            parts.append(f"export {k}='{esc}'")
        parts.append(command)
        full = "\n".join(parts)
        cmd = f"ssh {self._ssh_opts()} -p {port} root@{host} bash -s"
        return subprocess.run(cmd, shell=True, input=full.encode(), timeout=timeout_s).returncode

    # -- one-shot ------------------------------------------------------------
    def run_command(
        self,
        command: str,
        spec: Optional[PodSpec] = None,
        wall_clock_cap: int = 1800,
        gpu_types: Optional[tuple[str, ...]] = None,
        hf_token: Optional[str] = None,
        env_vars: Optional[dict] = None,
    ) -> dict:
        """Spin up a pod, run ``command`` over SSH, then tear the pod down.

        Returns ``{pod_id, returncode, elapsed_s, cost_usd}``. The pod is always
        deleted, even on error.
        """
        spec = spec or PodSpec()
        result = {"pod_id": None, "returncode": -1, "elapsed_s": 0.0, "cost_usd": 0.0}
        t0 = None
        try:
            result["pod_id"] = self.create_pod(spec, gpu_types=gpu_types, hf_token=hf_token)
            t0 = time.time()
            self._log(f"  pod {result['pod_id']} (billing starts NOW)")
            p = self.wait_for_ready(result["pod_id"])
            host, port = p["publicIp"], int(p["portMappings"]["22"])
            self._log(f"  RUNNING on {host}:{port}")
            self.wait_for_ssh(host, port)
            remaining = max(60, wall_clock_cap - int(time.time() - t0) - 60)
            result["returncode"] = self.ssh_run(
                host, port, command, timeout_s=remaining, env_vars=env_vars
            )
        finally:
            if result["pod_id"] and t0:
                result["elapsed_s"] = time.time() - t0
                self.delete_pod(result["pod_id"])
                result["cost_usd"] = result["elapsed_s"] / 3600 * HOURLY_RATE_USD * spec.gpu_count
                self._log(f"  pod lived {result['elapsed_s']/60:.1f} min  ~${result['cost_usd']:.2f}")
        return result


__all__ = ["RunPodClient", "PodSpec", "load_api_key"]
