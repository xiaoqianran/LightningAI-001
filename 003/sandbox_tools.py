"""Execute commands inside a remote Lightning Sandbox via CLI."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any


def _require_key() -> None:
    if not os.environ.get("LIGHTNING_SANDBOX_API_KEY"):
        raise SystemExit(
            "Set LIGHTNING_SANDBOX_API_KEY to an org/teamspace-scoped key "
            "(lightning api-key create --org seachenxyt-org --name sandbox-lab). "
            "Personal LIGHTNING_API_KEY cannot create sandboxes."
        )


def sandbox_create(
    name: str = "lab-agent",
    *,
    teamspace: str = "seachenxyt/seachenxyt",
    runtime: str = "python313",
    timeout_ms: int = 900_000,
) -> str:
    """Create sandbox; return sandbox id."""
    _require_key()
    cmd = [
        "lightning",
        "sandbox",
        "create",
        "--name",
        name,
        "--teamspace",
        teamspace,
        "--runtime",
        runtime,
        "--timeout",
        str(timeout_ms),
        "--json",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout or "sandbox create failed")
    data = json.loads(r.stdout)
    return data["id"]


def sandbox_run(sandbox_id: str, command: str, *, timeout: int = 120) -> str:
    """Run a shell command in the sandbox; return combined output."""
    _require_key()
    cmd = [
        "lightning",
        "sandbox",
        "run",
        sandbox_id,
        "--",
        "bash",
        "-lc",
        command,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    return f"exit={r.returncode}\n{out[:12000]}"


def sandbox_delete(sandbox_id: str) -> str:
    _require_key()
    r = subprocess.run(
        ["lightning", "sandbox", "delete", sandbox_id],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return (r.stdout or r.stderr or "")[:2000]


def probe_limits(sandbox_id: str) -> dict[str, Any]:
    """Run a fixed suite of probes; return structured results."""
    probes = {
        "python": "python --version",
        "os": "uname -a; whoami; nproc; df -h / | tail -1",
        "mem": "python -c \"import os; print('pagesize', os.sysconf('SC_PAGE_SIZE')); print('phys', os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')//1024//1024, 'MB')\"",
        "network": "python -c \"import urllib.request; print(urllib.request.urlopen('https://example.com', timeout=15).status)\"",
        "pip": "pip install -q six && python -c 'import six; print(six.__version__)'",
        "write_run": "printf 'print(40+2)\\n' > /workspace/probe.py && python /workspace/probe.py",
        "gpu": "nvidia-smi 2>&1 | head -5 || echo NO_GPU",
        "disk_write": "dd if=/dev/zero of=/workspace/tmp.bin bs=1M count=50 2>&1 | tail -2; rm -f /workspace/tmp.bin",
    }
    results = {}
    for k, c in probes.items():
        try:
            results[k] = sandbox_run(sandbox_id, c, timeout=180)
        except Exception as e:  # noqa: BLE001
            results[k] = f"ERROR: {e}"
    return results
