#!/usr/bin/env python3
"""Submit S1 smoke training as a Lightning Job (GPU when available).

Requires lightning-sdk and credentials. Repo must be available in the image
or cloned in the command. This script submits a self-contained command that:
  1) clones or uses baked path
  2) installs deps
  3) prepare_d0 + train + generate

For restricted envs without GPU, use scripts/s1_smoke_local.sh instead.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

try:
    from lightning_sdk import Job, Machine
except ImportError:
    print("lightning-sdk not installed. Local test: bash scripts/s1_smoke_local.sh", file=sys.stderr)
    raise SystemExit(2)

# Prefer cheap allowed GPUs; override with LIGHTNING_MACHINE
_DEFAULT_MACHINE = os.environ.get("LIGHTNING_MACHINE", "T4")


def _owner_kwargs() -> dict:
    org = os.environ.get("LIGHTNING_ORG")
    user = os.environ.get("LIGHTNING_USER") or os.environ.get("LIGHTNING_USERNAME")
    if org:
        return {"org": org}
    return {"user": user or "seachenxyt"}


def main() -> int:
    teamspace = os.environ.get("LIGHTNING_TEAMSPACE", "seachenxyt")
    owner = _owner_kwargs()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = os.environ.get("LIGHTNING_JOB_NAME", f"moe-zh-s1-{stamp}")
    machine = os.environ.get("LIGHTNING_MACHINE", _DEFAULT_MACHINE)
    interruptible = os.environ.get("LIGHTNING_INTERRUPTIBLE", "1") == "1"
    repo = os.environ.get(
        "LIGHTNING_REPO_URL",
        "https://github.com/xiaoqianran/LightningAI-001.git",
    )
    max_steps = os.environ.get("MOE_MAX_STEPS", "2000")
    force_sample = os.environ.get("MOE_FORCE_SAMPLE", "0") == "1"
    prep_flag = "--force-sample" if force_sample else ""

    # PyTorch CUDA image; install repo deps then run pipeline
    command = f"""
set -e
cd /tmp
git clone --depth 1 {repo} app || true
cd app
pip install -q -r requirements-train.txt
export PYTHONPATH=/tmp/app
export MOE_ARTIFACT_DIR=/tmp/app/artifacts
python -m moe_zh.prepare_d0 {prep_flag} --max-bytes 30000000
python -m moe_zh.train --resume auto --max-steps {max_steps}
python -m moe_zh.generate
echo '=== metrics ==='
cat artifacts/metrics.json | head -c 2000 || true
echo
echo '=== generations ==='
cat artifacts/generations_smoke.txt || true
""".strip()

    print("submitting", name, "machine", machine, "interruptible", interruptible)
    job = Job.run(
        name=name,
        machine=machine,
        image=os.environ.get(
            "LIGHTNING_IMAGE",
            "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime",
        ),
        command=command,
        teamspace=teamspace,
        interruptible=interruptible,
        max_runtime=int(os.environ.get("LIGHTNING_MAX_RUNTIME", "3600")),
        **owner,
    )
    print("link:", getattr(job, "link", None))
    job.wait(interval=15, timeout=60 * 60, stop_on_timeout=True)
    print("status:", job.status)
    try:
        print(job.logs)
    except Exception as exc:  # noqa: BLE001
        print("logs error:", exc, file=sys.stderr)
    return 0 if str(job.status).endswith("Completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
