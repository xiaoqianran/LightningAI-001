#!/usr/bin/env python3
"""01 — Basic Lightning AI batch job (hello world).

Submits a tiny CPU Docker job, waits for completion, prints logs.

Auth (one of):
  lightning login
  export LIGHTNING_USER_ID=... LIGHTNING_API_KEY=...

Optional overrides:
  LIGHTNING_ORG / LIGHTNING_USER   owner of the teamspace (default: seachenxyt)
  LIGHTNING_TEAMSPACE              teamspace name (default: seachenxyt)
  LIGHTNING_JOB_NAME               job name (default: hello-cpu-<timestamp>)
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

from lightning_sdk import Job, Machine, Status


def _owner_kwargs() -> dict:
    """Resolve teamspace owner as org= or user= (mutually exclusive)."""
    org = os.environ.get("LIGHTNING_ORG")
    user = os.environ.get("LIGHTNING_USER") or os.environ.get("LIGHTNING_USERNAME")
    if org and user:
        raise SystemExit("Set only one of LIGHTNING_ORG or LIGHTNING_USER, not both.")
    if org:
        return {"org": org}
    # Default: personal teamspace owner for this device login
    return {"user": user or "seachenxyt"}


def main() -> int:
    teamspace = os.environ.get("LIGHTNING_TEAMSPACE", "seachenxyt")
    owner = _owner_kwargs()
    owner_label = owner.get("org") or owner.get("user")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = os.environ.get("LIGHTNING_JOB_NAME", f"hello-cpu-{stamp}")

    print("=== LightningAI-001 / 01 hello job ===")
    print(f"teamspace : {owner_label}/{teamspace}")
    print(f"job name  : {name}")
    print(f"machine   : CPU")
    print(f"image     : python:3.11-slim")
    print()

    # Headless-friendly: pass teamspace name + owner separately (not "owner/name").
    job = Job.run(
        name=name,
        machine=Machine.CPU,
        image="python:3.11-slim",
        command=(
            'python -c "'
            "import platform, sys; "
            "print('Hello from Lightning Job!'); "
            "print('python', sys.version.split()[0]); "
            "print('platform', platform.platform()); "
            '"'
        ),
        teamspace=teamspace,
        interruptible=False,
        **owner,
    )

    print(f"submitted : {job.name}")
    print(f"link      : {getattr(job, 'link', None)}")
    print("waiting for terminal status...")

    # wait() blocks until Completed/Failed/Stopped (or timeout)
    try:
        job.wait(interval=10, timeout=30 * 60, stop_on_timeout=True)
    except TypeError:
        # older SDK: poll manually
        deadline = time.time() + 30 * 60
        while str(job.status) in ("Status.Pending", "Status.Running", "Pending", "Running"):
            if time.time() > deadline:
                print("timeout — stopping job", file=sys.stderr)
                job.stop()
                break
            print(f"  status={job.status}")
            time.sleep(10)

    status = job.status
    print()
    print(f"status    : {status}")
    if hasattr(job, "total_cost") and job.total_cost is not None:
        try:
            print(f"cost      : ${float(job.total_cost):.4f}")
        except (TypeError, ValueError):
            print(f"cost      : {job.total_cost}")

    # logs only available in terminal states
    terminal = {
        Status.Completed,
        Status.Failed,
        Status.Stopped,
        "Completed",
        "Failed",
        "Stopped",
        "Status.Completed",
        "Status.Failed",
        "Status.Stopped",
    }
    if status in terminal or str(status).split(".")[-1] in ("Completed", "Failed", "Stopped"):
        try:
            print("--- logs ---")
            print(job.logs)
            print("--- end logs ---")
        except Exception as exc:  # noqa: BLE001
            print(f"(could not fetch logs: {exc})", file=sys.stderr)
    else:
        print("(job not terminal yet — skip logs)", file=sys.stderr)

    ok = status == Status.Completed or str(status).endswith("Completed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
