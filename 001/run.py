#!/usr/bin/env python3
"""001 — Hello Lightning Job (CPU Docker smoke).

Usage:
  python run.py run
  python main.py 001 run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone


def _owner_kwargs() -> dict:
    org = os.environ.get("LIGHTNING_ORG")
    user = os.environ.get("LIGHTNING_USER") or os.environ.get("LIGHTNING_USERNAME")
    if org and user:
        raise SystemExit("Set only one of LIGHTNING_ORG or LIGHTNING_USER, not both.")
    if org:
        return {"org": org}
    return {"user": user or "seachenxyt"}


def cmd_run() -> int:
    try:
        from lightning_sdk import Job, Machine, Status
    except ImportError:
        print("pip install lightning-sdk", file=sys.stderr)
        return 2

    teamspace = os.environ.get("LIGHTNING_TEAMSPACE", "seachenxyt")
    owner = _owner_kwargs()
    owner_label = owner.get("org") or owner.get("user")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = os.environ.get("LIGHTNING_JOB_NAME", f"hello-cpu-{stamp}")

    print("=== LightningAI-Lab / 001 hello job ===")
    print(f"teamspace : {owner_label}/{teamspace}")
    print(f"job name  : {name}")
    print()

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
    print("waiting...")
    try:
        job.wait(interval=10, timeout=30 * 60, stop_on_timeout=True)
    except TypeError:
        deadline = time.time() + 30 * 60
        while str(job.status) in ("Status.Pending", "Status.Running", "Pending", "Running"):
            if time.time() > deadline:
                job.stop()
                break
            time.sleep(10)

    status = job.status
    print(f"status    : {status}")
    if getattr(job, "total_cost", None) is not None:
        try:
            print(f"cost      : ${float(job.total_cost):.4f}")
        except (TypeError, ValueError):
            print(f"cost      : {job.total_cost}")

    if any(x in str(status) for x in ("Completed", "Failed", "Stopped")):
        try:
            print("--- logs ---")
            print(job.logs)
        except Exception as exc:  # noqa: BLE001
            print(f"(logs: {exc})", file=sys.stderr)

    return 0 if "Completed" in str(status) else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="001 Hello Lightning Job")
    p.add_argument(
        "cmd",
        nargs="?",
        default="run",
        choices=["run"],
        help="run = submit remote job (default)",
    )
    args = p.parse_args(argv)
    if args.cmd == "run":
        return cmd_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
