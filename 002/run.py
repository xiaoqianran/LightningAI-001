#!/usr/bin/env python3
"""002 — Chinese MoE smoke (prepare / train / generate / remote job).

Usage:
  python run.py prepare --force-sample
  python run.py train --max-steps 200
  python run.py generate
  python run.py job
  python main.py 002 train --max-steps 50
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))


def cmd_prepare(args: argparse.Namespace) -> int:
    from moe_zh.prepare_d0 import main as prepare_main

    argv = ["--config", str(EXP / "configs" / "smoke.yaml")]
    if args.force_sample:
        argv.append("--force-sample")
    if args.max_bytes is not None:
        argv.extend(["--max-bytes", str(args.max_bytes)])
    if args.out:
        argv.extend(["--out", args.out])
    return prepare_main(argv)


def cmd_train(args: argparse.Namespace) -> int:
    from moe_zh.train import main as train_main

    argv = [
        "--config",
        str(EXP / "configs" / "smoke.yaml"),
        "--resume",
        args.resume,
    ]
    if args.max_steps is not None:
        argv.extend(["--max-steps", str(args.max_steps)])
    os.environ.setdefault("MOE_ARTIFACT_DIR", str(EXP / "artifacts"))
    # ensure data path relative to 002
    os.chdir(EXP)
    return train_main(argv)


def cmd_generate(args: argparse.Namespace) -> int:
    from moe_zh.generate import main as gen_main

    os.environ.setdefault("MOE_ARTIFACT_DIR", str(EXP / "artifacts"))
    os.chdir(EXP)
    argv = ["--config", str(EXP / "configs" / "smoke.yaml")]
    if args.ckpt:
        argv.extend(["--ckpt", args.ckpt])
    if args.out:
        argv.extend(["--out", args.out])
    return gen_main(argv)


def cmd_job(args: argparse.Namespace) -> int:
    try:
        from lightning_sdk import Job, Machine
    except ImportError:
        print("pip install lightning-sdk", file=sys.stderr)
        return 2

    teamspace = os.environ.get("LIGHTNING_TEAMSPACE", "seachenxyt")
    org = os.environ.get("LIGHTNING_ORG")
    user = os.environ.get("LIGHTNING_USER") or os.environ.get("LIGHTNING_USERNAME") or "seachenxyt"
    owner = {"org": org} if org else {"user": user}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = os.environ.get("LIGHTNING_JOB_NAME", f"moe-zh-s1-{stamp}")
    machine_name = os.environ.get("LIGHTNING_MACHINE", args.machine)
    interruptible = os.environ.get("LIGHTNING_INTERRUPTIBLE", "0") == "1" or args.interruptible
    max_steps = args.max_steps or int(os.environ.get("MOE_MAX_STEPS", "200"))
    force_sample = args.force_sample or os.environ.get("MOE_FORCE_SAMPLE", "1") == "1"
    prep = "--force-sample" if force_sample else ""

    tarball = os.environ.get(
        "LIGHTNING_TARBALL",
        "https://codeload.github.com/xiaoqianran/LightningAI-Lab/tar.gz/refs/heads/main",
    )

    # Remote layout: repo root has 002/
    command = f"""
set -e
cd /tmp
apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null || true
curl -sL {tarball} -o src.tgz
mkdir -p lab && tar -xzf src.tgz -C lab --strip-components=1
cd lab/002
pip install -q torch sentencepiece pyyaml tqdm numpy
export PYTHONPATH=/tmp/lab/002
export MOE_ARTIFACT_DIR=/tmp/lab/002/artifacts
python -m moe_zh.prepare_d0 {prep} --max-bytes 2000000
python -m moe_zh.train --resume auto --max-steps {max_steps}
python -m moe_zh.generate
python -c "import json;m=json.load(open('artifacts/metrics.json'));print('LOSS', m['train_loss'][0],'->',m['train_loss'][-1])"
cat artifacts/router_stats.json
cat artifacts/generations_smoke.txt
""".strip()

    machine = getattr(Machine, machine_name) if hasattr(Machine, machine_name) else machine_name

    print("=== LightningAI-Lab / 002 job ===")
    print(f"name={name} machine={machine_name} interruptible={interruptible}")
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
    if not args.no_wait:
        job.wait(interval=15, timeout=55 * 60, stop_on_timeout=True)
        print("status:", job.status, "cost:", getattr(job, "total_cost", None))
        try:
            logs = job.logs or ""
            print(logs[-8000:] if len(logs) > 8000 else logs)
        except Exception as exc:  # noqa: BLE001
            print("logs:", exc, file=sys.stderr)
        return 0 if "Completed" in str(job.status) else 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="002 Chinese MoE smoke")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare", help="build D0 data + tokenizer")
    sp.add_argument("--force-sample", action="store_true")
    sp.add_argument("--max-bytes", type=int, default=None)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_prepare)

    st = sub.add_parser("train", help="train MoE")
    st.add_argument("--max-steps", type=int, default=None)
    st.add_argument("--resume", default="auto")
    st.set_defaults(func=cmd_train)

    sg = sub.add_parser("generate", help="generate from last.pt")
    sg.add_argument("--ckpt", default=None)
    sg.add_argument("--out", default=None)
    sg.set_defaults(func=cmd_generate)

    sj = sub.add_parser("job", help="submit remote Lightning Job")
    sj.add_argument("--machine", default="CPU", help="CPU / L4 / A100 / ...")
    sj.add_argument("--max-steps", type=int, default=None)
    sj.add_argument("--force-sample", action="store_true", default=True)
    sj.add_argument("--interruptible", action="store_true")
    sj.add_argument("--no-wait", action="store_true")
    sj.set_defaults(func=cmd_job)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
