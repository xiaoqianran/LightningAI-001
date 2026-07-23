#!/usr/bin/env python3
"""002 — Chinese MoE (S1 smoke / S2 mini).

Usage:
  python run.py prepare --config smoke --force-sample
  python run.py prepare --config mini
  python run.py train --config mini --max-steps 5000
  python run.py generate --config mini
  python run.py job --config mini --machine T4 --interruptible
  python main.py 002 job --config mini --machine T4
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


def _config_path(name: str) -> Path:
    """Resolve configs/smoke.yaml or configs/mini.yaml (also bare path)."""
    p = Path(name)
    if p.suffix in (".yaml", ".yml") and p.is_file():
        return p.resolve()
    cand = EXP / "configs" / f"{name}.yaml"
    if cand.is_file():
        return cand
    raise SystemExit(f"config not found: {name} (tried {cand})")


def cmd_prepare(args: argparse.Namespace) -> int:
    from moe_zh.prepare_d0 import main as prepare_main

    cfg_path = _config_path(args.config)
    argv = ["--config", str(cfg_path)]
    if args.force_sample:
        argv.append("--force-sample")
    if args.max_bytes is not None:
        argv.extend(["--max-bytes", str(args.max_bytes)])
    if args.out:
        argv.extend(["--out", args.out])
    os.chdir(EXP)
    return prepare_main(argv)


def cmd_train(args: argparse.Namespace) -> int:
    from moe_zh.config import load_config
    from moe_zh.train import main as train_main

    cfg_path = _config_path(args.config)
    cfg = load_config(cfg_path)
    art = EXP / cfg.paths.artifact_dir
    os.environ["MOE_ARTIFACT_DIR"] = str(art)
    argv = ["--config", str(cfg_path), "--resume", args.resume]
    if args.max_steps is not None:
        argv.extend(["--max-steps", str(args.max_steps)])
    os.chdir(EXP)
    return train_main(argv)


def cmd_generate(args: argparse.Namespace) -> int:
    from moe_zh.config import load_config
    from moe_zh.generate import main as gen_main

    cfg_path = _config_path(args.config)
    cfg = load_config(cfg_path)
    os.environ["MOE_ARTIFACT_DIR"] = str(EXP / cfg.paths.artifact_dir)
    os.chdir(EXP)
    argv = ["--config", str(cfg_path)]
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

    cfg_name = args.config  # smoke | mini
    cfg_path = _config_path(cfg_name)
    from moe_zh.config import load_config

    cfg = load_config(cfg_path)

    teamspace = os.environ.get("LIGHTNING_TEAMSPACE", "seachenxyt")
    org = os.environ.get("LIGHTNING_ORG")
    user = os.environ.get("LIGHTNING_USER") or os.environ.get("LIGHTNING_USERNAME") or "seachenxyt"
    owner = {"org": org} if org else {"user": user}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    prefix = "moe-zh-s2" if cfg_name in ("mini", "configs/mini.yaml") or str(cfg_path).endswith("mini.yaml") else "moe-zh-s1"
    name = os.environ.get("LIGHTNING_JOB_NAME", f"{prefix}-{stamp}")

    # Defaults: S2 -> T4 (cheapest GPU under $3/h per pricing page)
    default_machine = "T4" if prefix == "moe-zh-s2" else "CPU"
    machine_name = os.environ.get("LIGHTNING_MACHINE", args.machine or default_machine)
    interruptible = (
        os.environ.get("LIGHTNING_INTERRUPTIBLE", "0") == "1"
        or args.interruptible
        or prefix == "moe-zh-s2"
    )
    max_steps = args.max_steps or cfg.train.max_steps
    max_runtime = int(os.environ.get("LIGHTNING_MAX_RUNTIME", str(args.max_runtime or (10800 if prefix == "moe-zh-s2" else 3600))))
    max_bytes = args.max_bytes or cfg.data.max_bytes

    # S2 uses public data by default; S1 smoke defaults force-sample unless disabled
    if args.force_sample:
        force_sample = True
    elif args.no_force_sample:
        force_sample = False
    else:
        force_sample = prefix != "moe-zh-s2"

    prep = "--force-sample" if force_sample else ""
    cloud = os.environ.get("LIGHTNING_CLOUD", args.cloud)
    tarball = os.environ.get(
        "LIGHTNING_TARBALL",
        "https://codeload.github.com/xiaoqianran/LightningAI-Lab/tar.gz/refs/heads/main",
    )
    cfg_file = cfg_path.name  # smoke.yaml / mini.yaml
    art_rel = cfg.paths.artifact_dir
    gen_out = f"{art_rel}/generations_mini.txt" if prefix == "moe-zh-s2" else f"{art_rel}/generations_smoke.txt"

    command = f"""
set -e
cd /tmp
apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null || true
curl -sL {tarball} -o src.tgz
mkdir -p lab && tar -xzf src.tgz -C lab --strip-components=1
cd lab/002
pip install -q torch sentencepiece pyyaml tqdm numpy datasets
export PYTHONPATH=/tmp/lab/002
export MOE_ARTIFACT_DIR=/tmp/lab/002/{art_rel}
python -m moe_zh.prepare_d0 --config configs/{cfg_file} {prep} --max-bytes {max_bytes}
python -m moe_zh.train --config configs/{cfg_file} --resume auto --max-steps {max_steps}
python -m moe_zh.generate --config configs/{cfg_file} --out {gen_out}
python -c "import json;m=json.load(open('{art_rel}/metrics.json'));print('LOSS', m['train_loss'][0],'->',m['train_loss'][-1]);print('VAL', m.get('val_loss'))"
echo ROUTER; cat {art_rel}/router_stats.json
echo GEN; cat {gen_out}
nvidia-smi || true
""".strip()

    machine = getattr(Machine, machine_name) if hasattr(Machine, machine_name) else machine_name

    print("=== LightningAI-Lab / 002 job ===")
    print(f"config={cfg_file} name={name}")
    print(f"machine={machine_name} cloud={cloud or 'default'} interruptible={interruptible}")
    print(f"max_steps={max_steps} max_runtime={max_runtime}s max_bytes={max_bytes} force_sample={force_sample}")

    kw = dict(
        name=name,
        machine=machine,
        image=os.environ.get(
            "LIGHTNING_IMAGE",
            "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime",
        ),
        command=command,
        teamspace=teamspace,
        interruptible=interruptible,
        max_runtime=max_runtime,
        **owner,
    )
    if cloud:
        kw["cloud"] = cloud

    job = Job.run(**kw)
    print("link:", getattr(job, "link", None))
    if not args.no_wait:
        wait_to = max_runtime + 600
        job.wait(interval=20, timeout=wait_to, stop_on_timeout=True)
        print("status:", job.status, "cost:", getattr(job, "total_cost", None))
        try:
            logs = job.logs or ""
            print(logs[-10000:] if len(logs) > 10000 else logs)
        except Exception as exc:  # noqa: BLE001
            print("logs:", exc, file=sys.stderr)
        return 0 if "Completed" in str(job.status) else 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="002 Chinese MoE (smoke / mini)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_config(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--config",
            default="smoke",
            help="smoke | mini | path/to.yaml (default: smoke)",
        )

    sp = sub.add_parser("prepare", help="build data + tokenizer")
    add_config(sp)
    sp.add_argument("--force-sample", action="store_true")
    sp.add_argument("--max-bytes", type=int, default=None)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_prepare)

    st = sub.add_parser("train", help="train MoE")
    add_config(st)
    st.add_argument("--max-steps", type=int, default=None)
    st.add_argument("--resume", default="auto")
    st.set_defaults(func=cmd_train)

    sg = sub.add_parser("generate", help="generate from last.pt")
    add_config(sg)
    sg.add_argument("--ckpt", default=None)
    sg.add_argument("--out", default=None)
    sg.set_defaults(func=cmd_generate)

    sj = sub.add_parser("job", help="submit remote Lightning Job")
    add_config(sj)
    sj.add_argument("--machine", default=None, help="default: T4 for mini, CPU for smoke")
    sj.add_argument("--cloud", default=None, help="AWS | GCP | ... optional")
    sj.add_argument("--max-steps", type=int, default=None)
    sj.add_argument("--max-runtime", type=int, default=None, help="seconds")
    sj.add_argument("--max-bytes", type=int, default=None)
    sj.add_argument("--force-sample", action="store_true", help="use sample corpus")
    sj.add_argument("--no-force-sample", action="store_true", help="prefer public wiki")
    sj.add_argument("--interruptible", action="store_true")
    sj.add_argument("--no-wait", action="store_true")
    sj.set_defaults(func=cmd_job)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
