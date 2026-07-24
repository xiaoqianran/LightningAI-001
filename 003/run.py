#!/usr/bin/env python3
"""003 — NVIDIA NIM agent entry.

Usage:
  export NVIDIA_API_KEY=nvapi-...
  python run.py ping
  python run.py run --task "列出 002 目录"
  python main.py 003 run --task "..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="003 NVIDIA NIM Agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("ping", help="test NIM API connectivity")
    sp.set_defaults(func=lambda a: _ping())

    sr = sub.add_parser("run", help="run agent on a task")
    sr.add_argument("--task", required=True, help="user task in natural language")
    sr.add_argument("--max-rounds", type=int, default=6)
    sr.set_defaults(func=lambda a: _run(a.task, a.max_rounds))

    sc = sub.add_parser("chat", help="single-turn chat without tools")
    sc.add_argument("--message", required=True)
    sc.set_defaults(func=lambda a: _chat(a.message))

    args = p.parse_args(argv)
    return int(args.func(args))


def _ping() -> int:
    from agent import ping

    print(ping())
    return 0


def _run(task: str, max_rounds: int) -> int:
    from agent import run_agent

    print(f"=== 003 NIM agent ===\ntask: {task}\n")
    answer = run_agent(task, max_rounds=max_rounds)
    print("\n=== final ===\n")
    print(answer)
    return 0


def _chat(message: str) -> int:
    from nim_client import chat_text

    print(chat_text(message))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
