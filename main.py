"""Workspace entry — dispatches to experiment run.py (cuda-lab style).

Usage:
  python main.py 001 run
  python main.py 002 prepare --force-sample
  python main.py 002 train --max-steps 200
  python main.py 002 job

If the first arg is not an experiment id, defaults to 001.
Uses each experiment's .venv when present.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_EXP = "001"


def venv_python(exp_dir: Path) -> Path | None:
    if sys.platform == "win32":
        p = exp_dir / ".venv" / "Scripts" / "python.exe"
    else:
        p = exp_dir / ".venv" / "bin" / "python"
    return p if p.is_file() else None


def resolve_experiment(argv: list[str]) -> tuple[Path, list[str]]:
    if not argv:
        return ROOT / DEFAULT_EXP / "run.py", []
    first = argv[0]
    cand = ROOT / first / "run.py"
    if cand.is_file():
        return cand, argv[1:]
    return ROOT / DEFAULT_EXP / "run.py", argv


def main() -> None:
    raw = sys.argv[1:]
    run_py, rest = resolve_experiment(raw)
    if not run_py.is_file():
        raise SystemExit(f"experiment script not found: {run_py}")

    exp_dir = run_py.parent
    vpy = venv_python(exp_dir)
    if vpy is not None and Path(sys.executable).resolve() != vpy.resolve():
        exp_id = exp_dir.name
        if raw and raw[0] == exp_id:
            new_argv = [str(vpy), str(ROOT / "main.py"), exp_id, *rest]
        else:
            new_argv = [str(vpy), str(ROOT / "main.py"), exp_id, *rest]
        os.execv(str(vpy), new_argv)

    sys.argv = [str(run_py), *rest]
    runpy.run_path(str(run_py), run_name="__main__")


if __name__ == "__main__":
    main()
