"""Whitelisted tools the NIM agent may call."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable

LAB_ROOT = Path(__file__).resolve().parents[1]

# Only these shell prefixes are allowed (space-separated start).
_SHELL_ALLOW = (
    "ls",
    "pwd",
    "date",
    "uname",
    "python --version",
    "python3 --version",
    "git status",
    "git log",
    "git branch",
    "git remote",
    "head ",
    "wc ",
    "find ",
    "cat ",
    "echo ",
)


def _safe_rel(path: str) -> Path:
    # normalize absolute-looking paths to repo-relative
    path = (path or ".").strip() or "."
    if path in ("/", "\\"):
        path = "."
    p = (LAB_ROOT / path).resolve()
    root = LAB_ROOT.resolve()
    if not str(p).startswith(str(root)):
        raise ValueError(f"path escapes lab root: {path}")
    return p


def tool_list_dir(path: str = ".") -> str:
    path = path or "."
    p = _safe_rel(path)
    if not p.is_dir():
        return f"not a directory: {path}"
    lines = []
    for child in sorted(p.iterdir()):
        kind = "dir" if child.is_dir() else "file"
        lines.append(f"{kind}\t{child.relative_to(LAB_ROOT)}")
    return "\n".join(lines) if lines else "(empty)"


def tool_read_file(path: str, max_chars: int = 4000) -> str:
    p = _safe_rel(path)
    if not p.is_file():
        return f"not a file: {path}"
    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"
    return text


def tool_run_shell(command: str) -> str:
    cmd = command.strip()
    if not cmd:
        return "empty command"
    allowed = False
    for prefix in _SHELL_ALLOW:
        if cmd == prefix.strip() or cmd.startswith(prefix):
            allowed = True
            break
    if not allowed:
        return (
            "DENIED: command not in whitelist. "
            f"Allowed prefixes: {_SHELL_ALLOW}"
        )
    try:
        r = subprocess.run(
            cmd,
            shell=True,
            cwd=str(LAB_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = (r.stdout or "") + (r.stderr or "")
        return f"exit={r.returncode}\n{out[:6000]}"
    except subprocess.TimeoutExpired:
        return "TIMEOUT after 30s"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def tool_lab_help() -> str:
    return """LightningAI-Lab quick help:
- 001: python main.py 001 run  → Hello Lightning Job
- 002 smoke: python main.py 002 prepare --config smoke --force-sample
- 002 S2: python main.py 002 job --config mini --machine T4
- 003: this NIM agent (export NVIDIA_API_KEY first)
- Sandbox: needs LIGHTNING_SANDBOX_API_KEY (org key seachenxyt-org); remote isolated VM
Jobs always on-demand (non-interruptible). S2 default GPU: T4.
"""


def tool_sandbox_run(command: str) -> str:
    """Run command in existing sandbox (LIGHTNING_SANDBOX_ID)."""
    import os

    sid = os.environ.get("LIGHTNING_SANDBOX_ID", "").strip()
    if not sid:
        return (
            "No LIGHTNING_SANDBOX_ID set. Create one first: "
            "python main.py 003 sandbox-create"
        )
    try:
        from sandbox_tools import sandbox_run

        return sandbox_run(sid, command)
    except Exception as e:  # noqa: BLE001
        return f"sandbox error: {e}"


TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files under LightningAI-Lab relative path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path, default .",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file under the lab repo",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a whitelisted shell command in the lab root",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lab_help",
            "description": "Summarize how to use LightningAI-Lab experiments",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sandbox_run",
            "description": (
                "Run a bash command inside the remote Lightning Sandbox "
                "(isolated cloud VM). Use for untrusted/experimental code."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


_HANDLERS: dict[str, Callable[..., str]] = {
    "list_dir": lambda **kw: tool_list_dir(kw.get("path") or "."),
    "read_file": lambda **kw: tool_read_file(
        kw["path"], int(kw.get("max_chars") or 4000)
    ),
    "run_shell": lambda **kw: tool_run_shell(kw["command"]),
    "lab_help": lambda **kw: tool_lab_help(),
    "sandbox_run": lambda **kw: tool_sandbox_run(kw["command"]),
}


def dispatch(name: str, arguments: str | dict) -> str:
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            args = {}
    else:
        args = arguments or {}
    fn = _HANDLERS.get(name)
    if not fn:
        return f"unknown tool: {name}"
    try:
        return fn(**args)
    except Exception as e:  # noqa: BLE001
        return f"tool error: {e}"
