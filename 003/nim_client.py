"""Minimal NVIDIA NIM OpenAI-compatible client (stdlib urllib only)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BASE = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"


def _api_key() -> str:
    key = (
        os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("NIM_API_KEY")
        or os.environ.get("NGC_API_KEY")
        or ""
    ).strip()
    if not key:
        raise SystemExit(
            "Set NVIDIA_API_KEY (or NIM_API_KEY) in the environment. "
            "Do not commit keys to git."
        )
    return key


def _base_url() -> str:
    return os.environ.get("NIM_BASE_URL", DEFAULT_BASE).rstrip("/")


def model_name() -> str:
    return os.environ.get("NIM_MODEL", DEFAULT_MODEL)


def _request(method: str, path: str, body: dict | None = None) -> dict[str, Any]:
    url = f"{_base_url()}{path}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NIM HTTP {e.code}: {err[:1200]}") from e


def list_models() -> list[str]:
    data = _request("GET", "/models")
    return [m.get("id", "") for m in data.get("data", []) if m.get("id")]


def chat(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = "auto",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Return choices[0].message from chat completions."""
    body: dict[str, Any] = {
        "model": model_name(),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
    data = _request("POST", "/chat/completions", body)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"empty choices from NIM: {str(data)[:500]}")
    return choices[0].get("message") or {}


def chat_text(user: str, system: str | None = None) -> str:
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    msg = chat(messages, tools=None, tool_choice=None)
    return (msg.get("content") or "").strip()
