"""Simple ReAct / tool-calling agent over NVIDIA NIM."""

from __future__ import annotations

import json
from typing import Any

from nim_client import chat, chat_text, list_models, model_name
from tools import TOOLS_SPEC, dispatch

SYSTEM = """You are a helpful coding agent for the LightningAI-Lab repository.
You may call tools to inspect the repo and run safe whitelisted shell commands.
IMPORTANT: Call at most ONE tool per turn (this model rejects multi tool-calls).
Prefer tools over guessing file contents.
When done, give a clear final answer in Chinese if the user writes Chinese.
Do not invent API keys or claim jobs finished without tool evidence.
"""


def _parse_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    tcs = message.get("tool_calls") or []
    if tcs:
        return tcs
    # Some models put a single function_call
    fc = message.get("function_call")
    if fc:
        return [{"id": "call_0", "type": "function", "function": fc}]
    return []


def run_agent(task: str, max_rounds: int = 6) -> str:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task},
    ]
    use_tools = True
    for round_i in range(max_rounds):
        try:
            if use_tools:
                msg = chat(messages, tools=TOOLS_SPEC, tool_choice="auto")
            else:
                msg = chat(messages, tools=None, tool_choice=None)
        except RuntimeError as e:
            err = str(e)
            # Fallback: model may not support tools — switch to JSON-action protocol
            if use_tools and ("tool" in err.lower() or "400" in err or "tools" in err.lower()):
                use_tools = False
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Tools API failed. Reply ONLY with JSON like "
                            '{"tool":"list_dir","args":{"path":"002"}} or '
                            '{"final":"your answer"}. Available tools: '
                            "list_dir, read_file, run_shell, lab_help."
                        ),
                    }
                )
                continue
            raise

        tool_calls = _parse_tool_calls(msg)
        content = (msg.get("content") or "").strip()

        if not use_tools and content:
            # JSON action fallback
            try:
                # extract JSON object
                start = content.find("{")
                end = content.rfind("}")
                if start >= 0 and end > start:
                    obj = json.loads(content[start : end + 1])
                    if "final" in obj:
                        return str(obj["final"])
                    if "tool" in obj:
                        result = dispatch(obj["tool"], obj.get("args") or {})
                        messages.append({"role": "assistant", "content": content})
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Tool {obj['tool']} result:\n{result}\nContinue.",
                            }
                        )
                        continue
            except json.JSONDecodeError:
                return content
            return content

        if not tool_calls:
            return content or "(empty model response)"

        # NIM: many models only support a single tool-call per turn
        tc = tool_calls[0]
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        args = fn.get("arguments") or "{}"
        print(f"  [tool round {round_i+1}] {name}({args})")
        result = dispatch(name, args)

        # Keep only the first tool_call in the assistant message for API compatibility
        msg_one = dict(msg)
        msg_one["tool_calls"] = [tc]
        messages.append(msg_one)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.get("id") or "call_0",
                "name": name,
                "content": result[:8000],
            }
        )

    return "达到最大工具轮次，请缩小任务后重试。"


def ping() -> str:
    models = []
    try:
        models = list_models()[:15]
    except Exception as e:  # noqa: BLE001
        models = [f"(list_models failed: {e})"]
    reply = chat_text(
        "用一句话介绍你自己，并确认你是通过 NVIDIA NIM 提供的模型。",
        system="Be brief.",
    )
    return (
        f"model={model_name()}\n"
        f"sample_models={models}\n"
        f"reply={reply}"
    )
