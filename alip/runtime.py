"""统一 ReAct 运行时。

智能体包 = prompt.md + tools.py + agent.yaml；
运行时负责加载智能体包，并执行极简 ReAct 循环：
    LLM 思考 -> 调用工具 -> 观察结果 -> 循环 -> 最终回答
同时采集 token 消耗、耗时、工具调用情况（供评测指标使用）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import tools as tool_mod
from .llm import LLMClient


@dataclass
class AgentPackage:
    """一个已加载的智能体包（对应 agents/<id>/<version>/ 目录）。"""

    agent_dir: Path
    prompt: str
    config: dict
    tools: dict

    @property
    def model(self) -> str:
        return self.config.get("model", "")

    @property
    def temperature(self) -> float:
        return float(self.config.get("temperature", 0.2))

    @property
    def max_steps(self) -> int:
        return int(self.config.get("max_steps", 6))


@dataclass
class RunResult:
    """一次智能体调用的运行结果与开销统计。"""

    output: str
    steps: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    tool_calls: int = 0
    tool_errors: int = 0
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def load_package(agent_dir: Path) -> AgentPackage:
    """加载智能体包：prompt.md + agent.yaml + tools.py。"""
    agent_dir = Path(agent_dir)
    prompt = (agent_dir / "prompt.md").read_text(encoding="utf-8")
    config = yaml.safe_load((agent_dir / "agent.yaml").read_text(encoding="utf-8")) or {}
    tools = tool_mod.load_tools(agent_dir / "tools.py")
    return AgentPackage(agent_dir=agent_dir, prompt=prompt, config=config, tools=tools)


def run_agent(
    package: AgentPackage,
    user_input: str,
    llm: LLMClient,
    temperature: float | None = None,
    max_steps: int | None = None,
) -> RunResult:
    """执行 ReAct 循环，返回最终输出与开销统计。"""
    max_steps = max_steps or package.max_steps
    temperature = package.temperature if temperature is None else temperature
    tool_schema = tool_mod.schema_for_tools(package.tools)

    messages: list[dict] = [
        {"role": "system", "content": package.prompt},
        {"role": "user", "content": user_input},
    ]

    result = RunResult(output="")
    start = time.perf_counter()
    final_output = ""

    try:
        for _ in range(max_steps):
            resp = llm.chat(
                messages,
                tools=tool_schema or None,
                temperature=temperature,
                model=package.model or None,
            )
            result.prompt_tokens += resp.prompt_tokens
            result.completion_tokens += resp.completion_tokens

            if resp.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": resp.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": tc.arguments},
                            }
                            for tc in resp.tool_calls
                        ],
                    }
                )
                for tc in resp.tool_calls:
                    result.tool_calls += 1
                    fn = package.tools.get(tc.name)
                    if fn is None:
                        obs = f"Error: 未知工具 '{tc.name}'"
                        result.tool_errors += 1
                    else:
                        try:
                            args = json.loads(tc.arguments) if tc.arguments.strip() else {}
                            if isinstance(args, dict):
                                out = fn(**args)
                            else:
                                out = fn(args)
                            obs = str(out)
                        except Exception as exc:  # noqa: BLE001
                            obs = f"ToolError: {exc}"
                            result.tool_errors += 1
                    result.steps.append(
                        {"type": "tool_call", "name": tc.name, "arguments": tc.arguments, "result": obs}
                    )
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs})
            else:
                final_output = resp.content or ""
                result.steps.append({"type": "answer", "content": final_output})
                break
        else:
            result.steps.append({"type": "max_steps_reached", "content": final_output})
            final_output = final_output or "（未在最大步数内得到最终答案）"
    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)
        final_output = f"运行错误: {exc}"

    result.output = final_output
    result.latency_ms = (time.perf_counter() - start) * 1000.0
    return result
