"""统一 ReAct 运行时。

智能体包 = prompt.md + tools.py + agent.yaml；
运行时负责加载智能体包，并执行极简 ReAct 循环：
    LLM 思考 -> 调用工具 -> 观察结果 -> 循环 -> 最终回答
同时采集 token 消耗、耗时、工具调用情况（供评测指标使用）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import create_react_agent

from . import tools as tool_mod
from .logging_setup import get_logger

llm_log = get_logger("llm")


def _render_content(content) -> str:
    if isinstance(content, list):
        return "".join(str(c) for c in content)
    return str(content)


class _LLMCallLoggingHandler(BaseCallbackHandler):
    """把 ReAct 循环里每次 LLM 请求/输出记录到 alip.llm 日志。"""

    def on_chat_model_start(self, serialized, messages, **kwargs):
        for batch in messages:
            for m in batch:
                role = getattr(m, "type", type(m).__name__)
                llm_log.debug("请求 [%s]: %s", role, _render_content(m.content))

    def on_llm_end(self, response, **kwargs):
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is not None:
                    llm_log.debug(
                        "输出: content=%s tool_calls=%s",
                        _render_content(msg.content),
                        getattr(msg, "tool_calls", None),
                    )


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
    llm: ChatDeepSeek,
    temperature: float | None = None,
    max_steps: int | None = None,
) -> RunResult:
    """用 LangGraph 预构建 ReAct agent 执行，返回最终输出与开销统计。"""
    result = RunResult(output="")
    start = time.perf_counter()
    try:
        lc_tools = tool_mod.to_langchain_tools(package.tools)
        temp = package.temperature if temperature is None else temperature
        model = llm.bind(temperature=temp) if temp is not None else llm

        agent = create_react_agent(model, lc_tools, prompt=package.prompt)
        recursion_limit = (max_steps or package.max_steps) * 2 + 4
        final_state = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"recursion_limit": recursion_limit, "callbacks": [_LLMCallLoggingHandler()]},
        )
        messages = final_state.get("messages", [])

        for msg in messages:
            if isinstance(msg, AIMessage):
                um = msg.usage_metadata or {}
                result.prompt_tokens += int(um.get("input_tokens", 0))
                result.completion_tokens += int(um.get("output_tokens", 0))
                if msg.tool_calls:
                    result.tool_calls += len(msg.tool_calls)
            elif isinstance(msg, ToolMessage):
                if getattr(msg, "status", None) == "error" or str(msg.content).startswith("Error:"):
                    result.tool_errors += 1

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                result.output = str(msg.content)
                break
        if not result.output:
            result.output = "（未得到最终答案）"
    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)
        result.output = f"运行错误: {exc}"

    result.latency_ms = (time.perf_counter() - start) * 1000.0
    return result
