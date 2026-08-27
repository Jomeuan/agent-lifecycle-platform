"""开发智能体（DevAgent）：基于 LangGraph 状态图，根据需求生成可运行的智能体包。

改造 1 落地：
- 从"单次 LLM 调用生成 JSON"重构为 LangGraph 状态图：
    design（设计）→ implement（实现）→ syntax_check（语法门禁）→ smoke_test（冒烟测试）→ finish
  任一检查失败则进入 fix（修复）回环，直到通过或超过重试上限。
- 语法检查（py_compile）与冒烟测试内置为图节点，开发与测试合并在图内，
  图成功出口等价于"已通过语法门禁 + 冒烟测试"。
- 全流程通过 `logging` 输出持久化日志（DEBUG 级落盘）。

对外入口：`develop(requirement, llm, agents_root, max_attempts)`。
"""

from __future__ import annotations

import hashlib
import json
import re
from operator import add
from pathlib import Path
from typing import Annotated, TypedDict

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, StateGraph

from .logging_setup import get_logger
from .runtime import load_package, run_agent

log = get_logger("dev_agent")
llm_log = get_logger("llm")

# ---------------------------------------------------------------------------
# 状态定义
# ---------------------------------------------------------------------------


class DevState(TypedDict, total=False):
    """LangGraph 图共享状态；所有节点读写同一份 state。"""

    # 输入
    requirement: str
    # 设计产物
    design: dict
    # 实现产物（三件套，落盘前以字符串/字典形态在内存中流转）
    files: dict
    # 语法门禁
    syntax_ok: bool
    syntax_errors: str
    # 冒烟测试
    smoke_ok: bool
    smoke_output: str
    # 循环控制
    attempts: int
    max_attempts: int
    last_feedback: str
    # 产出
    agent_id: str
    agent_dir: str
    status: str
    error: str
    # 全量过程记录（节点名 + 摘要，供日志/回放）
    messages: Annotated[list, add]


# ---------------------------------------------------------------------------
# 提示词模板
# ---------------------------------------------------------------------------

DESIGN_SYSTEM = """你是「智能体开发智能体」（DevAgent）的**设计**阶段。

根据用户对目标智能体的需求描述，产出一份**设计稿**（只做设计，不写最终代码）。

设计稿必须是 JSON 对象，字段如下：
{
  "name": "智能体名称",
  "description": "一句话描述",
  "prompt_draft": "system prompt 草案（写清任务、工作流程、输出格式）",
  "tools": [{"name": "工具函数名", "signature": "参数列表（含类型注解）", "purpose": "一句话用途"}],
  "input_schema": {"type": "object", "properties": {}, "required": []},
  "output_schema": {"type": "string", "description": "..."},
  "smoke_input_draft": "一段示例输入，用于自测冒烟"
}

约束：
- tools 里每个工具都必须是「无副作用纯函数」，返回字符串或可转字符串的值；
- output_schema 要简单、可被代码规则校验（例如只输出「通过」或「驳回：理由」）；
- 只输出 JSON，不要任何额外解释。
"""

IMPLEMENT_SYSTEM = """你是「智能体开发智能体」（DevAgent）的**实现**阶段。

根据给定的设计稿，生成智能体包的三个文件内容。

智能体包由统一 ReAct 运行时执行，包含三个文件：
1. prompt —— 该智能体的 system prompt 全文，写清任务、工作流程、输出格式；
2. tools_py —— tools.py 全文。必须用 @tool 装饰器（`from alip.tools import tool`），
   每个函数要有类型注解和 docstring，函数必须是无副作用的纯函数、返回字符串或可转字符串的值；
   只允许 import 标准库和 alip.tools；
3. agent_yaml —— 配置 dict，必须包含：model、temperature、max_steps、smoke_input、input_schema、output_schema。

如果提供了「上次失败反馈」，只修复问题部分，不要大改已经通过的部分。

只输出一个 JSON 对象（不要任何额外解释）：
{
  "prompt": "system prompt 全文",
  "tools_py": "tools.py 全文",
  "agent_yaml": {"model": "...", "temperature": 0.1, "max_steps": 5, "smoke_input": "...", "input_schema": {}, "output_schema": {}}
}
"""

FIX_SYSTEM = """你是「智能体开发智能体」（DevAgent）的**修复**阶段。

给定：目标智能体的当前三件套 + 一次失败原因（语法报错或冒烟测试输出）。

请修复问题，并返回修复后的完整三件套 JSON，只修改必要部分。

失败原因可能是：
- 语法错误：修正 tools.py 的语法；
- 冒烟失败：可能是工具写错，也可能是 prompt 指令不清，请判断后修复 tools.py 或 prompt。

只输出一个 JSON 对象：
{
  "prompt": "system prompt 全文",
  "tools_py": "tools.py 全文",
  "agent_yaml": {}
}
"""

MAIN_PY_TEMPLATE = '''"""平台统一提供的入口：独立运行该智能体包。"""

import sys
from pathlib import Path

from alip.llm import get_llm
from alip.runtime import load_package, run_agent

if __name__ == "__main__":
    pkg = load_package(Path(__file__).parent)
    user_input = sys.argv[1] if len(sys.argv) > 1 else pkg.config.get("smoke_input", "你好")
    result = run_agent(pkg, user_input, get_llm())
    print(result.output)
'''


# ---------------------------------------------------------------------------
# 工具函数（沿用旧实现）
# ---------------------------------------------------------------------------


def extract_json(text: str) -> str:
    """从模型输出中提取 JSON（容忍代码围栏与前后杂文）。"""
    text = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*\n(.*?)\n?```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("无法从模型输出中解析 JSON")
    return text[start : end + 1]


def make_agent_id(name: str) -> str:
    """由名称生成目录友好的 agent_id。"""
    base = re.sub(r"[^a-zA-Z0-9]+", "-", name or "").strip("-").lower()
    if not base:
        base = "agent"
    h = hashlib.md5((name or "agent").encode("utf-8")).hexdigest()[:6]
    return f"{base}-{h}"


def scaffold_agent(data: dict, agents_root: Path, agent_id: str, version: str = "v1") -> Path:
    """把生成结果落盘为智能体包目录，返回版本目录。"""
    vdir = Path(agents_root) / agent_id / version
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "prompt.md").write_text(data["prompt"], encoding="utf-8")
    (vdir / "tools.py").write_text(data["tools_py"], encoding="utf-8")
    yaml_text = yaml.safe_dump(data["agent_yaml"], allow_unicode=True, sort_keys=False)
    (vdir / "agent.yaml").write_text(yaml_text, encoding="utf-8")
    (vdir / "main.py").write_text(MAIN_PY_TEMPLATE, encoding="utf-8")
    return vdir


def _chat(llm: ChatDeepSeek, system: str, user: str) -> str:
    """调用 ChatDeepSeek 返回文本内容；空内容自动重试一次；请求/输出落 llm 日志。"""
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    llm_log.debug("请求：\n--- system ---\n%s\n--- user ---\n%s", system, user)
    msg = llm.invoke(messages)
    if not msg.content:
        log.warning("LLM 返回空 content，自动重试一次")
        llm_log.debug("（空输出，自动重试一次）")
        msg = llm.invoke(messages)
    content = msg.content
    if isinstance(content, list):  # 多模态内容可能是 list
        content = "".join(str(c) for c in content)
    if not content:
        llm_log.debug("输出：<空>")
        raise RuntimeError("LLM 返回空内容，请检查模型/API 配置")
    llm_log.debug("输出：\n%s", content)
    return str(content)


# ---------------------------------------------------------------------------
# 图节点（工厂函数，闭包捕获 llm / agents_root）
# ---------------------------------------------------------------------------


def _design_node(llm: ChatDeepSeek, name: str | None = None):
    def node(state: DevState) -> dict:
        log.debug("design 节点：根据需求生成设计稿")
        content = _chat(llm, DESIGN_SYSTEM, state["requirement"])
        design = json.loads(extract_json(content))
        if name:
            design["name"] = name
        agent_id = make_agent_id(design.get("name") or "agent")
        log.debug("design 节点完成：name=%s agent_id=%s", design.get("name"), agent_id)
        return {
            "design": design,
            "agent_id": agent_id,
            "messages": [{"node": "design", "summary": f"设计稿：{design.get('name')}"}],
        }

    return node


def _implement_node(llm: ChatDeepSeek):
    def node(state: DevState) -> dict:
        last_feedback = state.get("last_feedback", "")
        if last_feedback:
            log.debug("implement 节点：根据修复反馈重新生成（attempts=%s）", state.get("attempts"))
            user = (
                f"设计稿：\n{json.dumps(state['design'], ensure_ascii=False)}\n\n"
                f"上次失败反馈（只修复问题部分）：\n{last_feedback}"
            )
        else:
            log.debug("implement 节点：根据设计稿首次生成")
            user = f"设计稿：\n{json.dumps(state['design'], ensure_ascii=False)}"

        content = _chat(llm, IMPLEMENT_SYSTEM, user)
        files = json.loads(extract_json(content))
        if not all(k in files for k in ("prompt", "tools_py", "agent_yaml")):
            raise RuntimeError("implement 输出缺少 prompt/tools_py/agent_yaml 字段")
        log.debug("implement 节点完成：prompt=%d 字符, tools_py=%d 字符", len(files["prompt"]), len(files["tools_py"]))
        return {"files": files, "messages": [{"node": "implement", "summary": "生成三件套"}]}

    return node


def _syntax_check_node():
    REQUIRED_YAML = ("model", "temperature", "max_steps", "smoke_input", "input_schema", "output_schema")

    def node(state: DevState) -> dict:
        files = state["files"]
        errors: list[str] = []

        # 1) tools.py 语法
        try:
            compile(files["tools_py"], "<tools.py>", "exec")
        except SyntaxError as exc:
            errors.append(f"tools.py 语法错误: {exc}")

        # 2) agent.yaml 可解析且含必填字段
        try:
            cfg = files["agent_yaml"]
            if isinstance(cfg, str):
                cfg = yaml.safe_load(cfg) or {}
            missing = [k for k in REQUIRED_YAML if k not in cfg]
            if missing:
                errors.append(f"agent.yaml 缺少必填字段: {', '.join(missing)}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"agent.yaml 解析失败: {exc}")

        # 3) prompt 非空
        if not (files.get("prompt") or "").strip():
            errors.append("prompt 为空")

        ok = not errors
        attempts = state.get("attempts", 0) + (0 if ok else 1)
        if ok:
            log.debug("syntax_check 节点：通过")
        else:
            log.debug("syntax_check 节点：失败（第 %d 次重试）→ %s", attempts, " | ".join(errors))
        return {
            "syntax_ok": ok,
            "syntax_errors": "\n".join(errors),
            "attempts": attempts,
            "messages": [{"node": "syntax_check", "summary": "语法通过" if ok else f"语法失败：{errors[0]}"}],
        }

    return node


def _smoke_test_node(llm: ChatDeepSeek, agents_root: Path):
    def node(state: DevState) -> dict:
        files = state["files"]
        agent_id = state["agent_id"]
        vdir = scaffold_agent(files, agents_root, agent_id, "v1")
        pkg = load_package(vdir)
        smoke_input = pkg.config.get("smoke_input", "你好")
        log.debug("smoke_test 节点：用 smoke_input 跑通自测（input=%s）", smoke_input[:50])
        try:
            result = run_agent(pkg, smoke_input, llm)
        except Exception as exc:  # noqa: BLE001
            ok = False
            out = str(exc)
        else:
            ok = bool(result.output and result.output.strip()) and not result.error and result.tool_errors == 0
            out = result.output
        attempts = state.get("attempts", 0) + (0 if ok else 1)
        if ok:
            log.debug("smoke_test 节点：通过")
        else:
            log.debug("smoke_test 节点：失败（第 %d 次重试）→ %s", attempts, out[:120])
        return {
            "smoke_ok": ok,
            "smoke_output": out,
            "attempts": attempts,
            "messages": [{"node": "smoke_test", "summary": "冒烟通过" if ok else f"冒烟失败：{out[:80]}"}],
        }

    return node


def _fix_node(llm: ChatDeepSeek):
    def node(state: DevState) -> dict:
        files = state["files"]
        reason = state.get("syntax_errors") or state.get("smoke_output") or "未知原因"
        log.debug("fix 节点：携带失败原因定向修复")
        user = (
            f"当前三件套：\n{json.dumps(files, ensure_ascii=False)}\n\n"
            f"失败原因：\n{reason}\n\n"
            f"请修复后返回完整三件套 JSON。"
        )
        content = _chat(llm, FIX_SYSTEM, user)
        new_files = json.loads(extract_json(content))
        if not all(k in new_files for k in ("prompt", "tools_py", "agent_yaml")):
            raise RuntimeError("fix 输出缺少 prompt/tools_py/agent_yaml 字段")
        return {
            "files": new_files,
            "last_feedback": reason,
            "messages": [{"node": "fix", "summary": f"修复：{reason[:80]}"}],
        }

    return node


def _finish_node(agents_root: Path):
    def node(state: DevState) -> dict:
        agent_id = state["agent_id"]
        vdir = scaffold_agent(state["files"], agents_root, agent_id, "v1")
        log.debug("finish 节点：落盘 %s", vdir)
        return {"status": "success", "agent_dir": str(vdir),
                "messages": [{"node": "finish", "summary": f"落盘 {vdir}"}]}

    return node


def _fail_node():
    def node(state: DevState) -> dict:
        reason = state.get("syntax_errors") or state.get("smoke_output") or "未知原因"
        err = f"开发失败（重试 {state.get('attempts', 0)} 次后仍失败）：{reason}"
        log.error("fail 节点：%s", err)
        return {"status": "failed", "error": err,
                "messages": [{"node": "fail", "summary": err}]}

    return node


# ---------------------------------------------------------------------------
# 路由函数
# ---------------------------------------------------------------------------


def _route_after_syntax(state: DevState) -> str:
    if state.get("syntax_ok"):
        return "smoke_test"
    if state.get("attempts", 0) >= state.get("max_attempts", 5):
        return "fail"
    return "fix"


def _route_after_smoke(state: DevState) -> str:
    if state.get("smoke_ok"):
        return "finish"
    if state.get("attempts", 0) >= state.get("max_attempts", 5):
        return "fail"
    return "fix"


# ---------------------------------------------------------------------------
# 图构建与对外入口
# ---------------------------------------------------------------------------


def build_dev_graph(llm: ChatDeepSeek, agents_root: Path, max_attempts: int = 5, name: str | None = None):
    """构建并编译 DevAgent 状态图。"""
    graph = StateGraph(DevState)

    graph.add_node("design", _design_node(llm, name))
    graph.add_node("implement", _implement_node(llm))
    graph.add_node("syntax_check", _syntax_check_node())
    graph.add_node("smoke_test", _smoke_test_node(llm, agents_root))
    graph.add_node("fix", _fix_node(llm))
    graph.add_node("finish", _finish_node(agents_root))
    graph.add_node("fail", _fail_node())

    graph.add_edge(START, "design")
    graph.add_edge("design", "implement")
    graph.add_edge("implement", "syntax_check")
    graph.add_conditional_edges(
        "syntax_check", _route_after_syntax, {"smoke_test": "smoke_test", "fix": "fix", "fail": "fail"}
    )
    graph.add_conditional_edges(
        "smoke_test", _route_after_smoke, {"finish": "finish", "fix": "fix", "fail": "fail"}
    )
    graph.add_edge("fix", "implement")
    graph.add_edge("finish", END)
    graph.add_edge("fail", END)

    return graph.compile()


def develop(
    requirement: str,
    llm: ChatDeepSeek,
    agents_root: Path,
    max_attempts: int = 5,
    name: str | None = None,
) -> dict:
    """运行 DevAgent 状态图，返回最终 state（status 为 success 或 failed）。"""
    log.info("DevAgent 启动：requirement=%s", requirement[:60])
    graph = build_dev_graph(llm, agents_root, max_attempts, name)
    state = graph.invoke(
        {
            "requirement": requirement,
            "design": {},
            "files": {},
            "syntax_ok": False,
            "syntax_errors": "",
            "smoke_ok": False,
            "smoke_output": "",
            "attempts": 0,
            "max_attempts": max_attempts,
            "last_feedback": "",
            "agent_id": "",
            "agent_dir": "",
            "status": "running",
            "error": "",
            "messages": [],
        }
    )
    if state.get("status") != "success":
        raise RuntimeError(state.get("error", "DevAgent 开发失败"))
    log.info("DevAgent 完成：agent_id=%s agent_dir=%s", state.get("agent_id"), state.get("agent_dir"))
    return state
