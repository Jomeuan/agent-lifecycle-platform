"""开发智能体（DevAgent）：根据需求描述，生成智能体「三件套」+ 冒烟测试。

三件套：prompt.md + tools.py + agent.yaml，外加平台统一提供的 main.py。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

from .llm import LLMClient
from .runtime import AgentPackage, load_package, run_agent

DEV_SYSTEM = """你是「智能体开发智能体」（DevAgent）。你的任务：根据用户对目标智能体的需求描述，生成一个可运行的智能体包。

智能体包由统一 ReAct 运行时执行，包含三个文件：
1. prompt.md —— 该智能体的 system prompt（设计产物），要写清任务、工作流程、输出格式；
2. tools.py —— 工具函数。必须用 @tool 装饰器（`from alip.tools import tool`），每个函数要有类型注解和 docstring，函数必须是无副作用的纯函数、返回字符串或可转字符串的值；只允许 import 标准库和 alip.tools；
3. agent.yaml —— 配置，必须包含以下字段：model、temperature、max_steps、smoke_input、input_schema、output_schema。

约束：
- 运行时会自动发现 @tool 装饰的函数，并作为 OpenAI function-calling tools 传给模型；
- smoke_input 是一段示例输入，用于冒烟测试验证智能体能跑通；
- 输出协议要简单、可被代码规则校验（例如只输出「通过」或「驳回：理由」）。

只输出一个 JSON 对象（不要任何额外解释），格式：
{
  "name": "智能体名称",
  "description": "一句话描述",
  "prompt": "system prompt 全文",
  "tools_py": "tools.py 全文",
  "agent_yaml": {"model": "deepseek-chat", "temperature": 0.1, "max_steps": 5, "smoke_input": "...", "input_schema": {...}, "output_schema": {...}}
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


def generate(requirement: str, llm: LLMClient) -> dict:
    """调用 LLM 生成智能体包定义。"""
    messages = [
        {"role": "system", "content": DEV_SYSTEM},
        {"role": "user", "content": requirement},
    ]
    resp = llm.chat(messages, temperature=0.2)
    if not resp.content:
        raise RuntimeError("DevAgent 未返回内容")
    return json.loads(extract_json(resp.content))


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


def smoke_test(package: AgentPackage, llm: LLMClient) -> tuple[bool, str]:
    """冒烟测试：用 smoke_input 跑一次，验证输出非空且无工具异常。"""
    smoke_input = package.config.get("smoke_input", "你好")
    try:
        r = run_agent(package, smoke_input, llm)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    ok = bool(r.output and r.output.strip()) and not r.error and r.tool_errors == 0
    return ok, r.output
