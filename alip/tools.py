"""工具协议：函数即 tool。

智能体在 tools.py 中使用 ``@tool`` 装饰器声明工具函数，
统一运行时通过 :func:`load_tools` 加载，并用 :func:`schema_for_tools`
将函数签名/类型注解/文档字符串转换为 OpenAI function-calling 的 JSON Schema。

预留：后续可在此处增加 MCP 包装点（把工具暴露为 MCP tool）。
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import StructuredTool

# 模块级工具注册表。load_tools 每次加载前都会清空，
# 避免多个智能体之间的工具相互污染。
_TOOL_REGISTRY: dict[str, Callable] = {}

_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool(func: Callable) -> Callable:
    """装饰器：把函数注册为智能体可用的 tool。"""
    _TOOL_REGISTRY[func.__name__] = func
    return func


def get_registered_tools() -> dict[str, Callable]:
    return dict(_TOOL_REGISTRY)


def clear_registry() -> None:
    _TOOL_REGISTRY.clear()


def load_tools(path: Path) -> dict[str, Callable]:
    """加载 tools.py 并返回其中注册的工具。"""
    clear_registry()
    src = path.read_text(encoding="utf-8")
    namespace: dict[str, Any] = {"__name__": "agent_tools", "__file__": str(path)}
    exec(compile(src, str(path), "exec"), namespace)
    return get_registered_tools()


def _py_type_to_json(t: Any) -> str:
    origin = getattr(t, "__origin__", None)
    if origin is not None:
        # 仅处理常见泛型容器（list/dict），其余回退为 string
        if origin in (list, tuple, set):
            return "array"
        if origin is dict:
            return "object"
    return _TYPE_MAP.get(t, "string")


def schema_for_tools(tools: dict[str, Callable]) -> list[dict]:
    """把工具字典转换为 OpenAI function-calling 的工具 schema 列表。"""
    result = []
    for name, fn in tools.items():
        sig = inspect.signature(fn)
        props: dict[str, dict] = {}
        required: list[str] = []
        for pname, p in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            props[pname] = {
                "type": _py_type_to_json(p.annotation),
                "description": "",
            }
            if p.default is inspect.Parameter.empty:
                required.append(pname)
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": (fn.__doc__ or "").strip(),
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                    },
                },
            }
        )
    return result


def to_langchain_tools(tools: dict[str, Callable]) -> list:
    """把 @tool 注册的函数字典转换为 LangChain BaseTool 列表（供 create_react_agent/ToolNode 使用）。

    依据函数类型注解自动生成 args_schema（Pydantic），无需手写 schema。
    """
    result = []
    for name, fn in tools.items():
        result.append(
            StructuredTool.from_function(
                func=fn,
                name=name,
                description=(fn.__doc__ or "").strip(),
            )
        )
    return result
