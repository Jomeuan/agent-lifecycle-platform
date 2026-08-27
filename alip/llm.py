"""LLM 客户端（DeepSeek，基于 langchain-deepseek 的 ChatDeepSeek）。

改造 1 落地：用 LangChain 官方维护的 ChatDeepSeek 替代自研 OpenAI 兼容客户端，
正确处理 reasoning_content、tool_calls、usage、max_retries 等边界，不再自行实现协议。

对外入口：`get_llm()` 返回 ChatDeepSeek 实例。
"""

from __future__ import annotations

from langchain_deepseek import ChatDeepSeek

from .config import LOGS_DIR, get_settings
from .logging_setup import setup_llm_logging


def get_llm(temperature: float = 0.2) -> ChatDeepSeek:
    """从环境配置构建 ChatDeepSeek；缺少 API Key 时给出清晰报错。"""
    s = get_settings()
    if not s["api_key"]:
        raise RuntimeError(
            "未检测到 DEEPSEEK_API_KEY。\n"
            "请复制 .env.example 为 .env，并填入你的 DeepSeek API Key，"
            "或在环境变量中设置 DEEPSEEK_API_KEY。"
        )
    setup_llm_logging(LOGS_DIR)
    return ChatDeepSeek(
        model=s["model"],
        api_key=s["api_key"],
        api_base=s["base_url"],
        temperature=temperature,
        max_tokens=32768,
        reasoning_effort="low",
        request_timeout=120,
    )
