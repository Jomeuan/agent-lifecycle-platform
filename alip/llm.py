"""LLM 客户端（DeepSeek API，OpenAI 兼容协议）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import requests

from .config import get_settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient:
    """极简 OpenAI 兼容的 chat completions 客户端。"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        msg = choice.get("message") or {}
        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments", ""),
                )
            )
        usage = data.get("usage") or {}
        return LLMResponse(
            content=msg.get("content"),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )


def get_llm() -> LLMClient:
    """从环境配置构建 LLM 客户端；缺少 API Key 时给出清晰报错。"""
    s = get_settings()
    if not s["api_key"]:
        raise RuntimeError(
            "未检测到 DEEPSEEK_API_KEY。\n"
            "请复制 .env.example 为 .env，并填入你的 DeepSeek API Key，"
            "或在环境变量中设置 DEEPSEEK_API_KEY。"
        )
    return LLMClient(base_url=s["base_url"], api_key=s["api_key"], model=s["model"])
