"""持久化日志配置：控制台 INFO + 文件 DEBUG。

所有 `alip.*` 命名的 logger 由这里统一配置：
- 控制台（StreamHandler）：INFO 级别，关键里程碑可见；
- 文件（FileHandler）：DEBUG 级别，完整开发流程落盘到 data/logs/。
"""

from __future__ import annotations

import logging
from pathlib import Path

_FMT = logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s")


def setup_logging(agent_id: str | None = None, logs_dir: Path | None = None) -> None:
    """配置 `alip` 命名空间的日志；可重复调用，重复调用会重置 handler。"""
    root = logging.getLogger("alip")
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)
    root.propagate = False

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(_FMT)
    root.addHandler(console)

    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        filename = f"dev_agent-{agent_id}.log" if agent_id else "platform.log"
        file_handler = logging.FileHandler(logs_dir / filename, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(_FMT)
        root.addHandler(file_handler)


def setup_llm_logging(logs_dir: Path | None = None) -> None:
    """配置 `alip.llm` 独立日志：记录每次 LLM 请求/输出，写入独立文件，不进入 platform.log。"""
    llm_logger = logging.getLogger("alip.llm")
    for h in list(llm_logger.handlers):
        llm_logger.removeHandler(h)
    llm_logger.setLevel(logging.DEBUG)
    llm_logger.propagate = False  # 关键：不传播到 alip 根 logger，因此不进 platform.log

    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logs_dir / "llm_calls.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(_FMT)
        llm_logger.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """获取 `alip.<name>` 命名空间下的 logger。"""
    return logging.getLogger(f"alip.{name}")
