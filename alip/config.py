"""路径与配置中心。

所有路径都集中在项目根目录下，本地即用、零部署。
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

AGENTS_DIR = ROOT / "agents"          # DevAgent 生成的智能体包
SAMPLES_DIR = ROOT / "samples"        # 手工编写的示例智能体
TESTSETS_DIR = ROOT / "testsets"      # 评测测试集（含指标阈值）
DATA_DIR = ROOT / "data"              # 本地数据
REPORTS_DIR = DATA_DIR / "reports"    # 评测报告
LOGS_DIR = DATA_DIR / "logs"          # 持久化日志
DB_PATH = DATA_DIR / "platform.db"    # SQLite 数据库

ENV_PATH = ROOT / ".env"

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


def ensure_dirs() -> None:
    """确保运行时所需目录存在。"""
    for d in (AGENTS_DIR, DATA_DIR, REPORTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_dotenv(path: Path | None = None) -> None:
    """极简 .env 加载器（避免额外依赖 python-dotenv）。"""
    env_path = path or ENV_PATH
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_settings() -> dict:
    """读取 LLM 相关配置。"""
    load_dotenv()
    return {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        "model": os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
    }
