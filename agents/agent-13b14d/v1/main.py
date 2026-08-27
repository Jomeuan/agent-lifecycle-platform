"""平台统一提供的入口：独立运行该智能体包。"""

import sys
from pathlib import Path

from alip.llm import get_llm
from alip.runtime import load_package, run_agent

if __name__ == "__main__":
    pkg = load_package(Path(__file__).parent)
    user_input = sys.argv[1] if len(sys.argv) > 1 else pkg.config.get("smoke_input", "你好")
    result = run_agent(pkg, user_input, get_llm())
    print(result.output)
