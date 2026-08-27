"""智能体全生命周期管理平台 —— 命令行入口。

用法示例：
    python main.py create "帮我做一个智能体：..."
    python main.py list
    python main.py register samples/account_compliance
    python main.py evaluate <agent_id>
    python main.py run <agent_id> "输入文本"
    python main.py demo
"""

from alip.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
