"""命令行入口：create / register / evaluate / run / list / status / demo。

把「开发 -> 冒烟 -> 注册 -> 评测 -> 上线 -> 调用」串成完整闭环。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config, dev_agent, evaluator, report
from .llm import get_llm
from .logging_setup import setup_logging
from .registry import Registry
from .runtime import load_package, run_agent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alip",
        description="智能体全生命周期管理平台 CLI（create/register/evaluate/run/list/status/demo）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="输入需求，由 DevAgent 开发一个智能体并注册")
    p_create.add_argument("requirement", nargs="+", help="需求描述（一句话）")
    p_create.add_argument("--name", default=None, help="智能体名称（可选）")

    p_register = sub.add_parser("register", help="把一个智能体包注册入库")
    p_register.add_argument("agent_dir", help="智能体包目录（含 prompt.md/tools.py/agent.yaml）")

    p_evaluate = sub.add_parser("evaluate", help="用测试集评测智能体并输出数字指标报告")
    p_evaluate.add_argument("agent_id", help="智能体 ID")
    p_evaluate.add_argument("--testset", default=None, help="测试集 JSON 路径（默认 testsets/account_compliance.json）")

    p_run = sub.add_parser("run", help="通过统一运行时调用智能体")
    p_run.add_argument("agent_id", help="智能体 ID")
    p_run.add_argument("input", nargs="+", help="输入文本")

    sub.add_parser("list", help="列出已注册智能体")

    p_status = sub.add_parser("status", help="查看智能体状态与最近一次评测")
    p_status.add_argument("agent_id", help="智能体 ID")

    sub.add_parser("demo", help="跑开户合规端到端演示（create -> register -> evaluate -> run）")

    return parser


# ---- 命令实现 ----

def _get_registry() -> Registry:
    config.ensure_dirs()
    return Registry(config.DB_PATH)


def _create_agent(requirement: str, name: str | None, llm) -> str:
    """启动 DevAgent LangGraph 图 -> 注册，返回 agent_id。"""
    setup_logging(agent_id=None, logs_dir=config.LOGS_DIR)
    print("▶ 1/3 开发：DevAgent（LangGraph）正在生成并自测智能体 ...")
    state = dev_agent.develop(requirement, llm, config.AGENTS_DIR, name=name)
    agent_id = state["agent_id"]
    agent_dir = state["agent_dir"]
    # 后续日志落到该智能体的专属文件
    setup_logging(agent_id=agent_id, logs_dir=config.LOGS_DIR)
    print(f"   ✅ 智能体已生成并通过语法 + 冒烟测试：{agent_dir}")

    print("▶ 2/3 注册入库 ...")
    reg = _get_registry()
    design = state.get("design", {})
    reg.register_agent(
        {
            "id": agent_id,
            "name": design.get("name", agent_id),
            "description": design.get("description", ""),
            "source": "developed",
            "version": "v1",
            "agent_dir": agent_dir,
        }
    )
    print(f"   ✅ 已注册：{agent_id}（状态: registered）")
    return agent_id


def cmd_create(args) -> int:
    llm = get_llm()
    agent_id = _create_agent(" ".join(args.requirement), args.name, llm)
    print(f"▶ 3/3 下一步：python main.py evaluate {agent_id}")
    return 0


def cmd_register(args) -> int:
    reg = _get_registry()
    agent_dir = Path(args.agent_dir).resolve()
    pkg = load_package(agent_dir)
    agent_id = agent_dir.parent.name if agent_dir.name.lower().startswith("v") else agent_dir.name
    version = agent_dir.name if agent_dir.name.lower().startswith("v") else "v1"
    reg.register_agent(
        {
            "id": agent_id,
            "name": pkg.config.get("name", agent_id),
            "description": pkg.config.get("description", ""),
            "source": "developed",
            "version": version,
            "agent_dir": str(agent_dir),
        }
    )
    print(f"✅ 已注册：{agent_id}（版本 {version}，状态: registered）")
    print(f"下一步：python main.py evaluate {agent_id}")
    return 0


def cmd_evaluate(args) -> int:
    llm = get_llm()
    reg = _get_registry()
    agent = reg.get_agent(args.agent_id)
    if not agent:
        print(f"❌ 未找到智能体：{args.agent_id}")
        return 1
    testset = Path(args.testset) if args.testset else config.TESTSETS_DIR / "account_compliance.json"
    if not testset.exists():
        print(f"❌ 测试集不存在：{testset}")
        return 1

    print(f"▶ 评测智能体 {agent['id']}（测试集：{testset.name}）...")
    rep = evaluator.evaluate(agent, testset, llm)
    report_path = report.write_report(rep)
    reg.save_evaluation(agent["id"], agent["current_version"], rep.metrics, rep.passed, str(report_path))

    print()
    print(report.render_markdown(rep))
    print(f"报告已写入：{report_path}")

    if rep.passed:
        reg.set_status(agent["id"], "releasable")
        print(f"✅ 指标达标，状态已置为「可上线」（releasable）")
    else:
        print(f"❌ 指标未达标，保持状态「registered」，请迭代后重新评测")
    return 0


def cmd_run(args) -> int:
    llm = get_llm()
    reg = _get_registry()
    agent = reg.get_agent(args.agent_id)
    if not agent:
        print(f"❌ 未找到智能体：{args.agent_id}")
        return 1
    user_input = " ".join(args.input)
    pkg = load_package(Path(agent["agent_dir"]))
    result = run_agent(pkg, user_input, llm)
    reg.save_run(agent["id"], agent["current_version"], user_input, result)
    print(result.output)
    print(
        f"\n— {result.latency_ms:.0f}ms · {result.total_tokens} tokens · "
        f"工具 {result.tool_calls} 次（异常 {result.tool_errors}）"
    )
    return 0


def cmd_list(args) -> int:
    reg = _get_registry()
    rows = reg.list_agents()
    if not rows:
        print("（暂无已注册智能体）")
        return 0
    print(f"{'ID':<28} {'状态':<12} {'来源':<12} {'版本':<6} 名称")
    print("-" * 80)
    for r in rows:
        print(f"{r['id']:<28} {r['status']:<12} {r['source']:<12} {r['current_version']:<6} {r['name']}")
    return 0


def cmd_status(args) -> int:
    reg = _get_registry()
    agent = reg.get_agent(args.agent_id)
    if not agent:
        print(f"❌ 未找到智能体：{args.agent_id}")
        return 1
    print(f"智能体：{agent['id']}")
    print(f"  名称：{agent['name']}")
    print(f"  描述：{agent['description']}")
    print(f"  状态：{agent['status']}")
    print(f"  来源：{agent['source']}  当前版本：{agent['current_version']}")
    print(f"  目录：{agent['agent_dir']}")
    ev = reg.latest_evaluation(args.agent_id)
    if ev:
        print(f"  最近评测：{'✅ 达标' if ev['passed'] else '❌ 未达标'}（{ev['created_at']}）")
        m = ev["metrics"]
        print(
            f"    命中率 {m.get('hit_rate', 0):.2f} · "
            f"平均 {m.get('avg_latency_ms', 0):.0f}ms · "
            f"{m.get('avg_tokens', 0):.0f} tokens · "
            f"违禁率 {m.get('banned_rate', 0):.2f}"
        )
    else:
        print("  最近评测：无")
    runs = reg.list_runs(args.agent_id, 3)
    if runs:
        print(f"  最近运行（{len(runs)} 条）：")
        for r in runs:
            print(f"    - [{r['created_at']}] {r['status']} · {r['latency_ms']:.0f}ms · {r['output'][:40]}")
    return 0


def cmd_demo(args) -> int:
    print("=" * 60)
    print("开户合规智能体 —— 端到端生命周期演示")
    print("=" * 60)
    llm = get_llm()
    requirement = (
        "帮我做一个智能体：根据传入的开户信息文本，自动判断开户是否合规，"
        "输出 通过/驳回 + 理由"
    )
    agent_id = _create_agent(requirement, "开户合规审核", llm)

    print(f"▶ 4/4 评测 + 上线 ...")
    reg = _get_registry()
    agent = reg.get_agent(agent_id)
    testset = config.TESTSETS_DIR / "account_compliance.json"
    rep = evaluator.evaluate(agent, testset, llm)
    report_path = report.write_report(rep)
    reg.save_evaluation(agent_id, agent["current_version"], rep.metrics, rep.passed, str(report_path))
    if rep.passed:
        reg.set_status(agent_id, "releasable")
    print(report.render_markdown(rep))

    print("▶ 统一运行时调用 ...")
    sample = "张三，身份证110101199001011234，手机号13800138000，银行卡6222020202020202，申请开户"
    pkg = load_package(Path(agent["agent_dir"]))
    result = run_agent(pkg, sample, llm)
    reg.save_run(agent_id, agent["current_version"], sample, result)
    print(f"   输入：{sample}")
    print(f"   输出：{result.output}")
    print()
    print("✅ 生命周期闭环完成：开发 → 注册 → 评测 → 上线 → 调用")
    return 0


_COMMANDS = {
    "create": cmd_create,
    "register": cmd_register,
    "evaluate": cmd_evaluate,
    "run": cmd_run,
    "list": cmd_list,
    "status": cmd_status,
    "demo": cmd_demo,
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handler = _COMMANDS.get(args.command)
    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\n已取消")
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
