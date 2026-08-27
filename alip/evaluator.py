"""评测器（Evaluator）：用数字/规则指标评测智能体，不靠 LLM 主观打分。

指标（见 Demo-Design §7）：
- 命中率：正确输出数 / 测试集总数
- 平均响应时间：Σ耗时 / N
- 平均 token 消耗：Σtoken / N
- 违禁回复率：命中违禁词库数 / N
- 工具调用成功率：工具无异常次数 / 总调用次数

指标计算全部是代码/规则。测试集、指标、阈值都存在本地 JSON 配置里、可修改。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from langchain_deepseek import ChatDeepSeek

from .runtime import AgentPackage, run_agent


@dataclass
class CaseResult:
    input: str
    output: str
    hit: bool
    banned: bool
    latency_ms: float
    total_tokens: int
    tool_calls: int
    tool_errors: int
    must_contain: list[str]
    must_not_contain: list[str]


@dataclass
class EvaluationReport:
    agent_id: str
    version: str
    testset_name: str
    metrics: dict
    checks: list[dict]
    passed: bool
    case_results: list[CaseResult]
    created_at: str


def _check(op: str, actual: float, threshold: float) -> bool:
    if op == ">=":
        return actual >= threshold
    if op == "<=":
        return actual <= threshold
    raise ValueError(f"未知比较符: {op}")


def evaluate(agent: dict, testset_path: Path, llm: ChatDeepSeek, package: AgentPackage | None = None) -> EvaluationReport:
    """运行评测并返回报告。agent 需含 id/current_version/agent_dir。"""
    ts = json.loads(testset_path.read_text(encoding="utf-8"))
    cases = ts["cases"]
    thresholds = ts.get("thresholds", {})
    banned_words = ts.get("banned_words", [])

    if package is None:
        from .runtime import load_package

        package = load_package(Path(agent["agent_dir"]))

    case_results: list[CaseResult] = []
    total_tokens = 0
    total_latency = 0.0
    correct = 0
    banned_hits = 0
    tool_ok = 0
    tool_total = 0

    for case in cases:
        r = run_agent(package, case["input"], llm)
        must = case.get("must_contain", [])
        must_not = case.get("must_not_contain", [])
        hit = all(m in r.output for m in must) and all(m not in r.output for m in must_not)
        banned = any(w in r.output for w in banned_words)

        correct += int(hit)
        banned_hits += int(banned)
        total_tokens += r.total_tokens
        total_latency += r.latency_ms
        tool_ok += r.tool_calls - r.tool_errors
        tool_total += r.tool_calls

        case_results.append(
            CaseResult(
                input=case["input"],
                output=r.output,
                hit=hit,
                banned=banned,
                latency_ms=r.latency_ms,
                total_tokens=r.total_tokens,
                tool_calls=r.tool_calls,
                tool_errors=r.tool_errors,
                must_contain=must,
                must_not_contain=must_not,
            )
        )

    n = len(cases) or 1
    metrics = {
        "total_cases": len(cases),
        "correct": correct,
        "hit_rate": correct / n,
        "avg_latency_ms": total_latency / n,
        "avg_tokens": total_tokens / n,
        "banned_rate": banned_hits / n,
        "tool_success_rate": (tool_ok / tool_total) if tool_total else 1.0,
    }

    # 阈值检查项：key, 指标名, 实际值, 阈值, 比较符, 描述
    check_specs = [
        ("hit_rate", "hit_rate", "命中率", ">="),
        ("avg_latency_ms", "max_avg_latency_ms", "平均响应时间(ms)", "<="),
        ("avg_tokens", "max_avg_tokens", "平均 token 消耗", "<="),
        ("banned_rate", "max_banned_rate", "违禁回复率", "<="),
        ("tool_success_rate", "min_tool_success_rate", "工具调用成功率", ">="),
    ]
    checks = []
    for metric_key, threshold_key, label, op in check_specs:
        threshold = thresholds.get(threshold_key)
        actual = metrics[metric_key]
        if threshold is None:
            ok = True
            threshold_display = "未设置"
        else:
            ok = _check(op, actual, threshold)
            threshold_display = threshold
        checks.append(
            {
                "label": label,
                "actual": actual,
                "threshold": threshold_display,
                "op": op,
                "ok": ok,
            }
        )

    passed = all(c["ok"] for c in checks)

    return EvaluationReport(
        agent_id=agent["id"],
        version=agent.get("current_version", "v1"),
        testset_name=ts.get("name", testset_path.stem),
        metrics=metrics,
        checks=checks,
        passed=passed,
        case_results=case_results,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
