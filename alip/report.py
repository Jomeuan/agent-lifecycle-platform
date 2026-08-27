"""评测报告渲染与落盘。"""

from __future__ import annotations

from pathlib import Path

from .config import REPORTS_DIR
from .evaluator import EvaluationReport


def render_markdown(report: EvaluationReport) -> str:
    lines: list[str] = []
    lines.append(f"# 评测报告：{report.agent_id}（{report.version}）")
    lines.append("")
    lines.append(f"- 测试集：{report.testset_name}")
    lines.append(f"- 时间：{report.created_at}")
    lines.append(f"- 结论：{'✅ 达标，可上线' if report.passed else '❌ 未达标，需迭代'}")
    lines.append("")

    lines.append("## 指标")
    lines.append("")
    lines.append("| 指标 | 实际值 | 阈值 | 是否达标 |")
    lines.append("|---|---|---|---|")
    for c in report.checks:
        actual = f"{c['actual']:.3f}" if isinstance(c["actual"], float) else c["actual"]
        threshold = f"{c['op']} {c['threshold']}" if c["threshold"] != "未设置" else "未设置"
        lines.append(f"| {c['label']} | {actual} | {threshold} | {'✅' if c['ok'] else '❌'} |")
    lines.append("")

    lines.append("## 用例明细")
    lines.append("")
    for i, cr in enumerate(report.case_results, 1):
        lines.append(f"### 用例 {i} {'✅ 命中' if cr.hit else '❌ 未命中'}")
        lines.append("")
        lines.append(f"- 输入：`{cr.input}`")
        lines.append(f"- 输出：`{cr.output}`")
        lines.append(
            f"- 开销：{cr.latency_ms:.0f}ms / {cr.total_tokens} tokens / "
            f"工具 {cr.tool_calls} 次（异常 {cr.tool_errors}）"
        )
        lines.append("")
    return "\n".join(lines)


def write_report(report: EvaluationReport) -> Path:
    """把报告写入 data/reports/ 并返回路径。"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{report.agent_id}-{report.version}-{report.created_at.replace(':', '-')}.md"
    path.write_text(render_markdown(report), encoding="utf-8")
    return path
