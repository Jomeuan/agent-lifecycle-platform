"""开户合规检查工具（示例智能体：函数即 tool）。"""

from alip.tools import tool


@tool
def check_account_rules(text: str) -> str:
    """根据内置规则检查开户申请文本，返回发现的问题；若全部合规则返回「合规」。"""
    issues = []
    if "身份证" not in text:
        issues.append("缺少身份证信息")
    if "手机号" not in text:
        issues.append("缺少手机号")
    if "银行卡" not in text:
        issues.append("缺少银行卡信息")
    for kw in ("代开户", "洗钱", "虚假信息", "冒用他人身份"):
        if kw in text:
            issues.append(f"疑似风险: {kw}")
            break
    if not issues:
        return "合规"
    return "；".join(issues)
