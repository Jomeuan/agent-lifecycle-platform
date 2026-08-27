import json
import re
from datetime import datetime
from alip.tools import tool

@tool
def extract_info(text: str) -> str:
    """从开户信息文本中提取关键字段，返回JSON格式的字符串，包含name, id_card, birth_date等。"""
    # 简单正则提取
    name_match = re.search(r'姓名[：:](\S+)', text)
    id_match = re.search(r'身份证号[：:](\S+)', text)
    birth_match = re.search(r'出生日期[：:](\S+)', text)
    name = name_match.group(1) if name_match else ''
    id_card = id_match.group(1) if id_match else ''
    birth_date = birth_match.group(1) if birth_match else ''
    return json.dumps({"name": name, "id_card": id_card, "birth_date": birth_date}, ensure_ascii=False)

@tool
def validate_id_card(id_card: str) -> str:
    """校验身份证号码格式和校验位，返回'通过'或'失败: 错误原因'。"""
    # 简单校验格式：18位数字+最后一位可能为X
    pattern = r'^\d{17}[\dXx]$'
    if not re.match(pattern, id_card):
        return '失败: 身份证格式错误'
    # 计算校验位（简化，忽略地区码）
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    check_codes = '10X98765432'
    sum_ = sum(int(id_card[i]) * weights[i] for i in range(17))
    expected = check_codes[sum_ % 11]
    if id_card[-1].upper() != expected:
        return '失败: 身份证校验位错误'
    return '通过'

@tool
def validate_adult(birth_date: str) -> str:
    """根据出生日期判断是否年满18周岁，返回'通过'或'失败: 未满18周岁'。"""
    try:
        birth = datetime.strptime(birth_date, '%Y-%m-%d').date()
    except ValueError:
        try:
            birth = datetime.strptime(birth_date, '%Y%m%d').date()
        except ValueError:
            return '失败: 出生日期格式错误'
    today = datetime.now().date()
    age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    if age < 18:
        return '失败: 未满18周岁'
    return '通过'

@tool
def check_blacklist(name: str, id_card: str) -> str:
    """检查姓名和身份证号是否在黑名单中，返回'通过'或'失败: 在黑名单中'。"""
    # 模拟黑名单，实际可接入外部数据
    blacklist = [
        {"name": "张三", "id_card": "110101190001011234"},  # 示例
    ]
    for item in blacklist:
        if item['name'] == name and item['id_card'] == id_card:
            return '失败: 在黑名单中'
    return '通过'
