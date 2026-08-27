# 智能体全生命周期管理平台（Demo）

> 轻量、敏捷、先跑通最小闭环。设计见 `Demo-Design.md`，构想背景见 `context/`。

用一个最小闭环证明三件事：
1. **智能体可以由智能体开发/编排出来**：输入一句需求 → 输出一个可运行的智能体；
2. **智能体可注册纳管**：注册、配置、版本、上线状态流转（本地 CLI 完成）；
3. **评测靠数字/指标**（不靠 LLM 主观打分）：命中率、响应速度、token 消耗、违禁率、工具调用成功率。

## 目录结构

```
alip/
  cli.py        # CLI 入口：create / register / evaluate / run / list / status / demo
  config.py     # 路径与配置
  llm.py        # DeepSeek API 客户端（OpenAI 兼容）
  tools.py      # 工具协议：@tool 装饰器 + JSON Schema 生成（预留 MCP 包装点）
  runtime.py    # 统一 ReAct 运行时（智能体无需关心）
  registry.py   # SQLite 注册中心（元信息/版本/评测/运行日志）
  dev_agent.py  # DevAgent：根据需求生成智能体「三件套」
  evaluator.py  # 评测器：数字指标
  report.py     # 评测报告渲染与落盘
samples/
  account_compliance/   # 手工编写的示例智能体（prompt.md + tools.py + agent.yaml + main.py）
testsets/
  account_compliance.json  # 测试集（含指标阈值、违禁词库）
main.py          # 入口
```

## 智能体 = 可执行包

一个智能体就是一个目录：

```
agents/<agent_id>/v1/
  prompt.md      # system prompt（设计产物）
  tools.py       # 工具函数（@tool 装饰，函数即 tool）
  agent.yaml     # 配置（模型、温度、max_steps、io schema、smoke_input）
  main.py        # 独立运行入口（平台统一提供）
```

统一运行时执行 ReAct 循环，智能体只负责「内容」。

## 快速开始

1. 安装依赖（Python 3.11+）：

   ```
   pip install -r requirements.txt
   ```

2. 配置 DeepSeek API Key：

   ```
   copy .env.example .env
   # 编辑 .env 填入 DEEPSEEK_API_KEY
   ```

3. 端到端演示（一条命令跑完整闭环）：

   ```
   python main.py demo
   ```

   等价的手工分步流程：

   ```
   # 1) DevAgent 开发智能体（生成三件套 -> 冒烟测试 -> 注册）
   python main.py create "帮我做一个智能体：根据传入的开户信息文本，自动判断开户是否合规，输出 通过/驳回 + 理由"

   # 2) 注册（也可直接注册手工编写的示例智能体）
   python main.py register samples/account_compliance

   # 3) 评测（数字指标报告 -> 达标置为可上线）
   python main.py evaluate <agent_id>

   # 4) 统一运行时调用
   python main.py run <agent_id> "张三，身份证110101199001011234，手机号13800138000，申请开户"
   ```

   其它命令：

   ```
   python main.py list           # 列出已注册智能体
   python main.py status <id>     # 查看状态与最近评测
   ```

## 生命周期状态

```
开发中(developing) → 已注册(registered) → 可上线(releasable)
```

评测不达标时保持 `registered`，回到「创建方式」迭代后重新评测。

## 评测指标（全部代码/规则计算）

| 指标 | 怎么算 |
|---|---|
| 命中率 | 正确输出数 / 测试集总数（must_contain / must_not_contain 规则校验） |
| 平均响应时间 | Σ耗时 / N |
| 平均 token 消耗 | Σtoken / N |
| 违禁回复率 | 命中违禁词库数 / N |
| 工具调用成功率 | 工具无异常次数 / 总调用次数 |

测试集、指标、阈值、违禁词库都存在 `testsets/*.json` 里、可修改。

## 本期范围（Demo）

✅ 输入需求自动生成智能体；注册/版本/评测/上线状态流转；数字指标报告；统一运行时调用。

❌ 先不做：Web 管理平台、容器沙箱、MCP server 封装（仅预留）、A2A 多智能体协同、知识库/RAG、审计登录、评估体系自动设计。

## 说明

- 生成智能体的 `agents/` 目录、SQLite 数据库与评测报告均在 `data/`，已加入 `.gitignore`。
- 运行 `create` / `evaluate` / `run` / `demo` 需要有效 `DEEPSEEK_API_KEY`；`list` / `status` / `register` 可离线使用。
