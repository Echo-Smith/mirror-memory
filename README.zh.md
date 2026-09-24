# mirror-memory

面向 AI Agent 的结构化记忆引擎。

## 这是什么

mirror-memory 是一个通用的记忆引擎，帮助 AI Agent 在多轮对话中积累、检索和利用对用户的结构化理解。

核心抽象是 **Belief（信念）**——一条带置信度、来源层级和生命周期管理的观察记录。

## 快速开始

```python
from mirror_memory import MemoryEngine

engine = MemoryEngine(
    config_path="config/",
    llm_api_key="sk-xxx",
    llm_model="deepseek-chat",
    llm_base_url="https://api.deepseek.com",
)

# 记录对话
engine.observe(user_id="u1", session_id="s1", text="我喜欢画画")
engine.observe(user_id="u1", session_id="s1", text="上周六去了一个艺术展")

# 检索记忆
context = engine.recall(user_id="u1", query="用户的爱好是什么？")
```

## 核心能力

### 结构化信念

每条信念包含：
- **维度**（topic/preference/fact/event/goal/pattern/boundary）
- **置信度**（7 因子加权：来源、层级、证据数、情境多样性、时新性、矛盾、待澄清）
- **层级**（L1 用户陈述 → L2 已确认 → L4 自动提取 → 已拒绝）

### 三层提取

| 层 | 机制 | 触发频率 |
|---|---|---|
| K1 | 确定性（关键词 + 正则） | 每轮 |
| K2 | LLM 语义提取 | 节流触发 |
| K3 | 综合理解 | 异步 Worker |

### 查询感知检索

不是按时间排序全量渲染，而是：
1. 从查询中提取主题关键词
2. 对每条信念做多信号加权打分（主题命中 +5、确认层级 +3、活性 +2、证据 +1、时新性 +1）
3. 过滤零相关 + 维度多样性限制
4. 预算装箱输出

### 双通道

- **结构化信念**：理解"这个人是什么样的"
- **会话摘要**（可选）：记住"这个人说了什么具体的话"

## 评测结果

### LOCOMO（10 段对话，1382 道题）

| 指标 | 值 |
|---|---|
| F1 均值（全部） | 0.080 |
| F1 均值（有回答） | 0.228 |
| 命中率 | 34.6% |
| 最高 F1 | 0.870 |

### LongMemEval（500 道题）

| 指标 | 值 |
|---|---|
| 有回答率 | 33.2% |

## 安装

```bash
pip install mirror-memory                    # 核心
pip install "mirror-memory[llm]"             # 含 LLM
pip install "mirror-memory[server]"          # 含服务端
pip install "mirror-memory[server,llm,dev]"  # 开发
```

## 配置

所有领域特定内容通过 YAML 文件加载：

- `dimensions.yaml` — 维度定义
- `anchors.yaml` — 关键词锚点
- `display.yaml` — 展示标签
- `extraction.yaml` — 提取关键词和正则
- `prompts/` — LLM prompt 模板

详见 [config/SCHEMA.md](config/SCHEMA.md)。

## 测试

```bash
pytest tests/ -q
# 769 passed in 9s
```

## 许可证

Apache-2.0