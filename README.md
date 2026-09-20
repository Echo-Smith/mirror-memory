# mirror-memory

> Structured memory engine for AI agents — belief lifecycle, confidence weighting, retrieval scoring, dual-channel recall.
>
> 面向 AI Agent 的结构化记忆引擎 — 信念生命周期、置信度加权、检索打分、双通道召回。

## Quick Start / 快速开始

```python
from mirror_memory import MemoryEngine

engine = MemoryEngine(
    config_path="config/",
    llm_api_key="sk-xxx",
    llm_model="deepseek-chat",
    llm_base_url="https://api.deepseek.com",
)

# Observe / 记录对话
engine.observe(user_id="u1", session_id="s1", text="I love painting landscapes")
engine.observe(user_id="u1", session_id="s1", text="I went to an art show last Saturday")

# Recall / 检索记忆
context = engine.recall(user_id="u1", query="What are the user's hobbies?")
```

## FastAPI Server / 服务端

```bash
pip install "mirror-memory[server]"
MM_LLM_API_KEY=sk-xxx MM_LLM_MODEL=deepseek-chat uvicorn mirror_memory.server:app
```

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check / 健康检查 |
| POST | `/observe` | Ingest conversation / 记录对话 |
| POST | `/recall` | Retrieve memory / 检索记忆 |
| GET | `/panel` | Structured panel / 面板数据 |
| GET | `/beliefs` | List beliefs / 信念列表 |
| POST | `/delete` | Delete memories / 删除记忆 |
| GET | `/stats` | Extraction stats / 提取统计 |
| GET | `/docs` | Swagger UI |

## Features / 特性

- **Structured beliefs** — dimension/key/confidence/layer with 7-factor weighted scoring
- **Three-layer extraction** — K1 deterministic + K2 LLM semantic + K3 synthesis
- **Query-aware retrieval** — multi-signal scoring, not just time-based sorting
- **Dual-channel** — structured beliefs + session summary fallback
- **Shadow lifecycle** — single-session → shadow, multi-session → auto-promote
- **Privacy-first** — cascade deletion, consent gating, sensitive data codenames
- **Built-in LLM adapter** — one-liner startup with any OpenAI-compatible API
- **Identity semantics** — SINGLE/MULTI/EVENT cardinality, canonicalization, contradiction handling
- **349 tests** — 0 failures, core paths fully covered

## Architecture / 架构

```
┌─────────────────────────────────────────────────┐
│              MemoryEngine.observe()              │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐ │
│  │ K1 关键词 │  │ K2 LLM  │  │ Session Snip. │ │
│  │ + 正则   │  │ 语义提取 │  │ 原文存储      │ │
│  └────┬─────┘  └────┬─────┘  └──────┬────────┘ │
│       ▼              ▼              ▼           │
│  ┌──────────────────────────────────────┐       │
│  │  Belief Store (structured)           │       │
│  │  dimension/key/confidence/layer      │       │
│  └──────────────────────────────────────┘       │
│  ┌──────────────────────────────────────┐       │
│  │  Session Summaries (optional)        │       │
│  └──────────────────────────────────────┘       │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│              MemoryEngine.recall()               │
│  1. Score beliefs (topic/layer/activity/evidence)│
│  2. Filter zero-relevance + diversity cap        │
│  3. If no match → search session summaries       │
│  4. Budget-packed output                         │
└─────────────────────────────────────────────────┘
```

## Benchmark Results / 评测结果

### LOCOMO Refined (10 conversations, 1382 QA)

| Metric | Value |
|--------|-------|
| F1 mean (all) | 0.080 |
| F1 mean (answered) | 0.228 |
| Hit rate | 34.6% |
| Top F1 | 0.870 |

### LongMemEval (500 QA, oracle dataset)

| Metric | Phase 1 (baseline) | Phase 1-2 (identity) | Improvement |
|--------|-------------------|---------------------|-------------|
| Has answer | 33.2% | **50.0%** | +50% |

See [docs/benchmarks.md](docs/benchmarks.md) for detailed results and comparison.

## Installation / 安装

```bash
# Core / 核心
pip install mirror-memory

# With LLM support / 含 LLM 支持
pip install "mirror-memory[llm]"

# With server / 含服务端
pip install "mirror-memory[server]"

# Development / 开发
pip install "mirror-memory[server,llm,dev]"
```

## Configuration / 配置

All domain-specific content is loaded from YAML files in `config/`. See [docs/config-guide.md](docs/config-guide.md).

## API Reference / API 参考

See [docs/api-reference.md](docs/api-reference.md).

## Testing / 测试

```bash
pip install "mirror-memory[dev]"
pytest tests/ -q
# 248 passed in 0.35s
```

## License

Apache-2.0