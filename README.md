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
- **Cognitive triples** — subject → predicate → object, with temporal support
- **Identity semantics** — SINGLE/MULTI/EVENT cardinality, canonicalization, contradiction handling
- **Three-layer extraction** — K1 deterministic + K2 LLM semantic + K3 synthesis
- **Query-aware retrieval** — predicate/object matching, not just time-based sorting
- **Dual-channel** — structured beliefs + session summary fallback
- **Shadow lifecycle** — single-session → shadow, multi-session → auto-promote
- **Built-in LLM adapter** — one-liner startup with any OpenAI-compatible API
- **FastAPI server** — auth, CORS, health check, Swagger UI
- **349 tests** — 0 failures, core paths fully covered

## Architecture / 架构

```
┌─────────────────────────────────────────────────────────┐
│                   MemoryEngine.observe()                 │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ K1 关键词 │  │ K2 LLM  │  │ K3 综合  │              │
│  │ + 正则   │  │ 语义提取 │  │ 理解合成 │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       ▼              ▼              ▼                    │
│  ┌──────────────────────────────────────────┐           │
│  │         IdentityResolver                 │           │
│  │  (subject, predicate, object, cardinality)│           │
│  │                                          │           │
│  │  SINGLE: lives_in Shanghai → Beijing     │           │
│  │          = UPDATE (旧值 superseded)      │           │
│  │  MULTI:  likes coffee + likes painting   │           │
│  │          = CREATE (两条共存)              │           │
│  │  EVENT:  went_to museum Mon + Fri        │           │
│  │          = CREATE (两条独立)              │           │
│  │  同 pred+obj → SUPPORT (增强置信度)      │           │
│  │  relation=contradicts → CONTRADICT       │           │
│  └──────────────────┬───────────────────────┘           │
│                     ▼                                   │
│  ┌──────────────────────────────────────────┐           │
│  │  Belief Store                            │           │
│  │  (user_id, key) → subject/predicate/     │           │
│  │  object/dimension/confidence/layer       │           │
│  │  + 7-factor weighted scoring             │           │
│  └──────────────────────────────────────────┘           │
│  ┌──────────────────────────────────────────┐           │
│  │  Session Summaries (optional fallback)   │           │
│  └──────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   MemoryEngine.recall()                  │
│  1. Score beliefs (topic/predicate/object/layer/activity)│
│  2. Filter zero-relevance + diversity cap                │
│  3. If no match → search session summaries               │
│  4. Budget-packed output                                 │
└─────────────────────────────────────────────────────────┘
```

## Benchmark Results / 评测结果

### LOCOMO Refined (10 conversations, 1382 QA)

| Metric | Baseline | Identity Engine | Improvement |
|--------|----------|-----------------|-------------|
| F1 mean | 0.019 | **0.084** | **+346%** |
| Hit rate | 12% | **35.8%** | **+3x** |
| Top F1 | 0.316 | **1.000** | perfect match |

| Category | Count | F1 |
|----------|-------|-----|
| Multi-hop reasoning | 213 | 0.074 |
| Single-hop reasoning | 299 | 0.042 |
| Temporal reasoning | 68 | 0.071 |
| Open-domain knowledge | 802 | 0.103 |

### LongMemEval (500 QA, oracle dataset)

| Metric | Baseline | Identity (mimo) | Identity (DeepSeek) |
|--------|----------|-----------------|---------------------|
| Has answer | 33.2% | **50.0%** | **43.8%** |

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
# 349 passed in 3s
```

## License

Apache-2.0
