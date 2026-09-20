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
- **State revision atomics** — SQL-level atomic bump prevents stale worker writes
- **Explainability** — full provenance chain: belief → events → evidence → source traces
- **368 tests** — 0 failures, core paths fully covered

## Architecture / 架构

```
┌─────────────────────────────────────────────────────────┐
│                   MemoryEngine.observe()                 │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ K1 关键词 │  │ K2 LLM  │  │ K3 综合  │              │
│  │ + 正则   │  │ 语义提取 │  │ 去重合并 │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       ▼              ▼              ▼                    │
│  ┌──────────────────────────────────────────┐           │
│  │         IdentityResolver                 │           │
│  │  (subject, predicate, object, cardinality)│           │
│  │                                          │           │
│  │  SINGLE: 同 predicate → UPDATE 覆盖     │           │
│  │  MULTI:  同 predicate → CREATE 共存      │           │
│  │  EVENT:  同 temporal  → CREATE 去重      │           │
│  │  新 pred → CREATE                        │           │
│  │  同 pred+obj → SUPPORT (增强置信度)      │           │
│  └──────────────────┬───────────────────────┘           │
│                     ▼                                   │
│  ┌──────────────────────────────────────────┐           │
│  │  Belief Store (structured)               │           │
│  │  subject/predicate/object/cardinality    │           │
│  │  dimension/key/confidence/layer          │           │
│  │  + 7-factor weighted scoring             │           │
│  │  + 12 composite indexes                  │           │
│  └──────────────────────────────────────────┘           │
│  ┌──────────────────────────────────────────┐           │
│  │  Session Summaries (dual-channel fallback)│           │
│  └──────────────────────────────────────────┘           │
│  ┌──────────────────────────────────────────┐           │
│  │  State Revision (atomic bump)            │           │
│  │  SQL UPDATE rev = rev + 1 RETURNING      │           │
│  │  → stale worker 自动检测丢弃             │           │
│  └──────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   MemoryEngine.recall()                  │
│  1. Score beliefs (topic/layer/activity/evidence)        │
│  2. Filter zero-relevance + diversity cap                │
│  3. If no match → search session summaries               │
│  4. Budget-packed output                                 │
└─────────────────────────────────────────────────────────┘
```

## Benchmark Results / 评测结果

### LOCOMO Refined (10 conversations, 1382 QA)

| Metric | Baseline (v1) | Identity Engine | Improvement |
|--------|--------------|-----------------|-------------|
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
# 368 passed in 0.35s
```

## FAQ

| 问题 | 回答 |
|------|------|
| **记忆会越记越多、越来越慢吗？** | 不会。置信度过低的旧记忆自动衰减（decay），每个维度有条数上限。memory_full 时按置信度排序淘汰最低的。 |
| **用户说"我没说过这个"怎么办？** | 调用 `correct_claim()`：原地覆盖旧事实，保留完整溯源（before → after → correction event），`explain_claim()` 可查完整证明链。 |
| **主题越改越乱怎么办？** | `correct_claim()`：原地覆盖旧事实。IdentityResolver 判断`CONTRADICT`时自动 UPDATE 而非追加，旧值被覆盖不会叠加。 |
| **两个并发写入会丢数据吗？** | 不会。`bump_state_revision()` 使用 SQL `UPDATE SET rev = rev + 1 RETURNING`，每个写事务拿到唯一递增版本号，后提交的 worker 通过比对 revision 检测到过期后自动丢弃结果。 |
| **想完全清除某主题记忆？** | `forget()` 支持 targeted 清除：指定 topic 时只删该主题相关事实 + session 段落；不指定则全量删除。会级联删除快照和取消待执行 job。 |
| **想完全清除所有记忆？** | `forget(user_id, confirm=True)` 一键全量清除。数据量大时返回 `ttl_seconds=10` 提示延迟生效。 |
| **LLM 提取失败会丢数据吗？** | 不会。LLM 失败时自动降级到 K1 规则提取（关键词+正则），并标记 `layer=layer_1`。原文始终保存在 SessionSummary。 |
| **session 摘要存原文会不会泄密？** | 摘要可配置裁剪（`truncate_length`），`forget()` 会同步清除相关 session 段落。敏感字段（电话/地址/证件）在 K1 提取时自动检测并使用代号存储。 |

## License

Apache-2.0