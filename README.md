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

**Belief model**
- **Structured beliefs** — dimension/key/confidence/layer with 7-factor weighted scoring
- **Cognitive triples** — subject → predicate → object, with temporal validity intervals
- **Identity semantics** — SINGLE/MULTI/EVENT cardinality, canonicalization, contradiction handling
- **Temporal lifecycle** — `valid_from`/`valid_to` intervals; a superseded value is closed, not deleted
- **Structured polarity** — `positive`/`negative`/`neutral` on the belief; a retraction is a state change, not a word-list guess at read time
- **Goal lifecycle** — `active`/`paused`/`cancelled`/`resumed`, so "gave up" then "started again" is a resumption of the same goal
- **Conflict kinds** — a self-correction closes the old state; a source conflict stays unresolved and requests clarification instead of picking a winner
- **Occurrence identity** — same object + different time is a separate event, not a merged one

**Extraction & storage**
- **Three-layer extraction** — K1 deterministic + K2 LLM semantic + K3 synthesis (Compute only)
- **Predicate canonicalization** — auxiliary and tense stripping (`was_hired_by`, `joined`, `worked_at` → `works_at`), with the raw spelling kept for audit
- **Evidence as an entity** — typed `support`/`contradict`/`verify`/`correct` links, not a blob of message ids
- **Key-collision-safe updates** — `(user_id, key)` is unique regardless of status, so a replacement never reuses the key it supersedes

**Retrieval**
- **Query-aware retrieval** — content-overlap ranking plus predicate/object matching; phrasing-independent
- **Temporal intent routing** — current / historical / all_occurrences, chosen from the question
- **Query-adaptive rendering** — enumeration queries ("what languages do they speak") lift the per-dimension cap; the character budget stays the hard limit
- **Dual-channel** — structured beliefs + session summary fallback
- **Shadow lifecycle** — single-session → shadow, multi-session → auto-promote

**Runtime**
- **Single-writer publish** — every state change is a proposal; only the Publisher commits, after checking revision / consent / scope / authority / invariants
- **Built-in LLM adapter** — one-liner startup with any OpenAI-compatible API
- **FastAPI server** — auth, CORS, health check, Swagger UI

**Tooling**
- **Benchmark funnel** — 5 stage metrics + failure attribution, not just a score
- **StateBench** — a dedicated stateful-memory benchmark with a release gate
- **769 tests** — 0 failures, core paths fully covered

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
│  │  Evidence (id, ref, source_type,          │           │
│  │  extraction_method, authority, observed_at)│          │
│  └──────────────────┬───────────────────────┘           │
│                     ▼                                   │
│  ┌──────────────────────────────────────────┐           │
│  │  IdentityResolver  (编排，不做决定)        │           │
│  │   └─ RelationReasoner  关系是什么？        │           │
│  │   └─ LifecyclePolicy   应该怎样？          │           │
│  │                                          │           │
│  │  lives_in Shanghai [2020,2026-05)         │           │
│  │  lives_in Berlin  [2026-05, ∞)            │           │
│  │    = TEMPORAL_UPDATE (旧区间被关闭，       │           │
│  │      而非删除 → "以前住哪" 仍可回答)       │           │
│  │  MULTI:  likes coffee + likes painting   │           │
│  │          = CREATE (两条共存)              │           │
│  │  EVENT:  went_to museum Mon + Fri        │           │
│  │          = CREATE (两条独立)              │           │
│  │  同 pred+obj → SUPPORT (增强置信度)      │           │
│  └──────────────────┬───────────────────────┘           │
│                     ▼                                   │
│  ┌──────────────────────────────────────────┐           │
│  │  StateTransitionProposal  (决定，未提交)  │           │
│  └──────────────────┬───────────────────────┘           │
│                     ▼                                   │
│  ┌──────────────────────────────────────────┐           │
│  │  Publisher  ← 唯一允许写库的组件           │           │
│  │  检查: revision / consent / scope /       │           │
│  │        authority / invariants             │           │
│  │  拒绝 = 数据库完全不变                     │           │
│  └──────────────────┬───────────────────────┘           │
│                     ▼                                   │
│  ┌──────────────────────────────────────────┐           │
│  │  Belief Store + BeliefEvidenceLink       │           │
│  │  (user_id, key) → subject/predicate/     │           │
│  │  object/dimension/confidence/layer       │           │
│  │  + observed_at/valid_from/valid_to       │           │
│  │  + polarity / lifecycle_state            │           │
│  │  + raw_predicate (audit)                 │           │
│  │  + 7-factor weighted scoring             │           │
│  └──────────────────────────────────────────┘           │
│  ┌──────────────────────────────────────────┐           │
│  │  Session Summaries (optional fallback)   │           │
│  └──────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   MemoryEngine.recall()                  │
│  1. Detect temporal intent → current / historical /      │
│     all_occurrences (chosen from the question)           │
│  2. Query-aware admission — lexical slice + recent floor  │
│  3. Score beliefs (content overlap, predicate, object,   │
│     layer, activity, evidence, freshness)                │
│  4. Gates: L4 watermark, confirmation, one-current-value  │
│     per single-valued slot                               │
│  5. Query-adaptive diversity cap (wider for enumerations) │
│  6. If no match → search session summaries               │
│  7. Budget-packed output                                 │
└─────────────────────────────────────────────────────────┘
```

## Benchmark Results / 评测结果

### StateBench v1.0 — the stateful-memory benchmark

LOCOMO and LongMemEval score whether an answer can be produced from a long
conversation. They cannot tell a correct state transition from a lucky
paraphrase. StateBench is a dedicated set of 200 scripted cases across seven
categories (replacement, round-trip, multi-value, contradiction, temporal
event, preference evolution, stale state), with a held-out evaluation split
and a full-context baseline.

```bash
python -c "
from mirror_memory.config.loader import load_config
from mirror_memory.llm import OpenAILLM
from mirror_memory.bench.statebench_runner import (
    run_statebench, run_full_context_baseline, render_report,
)
from mirror_memory.bench.gate import check_release_gate

cfg = load_config('config/')
cfg.llm_client = OpenAILLM(api_key='...', model='...', base_url='...')
report = run_statebench(config=cfg)
baseline = run_full_context_baseline()
print(render_report(report, baseline))
print(check_release_gate(report).summary)
"
```

| Category | mirror1 | full-context baseline |
|----------|---------|----------------------|
| contradiction | 0.867 | 1.000 |
| multi_value | 0.967 | 1.000 |
| temporal_event | 1.000 | 1.000 |
| preference_evolution | 0.667 | 0.000 |
| replacement | 0.867 | 0.000 |
| round_trip | 0.500 | 0.000 |
| stale_state | 0.650 | 0.000 |
| **overall** | **0.730** | 0.450 |

The pattern is the point: the engine beats raw context on exactly the four
categories that require state (raw transcript scores 0.000 there because
nothing in it marks which value is current), and trails on the three that are
pure recall. The architecture's claim is partially demonstrated, with the
boundary drawn.

**Release gate**: the plan's threshold is state overall ≥ 0.90 and every
category ≥ 0.85. At 0.730 the gate is **not** met, so the honest external
claim is a reproducible measurement, not a proven architectural advantage.
See [docs/benchmark-funnel.md](docs/benchmark-funnel.md) and
[docs/mirror-statebench-v1.1-fix-list.md](docs/mirror-statebench-v1.1-fix-list.md).

### Public benchmarks (historical)

Measured on an earlier engine revision; kept for continuity, not as the
current state of the code.

| Metric | Baseline | Identity Engine |
|--------|----------|-----------------|
| LOCOMO Refined F1 (1382 QA) | 0.019 | 0.084 |
| LOCOMO hit rate | 12% | 35.8% |
| LongMemEval has-answer (500 QA, DeepSeek) | 33.2% | 43.8% |

### Where the public-benchmark score is lost

`python -m mirror_memory.bench.run` splits the recall chain into measured
stages and attributes every failure to the first stage that lost it. On
LOCOMO single-hop with a working extractor: identity accuracy 1.000,
retrieval recall@5 1.000 -- retrieval was never the bottleneck. The loss is
concentrated between extraction and storage, and in the renderer's selection
of which ranked beliefs to emit.

See [docs/benchmark-funnel.md](docs/benchmark-funnel.md) for the full
analysis, [docs/ingestion-fixes.md](docs/ingestion-fixes.md) for the two
ingestion bugs those numbers hid, and
[docs/k2-semantic-classification-issues.md](docs/k2-semantic-classification-issues.md)
for the extractor defects that engine rules cannot fix.

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
# 769 passed in 9s
```

Runtime invariants (stale writes leave the database byte-identical, Publisher
gates, evidence graph, temporal intervals, polarity lifecycle) live in
`tests/test_runtime_invariants.py`, `tests/test_publisher.py`,
`tests/test_polarity_lifecycle.py` and `tests/test_temporal.py`.

## License

Apache-2.0
