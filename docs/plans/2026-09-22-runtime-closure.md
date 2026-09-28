# Mirror Runtime Closure 实施计划

## 目标

把当前并排存在的 Belief、Evidence、Temporal、Proposal/Publisher 和 Retrieval 原语，收敛为
一条可审计、可并发、可迁移的 Epistemic Runtime 主链：

```text
Observation -> Claim -> Relation -> Proposal -> Atomic Publish
            -> Versioned Belief State -> Governed Projections
```

本计划记录 Runtime Closure 的实施单元。发布门先聚焦状态区间/极性与 Publisher 安全；
Observation/Assertion 拆分、完整 RecallQuery、Hybrid Retrieval 和 Maintenance 保留为演进方向。
bugfix 已合入 `main`；当前代码复核见
`docs/epistemic-runtime-analysis-2026-09-24.md`。

## 当前基线

- 代码基线：`main`，`5393053`；分析前工作区干净；
- 完整测试：769 passed，1 skipped，1 个第三方 deprecation warning；
- 本轮 gate/CLI/测试修改后：776 passed，1 skipped，1 个第三方 deprecation warning；
- K1/K2 同槽切片后（2026-09-25）：**788 passed，1 skipped**，零回归；触及文件
  Ruff 无新增；
- 撤回 + 目标生命周期切片后（2026-09-25）：**800 passed，1 skipped**；
- **测量完整性守门（2026-09-25）**：forced/production 在 K2 调用数为 0 或全部失败时
  直接 `RuntimeError`；报告带 `extraction_health`（k2 尝试/失败数）。此前
  `OpenAILLM.generate` 与 `SemanticExtractor` 两处吞错会让 429 期间的 forced 运行
  静默退化为 K1 数字。每次复测必须先看 `k2 calls` 行。
- Ruff：63 errors（全仓，均为既有项）；
- StateBench v1.1 deterministic：state 25/200，recall 2/200；
- 同槽切片后 deterministic：state **38/200 = 0.190**，recall **46/200 = 0.230**；
  两个 pilot case（replacement、round-trip）state 与 recall 均转通过；
  8 例 state 回退均为此前靠 K1 keyword 行文本蒙混通过的 case，明细见分析文档第 9 节；
- StateBench v1.0 模型运行的 0.730 是历史测量；完整 v1.1 forced/production 结果尚未建立。
- **v1.1 全量成对已完成（2026-09-25，MiMo-V2.6-Flash，200 例，同 commit/数据集/配置）**：
  forced state 148/200 = 0.740、recall 168/200 = 0.840；production state 149/200 = 0.745、
  recall 168/200 = 0.840；gap -0.005（通过）。发布门 FAIL：state overall 与 4 个分类
  （contradiction 0.40、preference_evolution 0.50、round_trip 0.767、stale_state 0.75）
  低于阈值，recall 差 2 例。52 个 state 失败中 49 例为状态模型失败（旧当前值未关闭），
  详见分析文档第 10 节。
- v1.1 报告与原有 release gate 的类型契约不兼容；本轮工作区已修复，并要求完整 200 例覆盖。
- 使用 `nex-agi/nex-n2.5-pro:free` 完成 2 个 forced/production 配对 case：state 双方均 0/2，recall 双方均 2/2；模型端随后 HTTP 520、超时，未获得完整 v1.1 结果。
- 后续实施单元保持小而独立，避免格式化或夹带无关文件，确保每一层都能单独回滚和对比基准结果。

## 已完成切片：K1/K2 同槽归并与当前值资格（2026-09-25）

对应执行顺序第 3 项中的同槽部分。改动与验收：

- `PatternRule` 增加 `predicate` / `object_group`；`extraction.yaml` 为 age、name、
  i_live、moved_to（新）、i_want、like、dislike（两条）声明谓词；
  `deterministic.py` 对声明谓词的 pattern 产出三元组，key 为 `predicate:object`。
- 无三元组 claim 不再落为 Belief 行，观测仍写入 Evidence。
- `PATTERN_CONFIDENCE` 0.5 → 0.6（高于 L4 渲染水印 0.55）。
- `resolve_identity`：相反极性关于同一 object 的断言替换当前值。
- 验收：K1 pattern 行与 K2 claim 同槽归并（`tests/test_integration.py::TestK1K2SameSlot`）；
  副词与否定句式抽取（`test_extraction.py`）；极性替换与共存（`test_relation_lifecycle.py`）；
  无三元组轮次只留观测（`test_evidence_graph.py`）。

## 执行顺序（2026-09-24 修订）

下面的“实施单元 1–7”保留为设计与验收清单，编号不代表实际排期。

1. **测量准备**：修复 v1.1 gate 和 forced/production 比较；把所有 benchmark
   `transition_type` 映射到拟议 runtime 操作。代码中的
   `TRANSITION_RUNTIME_CONTRACT` 是诊断映射，不表示这些 intent 已实现。
2. **先测 forced + production**：在稳定且额度足够的端点上，同一 commit、case、数据集、模型、配置与回答器下运行；
   保存两份 manifest、按 transition 的 state/recall/answer 与失败漏斗。forced 总分给出当前
   端到端上限；抽取与状态的责任要靠逐回合 candidate、identity 和 persistence 记录拆分。
   v1.1 全量是每个 case 多轮抽取，不是单次模型请求。
3. **状态正确性优先**：先完成 CREATE 独立时间列、current interval、REVIVE 新 interval、
   polarity/END_CURRENT、`at_time` 读取和 K1/K2 同槽处理；把最小 schema migration 与版本表一起上线。
   版本容量策略是这个单元的前置设计条件，见实施单元 5。K1/K2 同槽归并、当前值资格与
   偏好极性替换已于 2026-09-25 落地（见上文“已完成切片”）；剩余 CREATE 时间列、
   END_CURRENT、版本化 interval 与 `at_time` 读取未完成，K1 仍缺 termination、
   correction、relationship、goal resume 与 event occurrence 句式。
4. **写入安全**：完成实施单元 1 的 CAS、幂等、savepoint/rollback 与证据校验；用并发和
   故障注入测试验收，单线程 StateBench 不衡量此项收益。
5. **迁移写入口**：按 pipeline → API → verification 顺序迁移至 Publisher，再清理旁路。
6. **复测与决策**：对同一 v1.1 forced/production 集复测；剩余错误按 transition 与漏斗
   归因，再决定继续修抽取、状态或召回。实施单元 6–7 的完整治理和检索重构随后推进。

## 下一切片（由 2026-09-25 成对实测决定）

forced 上限 0.740 的瓶颈已定位为 END_CURRENT/CORRECT 转移语义：52 个 state 失败中
49 例是“值已抽出但旧当前值未关闭”。按失败数排序的下一批工作：

1. **显式纠正与中性谓词替换**（explicit_correction 0/4、relationship_termination 0/3）：
   correction 轮产生新谓词 belief 而原 positive 行保持 active；需要 CORRECT 意图与
   “同一 object 的后续否定断言关闭旧行”的规则（现有极性机制只覆盖 likes/dislikes）。
   **已实现（2026-09-25）**：撤回分支（`SELF_CORRECTION`/`END_CURRENT`）+ token 指代
   查询，见分析文档第 11 节；forced 复测待完成。
2. **目标生命周期**（goal_cancellation 1/8、goal_reactivation 2/10）：multi 基数下
   lifecycle 状态变化目前只走 SUPPORT，旧阶段不关闭；forced 模式暴露出 deterministic
   隐藏的该缺口。**已实现（2026-09-25）**：通用目标指代回落（"I gave up on the goal"
   按目标族关闭当前目标）+ 重启分支（`is_resumption` 关闭被标记
   `transition=END_CURRENT` 的行）+ END_CURRENT 行打标。单测覆盖；**尚未复测**
   （端点 429 窗口内两次降级运行无效，额度恢复后重跑 forced）。
3. **往返复活关闭中间值**（current_value_revival 23/30）：复活路径已关闭旧行，但
   K2 产生的中间值行有部分未被关闭，需按 revival 语义统一。
4. **event occurrence 身份**（distinct_occurrences 2 例失败）：`_event_key` 的哈希
   后缀在部分路径上不一致地附加，破坏 occurrence 计数。
5. CREATE 独立时间列与版本化 interval（含容量界设计）按原计划跟进。

Publisher 原子化（CAS/幂等/回滚）与写入口迁移顺序不变，仍在状态切片之后；
Hybrid Retrieval 继续推迟——recall 0.840 的失败主要是状态失败的下游。

当前 deterministic 轨有 175 个 state 失败，五个零分 transition 合计 108 例：
`current_value_replacement` 30、`current_value_revival` 30、`preference_reversal` 20、
`query_time_selection` 20、`goal_cancellation` 8。`goal_reactivation` 10/10 和
`explicit_correction` 3/4 构成对照，但不能代替 forced 测量。
小样本实测发现：K2 已正确 supersede 旧 residence 值时，K1 pattern/keyword 仍可留下
同义的 active 行，使 replacement/revival state contract 失败。版本化必须连同这些行的
current 资格一起设计，不能只改 K2 的 interval。

## 已确定的架构决策

1. `Belief` 暂时保留为兼容 read model；历史真相逐步迁到 versioned ledger。
2. 所有 belief、snapshot 和 projection state mutation 必须经过 Publisher。
3. Evidence/Observation 可以在决策前记录，但不可原地改变原始来源含义。
4. Publisher 负责一次 transition 的 revision、事务、事件、evidence edge 和 outbox 原子性。
5. valid time、epistemic state、retention state 分轴存储。
6. Graph、embedding、snapshot、summary 都是可重建 projection，不是 truth source。
7. round trip 创建新的 belief version，不复活并覆盖旧 version。
8. explicit correction、state termination 和 unresolved conflict 是不同 transition intent。
9. Hybrid Retrieval 必须消费 `RecallQuery`，并在融合后经过统一 governance gate。
10. 公开版本使用正式 schema migration；`create_all()` 只负责全新数据库和测试数据库。

## 完成标准

### 写入边界

- 生产入口不直接调用 belief/snapshot raw mutator；
- 每个 mutation 都有 proposal id、expected revision、actor、evidence 和 publish receipt；
- stale proposal、重复 proposal、缺失/越权 evidence 均被确定性拒绝；
- handler 任意位置失败后没有部分写入；
- 每个成功 proposal 只增加一次 state revision。

### 状态正确性

- CREATE 与 UPDATE 都保存 subject、polarity、observed_at、valid_from、valid_to、scope；
- current-state slot 在同一 valid time 最多一个 current version；
- A -> B -> A 保留三个 interval，最后只有 A 为 current；
- 同一 preference object 同一时间最多一个 current polarity；
- event identity 包含 occurrence time 或原始 qualifier fingerprint；
- correction、termination、conflict 的状态和事件可区分。

### 治理与投影

- targeted forget 后 belief read model、summary、snapshot、semantic index 和 graph projection 都
  不可再召回目标内容；
- summary fallback 不绕过 validity、rejection、authority、consent 与 forget filter；
- 所有 projection 可从 ledger 重建；
- 上一公开 schema 的数据库可迁移到新 schema。

### 评测

- 每个实施单元通过相关测试，最终完整测试无回归；
- forced StateBench state overall >= 0.90，每分类 >= 0.85；
- production 与 forced state 差距 <= 0.05；
- recall overall >= 0.85 后再对外宣称 Hybrid Retrieval 能力；
- 每次基准结果带 manifest，不用 deterministic 结果代替生产结论。

## 实施单元 1：冻结写入协议

### 目的

先让 Proposal/Publisher 成为可靠协议，再把流量迁入。这个单元不改变 Belief 业务语义。

### 变更

1. 扩展 `StateTransitionProposal`：
   - `proposal_id`；
   - durable `idempotency_key`；
   - `expected_revision: int`，belief mutation 不允许缺失；
   - `actor_type` 与 `actor_id`；
   - evidence refs/ids 的明确类型；
   - schema version。
2. 增加 publish receipt/transition commit 表：
   - proposal/idempotency 唯一约束；
   - accepted/refused 状态；
   - revision before/after；
   - refusal reason；
   - transition event id。
3. Publisher 使用一个事务完成：
   - consent/feature consent 检查；
   - evidence 完整性与 user scope 检查；
   - atomic expected-revision claim；
   - raw mutation；
   - BeliefEvent、EvidenceEdge、receipt/outbox；
   - 单次 revision bump。
4. commit handler 放在 savepoint/transaction 中；异常必须 rollback 后再返回 decision。
5. repository raw mutator 支持由 Publisher 管理 revision，避免每个 helper 自行 bump。

### 涉及组件

- `src/mirror_memory/core/proposal.py`
- `src/mirror_memory/core/publisher.py`
- `src/mirror_memory/core/repository.py`
- `src/mirror_memory/core/models.py`
- `tests/test_publisher.py`
- `tests/test_runtime_invariants.py`

### 验证

- 两个 session 对同一 revision 提交，只有一个成功；
- handler 在 belief flush 后、event/edge flush 前故障，事务完全回滚；
- 同一 idempotency key 重放只返回原 receipt；
- evidence id 缺失、重复、跨 user 均有稳定结果；
- success proposal 的 revision 只加 1。

## 实施单元 2：迁移所有生产写入口

### 目的

让 single-writer 从文档目标变成实际约束。

### 变更

1. Extraction Pipeline：
   - Identity/Lifecycle 只生成 proposal；
   - SUPPORT、UPDATE、CONTRADICT、CREATE 全部调用 Publisher；
   - evidence edge 由 Publisher 在同一 mutation transaction 中建立；
   - trace 增加 proposal id、publish decision 和 refusal reason。
2. Public API：
   - `correct()`、`forget_belief()`、verification confirm/reject 改走 Publisher；
   - user full delete 使用独立治理 command，但仍生成 receipt/audit result；
   - `set_enabled()` 保留治理入口，并使在途 proposal 立即 stale。
3. K3/Snapshot 继续使用同一 Publisher，不保留特殊旁路。
4. 将 repository raw mutator 收到 internal namespace，公开 API 不再导出。
5. 增加静态边界检查：生产模块中禁止直接引用 raw mutator。

### 涉及组件

- `src/mirror_memory/extraction/pipeline.py`
- `src/mirror_memory/extraction/verification.py`
- `src/mirror_memory/api.py`
- `src/mirror_memory/worker/evolution.py`
- `src/mirror_memory/core/__init__.py`
- `tests/test_extraction.py`
- `tests/test_verification.py`
- `tests/test_api.py`
- `tests/test_runtime_invariants.py`

### 验证

- observe/correct/forget/verify/synthesize 都能从 trace 或 receipt 找到 proposal；
- memory disabled、feature consent revoked、stale revision 在每个入口表现一致；
- `rg` 边界测试确认生产入口没有直接 mutation import；
- 原有公开 API 返回值保持兼容。

## 实施单元 3：建立 schema migration 基础

### 目的

在增加版本化表和字段前，先解决已有数据库无法升级的问题。

### 变更

1. 引入 schema version 表与正式 migration runner；
2. 为当前新增的 Evidence、Temporal、Proposal/Receipt 结构补首个升级 migration；
3. `MemoryEngine._ensure_db()`：
   - 空数据库创建最新 schema；
   - 已有数据库运行有序 migration；
   - schema 高于当前代码版本时拒绝启动；
4. migration 支持 SQLite；其他已声明支持的 SQLAlchemy dialect 不使用 SQLite 专属 SQL；
5. 备份、失败恢复和幂等重跑写入运维文档。

### 涉及组件

- `pyproject.toml`
- `src/mirror_memory/api.py`
- 新增 `src/mirror_memory/migrations/`
- `src/mirror_memory/core/models.py`
- 新增 migration tests

### 验证

- 从当前 main schema 构造真实旧库，升级后数据与 revision 不丢失；
- migration 中途失败后可回滚或安全重跑；
- 新库与升级库的 schema inspection 结果一致；
- `create_all()` 不再被当作旧库升级机制。

## 实施单元 4：补齐 Claim 结构与 CREATE 语义

### 目的

让首条状态就拥有完整的身份、极性和时间信息。

### 变更

1. Candidate/Claim 增加：
   - canonical subject id；
   - canonical predicate 与 raw predicate；
   - object identity 与 surface text；
   - explicit polarity；
   - observed_at、valid_from、valid_to、raw temporal qualifier；
   - transition intent：assert/correct/end/dispute；
2. K2 schema/prompt 和 deterministic extraction 产出相同结构；
3. CREATE proposal 与 Publisher 完整保存上述字段；
4. RelationReasoner 实际比较 subject，不再固定 `same_subject=True`；
5. 对无法解析的时间保留 qualifier，并生成稳定 occurrence fingerprint；
6. 明确区分：
   - negative current assertion；
   - state termination；
   - explicit correction；
   - unresolved source conflict。

### 涉及组件

- `src/mirror_memory/memory/atom.py`
- `src/mirror_memory/memory/relation.py`
- `src/mirror_memory/memory/lifecycle.py`
- `src/mirror_memory/extraction/semantic.py`
- `src/mirror_memory/extraction/deterministic.py`
- `config/prompts/k2_system.txt`
- `src/mirror_memory/core/models.py`
- `src/mirror_memory/core/publisher.py`

### 验证

- 通过公开 `observe()` 创建的第一条 location belief，独立列中含完整 interval；
- 不同 subject 的相同 predicate 不合并；
- “不再喜欢”“刚才说错”“另一来源说不是”产生三个不同 transition；
- 同事件同时间重复为 SUPPORT，不同时间为 CREATE。

## 实施单元 5：引入版本化状态账本

### 目的

解决 `(user_id, key)` 单行模型无法表达 round trip 和多阶段历史的问题。

### 数据模型

新增最小版本账本：

```text
StateSlot
  id, user_id, subject_id, predicate_family, scope, slot_key

BeliefVersion
  id, slot_id, version_no
  object_id, object_text, polarity
  epistemic_state, confidence
  valid_from, valid_to
  recorded_revision, closed_revision
  transition_event_id
```

Slot identity 由 policy 生成：

- current state：subject + predicate；
- preference/goal：subject + predicate family + object；
- event：subject + predicate + object + occurrence fingerprint；
- persistent relation：subject + predicate + object。

### 容量边界与历史精度

版本化上线前必须明确每个 slot 的精确 interval 上限 `N`、年龄窗口、每用户总预算和
超界处理方式。候选策略是最近 `N` 个 interval 精确保留，较旧区间压缩为有大小上限的
归档摘要；压缩结果标记精度，`at_time` 查询不得把摘要误当作精确版本。

`N` 和历史查询精度需要作为产品契约确定。若要求任意远期时间点都能精确回答，则
仅靠丢弃或合并旧 interval 无法给出有界存储；必须另设有成本预算的冷存储或缩短精确
历史承诺。压缩还必须保留 targeted forget 的 lineage，且不可改变当前版本。测试覆盖
超过上限、多次 A → B → A、压缩后的查询和定向遗忘。

### 迁移策略

1. 新表上线，现有 `Belief` 继续提供 read compatibility；
2. backfill：每个现有 Belief 生成一个 slot/version，并记录 provenance limitation；
3. Publisher dual-write ledger 与 Belief projection；
4. 增加一致性检查，比较 ledger materialization 与 Belief read model；
5. recall 切换到 ledger-backed read projection；
6. 删除“复活旧 row”逻辑；
7. 修改当前要求复用旧 row id 的 regression test，改为断言三个 interval 与一个 current version。

### 涉及组件

- `src/mirror_memory/core/models.py`
- `src/mirror_memory/core/publisher.py`
- `src/mirror_memory/core/repository.py`
- `src/mirror_memory/memory/identity.py`
- `src/mirror_memory/memory/lifecycle.py`
- migration/backfill scripts
- `tests/test_longmemeval_regressions.py`
- `tests/test_temporal.py`
- `tests/test_statebench_v1_1.py`

### 验证

- Shanghai -> Berlin -> Shanghai 产生 3 versions；
- current query 只返回最后一个 Shanghai；
- history query 返回第一个 Shanghai 和 Berlin；
- backfill 前后非 round-trip 用户的公开 recall 不变；
- current projection 可完全由 ledger 重建。

## 实施单元 6：统一 Forget 与 Projection Invalidation

### 目的

消除 summary fallback、snapshot、semantic index 或 graph 重新暴露已忘记内容的风险。

### 变更

1. 定义 `ForgetScope`：belief version、slot、observation/session、user；
2. Publisher 写入 forget transition/tombstone，并生成 projection invalidation outbox；
3. Belief、summary、snapshot 和未来索引使用同一 content lineage；
4. recall governance gate 在所有 channel 上执行 tombstone/consent/authority 检查；
5. full user delete 保持物理删除，并保存不含用户内容的操作回执；
6. projection consumer 幂等处理 invalidation。

### 涉及组件

- `src/mirror_memory/core/publisher.py`
- `src/mirror_memory/core/repository.py`
- `src/mirror_memory/core/models.py`
- `src/mirror_memory/render/renderer.py`
- `src/mirror_memory/worker/`
- `tests/test_evidence_graph.py`
- `tests/test_repository.py`

### 验证

- targeted forget 后 structured、summary fallback、snapshot 三个现有通道都不返回目标；
- outbox 重放不产生错误或二次删除；
- forget 一个 belief 不误删同一 observation 支持的其他 belief；
- user full delete 的所有内容表计数归零。

## 实施单元 7：建立 RecallQuery/RecallResult

### 目的

先建立稳定检索契约，再接 embedding 和 graph。

### 接口

```text
RecallQuery
  user_id, subject_ids
  predicates, objects, free_text
  temporal_mode: current | history | at_time | all_occurrences
  at_time
  epistemic_floor
  include_disputed
  authority_floor
  top_k, context_budget

RecallResult
  items[]
    belief_version_id, claim surface, interval
    relevance_score, epistemic_score
    channel_scores, evidence refs, decision reasons
  packed_context
  omitted[] + omission reason
```

### 变更

1. 把 temporal intent/predicate extraction 移到 query compiler；
2. repository 只提供 structured/lexical candidate source；
3. retrieval 模块负责 rank/fusion；
4. governance gate 统一过滤 validity、rejected、forgotten、authority 和 consent；
5. context packer 负责 diversity 与预算；
6. renderer 只把 RecallResult 转成文本；
7. `MemoryEngine.recall()` 保持返回文本，同时增加结构化 recall API。

### 验证

- current/history/at_time/all_occurrences 各有端到端用例；
- omission reason 能区分未召回、被治理过滤、预算丢弃；
- summary source context 明确标注，不能冒充 accepted belief；
- 现有字符串 recall API 保持兼容。

## Runtime Closure 总体验收

本轮发布门先验收上述执行顺序 1–6；完整 Runtime Closure 继续以实施单元 1–7 为准。
每个阶段执行对应检查，最终执行：

1. 完整 pytest；
2. Ruff，并把本次触及文件清零；
3. schema upgrade/backfill tests；
4. concurrency、rollback、idempotency tests；
5. StateBench deterministic 作为结构回归；
6. StateBench forced 与 production 作为发布门槛；
7. manifest、失败 case 和完整 funnel 归档。

只有 P0 通过后，才开始下面两项：

## P1：Hybrid Retrieval

1. embedding 作为异步 projection；
2. lexical、semantic、structured/temporal 并行召回；
3. 使用 RRF 或经过标定的融合，不把不同 channel 的原始分数直接相加；
4. epistemic score 不参与候选相关性计算，只在融合后做可信度排序/过滤；
5. index failure 不阻断主写入，落后时可重放 outbox 重建；
6. 用 StateBench recall track、LOCOMO context coverage 和 answer track 分开验收。

## P2：Maintenance Runtime

1. stale/conflict/duplicate/interval/snapshot drift scanner；
2. scanner 只创建 proposal；
3. proposal 继续经过 Relation/Lifecycle/Publisher；
4. maintenance action 有独立 actor、reason、evidence 和预算；
5. 用户 verification 优先于自动 truth change。

## P3：Retention Lifecycle

1. active/cooling/dormant/archived 独立于 epistemic/validity state；
2. 生命周期只改变召回成本与默认可见性；
3. 用户 Hard Forget 继续走治理删除；
4. 生命周期策略通过离线 replay 评估后再启用自动迁移。

## 建议立即开始的切片

先在可稳定完成 200 例的端点上补齐 forced/production 成对基线；本轮免费端点仅留下
2 个有效配对，无法据此判断全量瓶颈。并行设计 interval 容量界、迁移与 K1/K2 同槽规则，
再实现上述“状态正确性优先”切片。迁移 Pipeline 流量前必须完成“实施单元 1：冻结写入
协议”的 CAS、幂等和回滚验收。Hybrid Retrieval 保持独立演进。
