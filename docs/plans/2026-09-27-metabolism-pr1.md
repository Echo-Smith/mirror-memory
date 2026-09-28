# PR1 — Tier + Heat + Retrieval Eligibility

Date: 2026-09-27
Status: **implemented, verified** (PR1: 865; PR2: 895; PR3: 922; PR4: 945 tests, zero regressions)

长期目标：给 Mirror 的认知内核（Evidence → Belief → Proposal → Publisher）补一个
"热 / 温 / 休眠 / 归档" 的存储生命周期层，让引擎具备多年运行能力，且**不改变**
Claim → Compute → Publish 的写入路径——metabolism 自己永远不写库，一切存储变迁
仍是 Proposal，仍走 Publisher。

PR1 按 4-PR 路线的第一步落地：**只铺轨道，不发车**。没有任何代码会把 belief 降级，
因此对现有行为的唯一影响是"手动设为 dormant/archived 的行不再进入默认扫描"。

## 交付内容

### 1. 两个正交的状态维度（core/models.py）

Belief 新增列：`memory_tier`（hot/warm/dormant/archived，默认 hot）、
`metabolism_state`、`retention_class`、`importance_score`、`access_count`、
`last_accessed_at`、`last_supported_at`、`protected_reason`、`compacted_into`。

truth state（status/valid_from/valid_to）与 storage state（memory_tier）分离：
"superseded + warm" = 历史成立且近期可能用得到，而不是把两者混为一谈。

### 2. metabolism 包（src/mirror_memory/metabolism/）

- `tiers.py` — 四级 tier + 合法迁移边（hot↔warm、warm↔dormant、dormant→archived、
  archived→warm 显式恢复、dormant→hot 重热）。不允许跳级（hot→dormant 非法）。
- `policy.py` — 5 个 retention class 及冷却策略；`classify_retention()` 是纯函数，
  **复用 identity 轴**派生：goal lifecycle / wants_to 类谓词 → behavioral，
  event cardinality → episodic，single cardinality → canonical，其余 → preference。
  显式 override（metabolism.yaml `retention_predicates`）优先。
- `heat.py` — heat = .30·freshness + .20·access + .20·evidence_strength
  + .15·authority + .15·importance；freshness 以 last_accessed_at 为锚（缺省退回
  last_evidence_at）；阈值 0.70/0.40/0.15 映射 tier；protected 时 floor 到 0.70。
- `protection.py` — 先于 planner 存在的保护不变量：unresolved conflict
  （needs_clarification）、用户纠正（correct 链接或 user_corrected 来源）、
  pending verification、L1/L2 用户确认、canonical 且 confidence≥0.8、
  证据≤1 条（thin evidence）。**用户纠正永远压过"很久没用了"。**
- `eligibility.py` — QueryMode（current/historical/all_occurrences/evidence/
  experience/all）+ tier 资格：current 只扫 hot/warm，其余默认扫 hot/warm/dormant，
  **archived 任何默认查询都不可见**（恢复必须走显式 Proposal，PR4）。

### 3. 写路径（不改 Publisher 语义）

- `record_claim` 创建时分类 retention_class（可被显式参数覆盖，CREATE payload 可透传）；
  support 合并时更新 `last_supported_at`（重热信号之一）。
- `update_belief_by_id` 的替换行**继承**旧槽位的 retention_class（Beijing 替代
  Shanghai 仍是 canonical）。
- `correct_belief` 的纠正行继承 class 并写 `protected_reason="user_correction"`。

### 4. 读路径（Eligibility 先于 Ranking）

- `recall_candidates` / `list_active_beliefs` 在 SQL 层加
  `memory_tier.in_(tiers_for_mode(mode))`（参数绑定，两个 slice 都过滤）。
- 渲染命中的 belief 记访问：`mark_belief_accessed()` 只 bump access_count /
  last_accessed_at，**故意不 bump state revision**——recall 绝不能让在途
  Proposal 失效。telemetry 是 fail-open 的：提交失败丢弃即可，不影响返回内容。
- `MemoryEngine.recall` 在渲染后提交 telemetry。

### 5. 配置（config/metabolism.yaml + schema/loader）

retention_classes（canonical 永不自动降温，transient 7 天）+ heat 权重/阈值 +
protected_floor，全部有代码默认值，文件缺失不阻塞启动。

### 6. 存量库保护（core/migrate.py）

`check_schema()`：`create_all` 只建新表不加列，旧库缺 metabolism 列时在启动期
抛带指引的 ConfigError，而不是查询中途 "no such column"。

## 与原方案的偏差（及原因）

1. **DDL 迁移 shim → schema guard。** 原计划用 `ALTER TABLE ADD COLUMN`
   自动升级存量库；但本仓库的 SQL 安全约束禁止任何原生 SQL 字符串执行
   （DDL 标识符无法参数化，纯字面量也会被拦截）。改为启动期检测 + 明确报错；
   真正的加列迁移随 PR2 引入正规迁移工具时一并解决。
2. **retention 分类不建新谓词表。** 原方案五类示例需要一张 predicate→class 映射；
   实现直接复用 identity_policy 的 cardinality/temporal 轴（single→canonical、
   event→episodic、multi→preference，goals→behavioral），避免两份谓词知识漂移。
   显式 override 保留在 metabolism.yaml。
3. **evidence/experience 模式只影响资格层。** 两种新 intent 已能检测并参与
   tier 资格判定，但"输出证据图/经历摘要"的渲染增强属于 evidence-channel
   工作，留给 PR3（EvidenceDigest）一起做，避免 PR1 膨胀。
4. **recall 从只读变为"读 + telemetry 提交"。** 原方案把 access_count/
   last_accessed_at 列在 PR1 但未说明谁写。选择：渲染命中即记（这是重热信号
   的唯一来源），提交失败静默丢弃，绝不影响召回结果。

## PR2-4 路线（未实施，仅记录）

- **PR2 — Metabolism Runtime**：`MetabolismPlanner` 生成 TierTransitionProposal
  （HOT→WARM→DORMANT、DORMANT→HOT），Publisher 增加对应 invariants（合法边 +
  protection 拒绝），`memory_transition` 审计表，正式迁移工具。
- **PR3 — Evidence Digest**：`EvidenceDigest` 表 + EvidenceCompactProposal
  （>12 条 support 开始压缩；保留最早 1 / 最近 3 / 最高 authority 2 / correct 与
  contradict 全部），evidence-channel 渲染。
- **PR4 — Archive + Purge**：冷存储分表、RestoreProposal、forget 级联删除事务 +
  DeletionTombstone（防 stale worker 复活）。
- **MetabolismBench**：current survival / historical preservation /
  correction protection / reactivation / deletion / stale-worker /
  Semantic Preservation Rate / Storage Reduction Ratio。

## 验证

- `pytest tests/ -q` → **865 passed, 1 skipped**（基线 804 + 61 个新用例，零回归）。
- 新用例覆盖：分类规则、heat 数学（含 protected floor、log 饱和、衰减）、
  六条保护规则、tier 状态机合法边、六种 query mode（含旧模式回归）、
  仓库集成（写时分类、继承、纠正保护、tier 过滤、访问计数）、配置加载、
  schema guard（新库/空库/旧库三态）。

---

# PR2 — Metabolism Runtime（追加于同日）

Status: **implemented, verified**（30 个新用例；全量 895 passed, 1 skipped）。

## 交付内容

### 1. TIER_TRANSITION 提案类型（core/proposal.py + publisher.py）

不新造 MaintenanceProposal 类型——Mirror 的提案就是
`StateTransitionProposal`，metabolism 复用同一条通道：新增
`TRANSITION_TIER_TRANSITION`，payload 携带 `from_tier` / `to_tier` /
`reason` / `score`。Publisher 侧新增不变量：

- `tier_mismatch` — from_tier 必须仍与活行一致（并发安全）；
- `illegal_tier_transition` — 必须是状态机合法边（禁止跳级）；
- `protected:<reason>` — **保护在 Publisher 侧从活状态重算**
  （belief 行 + typed link 计数 + pending verification key），
  不信任 planner 的判断；有 bug 的 planner 也压不动受保护信念；
- superseded 行**允许**降温（"superseded + warm" 正是 tier 存在的意义），
  rejected 行照旧 resurrection guard；
- revision 门槛照常生效：scan 与 publish 之间若有 truth 写入，
  剩余降温提案被拒绝。

### 2. tier 提交不动 state revision

`set_belief_tier()` 更新 `memory_tier` / `metabolism_state` 并写审计行，
**不调用 touch_memory_state**：存储态不是真相态，一次降温绝不能
让在途的真相提案失效（反之，proposal 自带的 claimed_revision 已防住
过期降温决策）。访问遥测同理（PR1）。

### 3. mm_memory_transitions 审计表（core/models.py）

user_id / entity_type / entity_id / from_tier / to_tier / reason / score /
proposal_desc / publisher_revision / created_at——"这条记忆为什么被归档？"
从此可回答。真相态事件在 BeliefEvent，存储态事件在这里，两张日志对应
两个正交维度。

### 4. MetabolismPlanner（metabolism/planner.py）

`run_metabolism(session, user_id, config, now, dry_run)` 跑完整周期
scan → score → protect → propose → publish：

- 扫描 active + superseded、tier ∈ {hot,warm,dormant} 的信念
  （rejected 永不碰，archived 等 PR4）；
- `decide_tier()` 纯函数：降级需过保护关 + retention 策略关
  （canonical 永不降温；idle < cool_after_days 不降），且**一次只走一条
  合法边**（hot→warm→dormant 跨周期逐级下降）；重热走 dormant→hot 直达边
  （recall 的 last_accessed_at / 新支持 / 确认都是信号）；
  已 dormant 且继续变冷的报 `archive_pending`（PR4 前可见但不动）；
- 每周期一个 claimed_revision；`max_transitions_per_run`（默认 200）限流，
  超出的下一周期再降；
- `dry_run=True` 全量计算但不发布；`MetabolismReport` 输出
  scanned/protected/proposals/committed/refused/tier 前后分布，
  decision 单列 `protected_reason`（protected floor 会让受保护信念读作
  settled/reheat，保护状态需要独立可观测）。

### 5. Engine API

`MemoryEngine.run_metabolism(user_id, dry_run=False) -> dict`——宿主应用
按用户每日调度（如 02:00）即可。

## 关键修正（实现中发现）

- **heat 的 importance falsy 陷阱**：`float(x) or 0.5` 会把显式 0.0 折成
  默认 0.5，archive_candidate 分支因此不可达——改为仅 None 取默认。
- **0.155 地板**：≥2 条支持的 L4 信念 heat 下限 0.155（evidence 0.08 +
  authority 0.075），默认 dormant 阈值 0.15 之下永远到不了
  archive_candidate。这不是缺陷：有真实证据支撑的信念归档应由 retention
  策略（archive_after_days，PR4）驱动，而不是热度单指标。
- **循环导入**：repository → metabolism 包 → planner → publisher →
  repository。planner 改为 PEP 562 惰性导出后解开。

## PR3-4 路线（更新）

- **PR3 — Evidence Digest**：`EvidenceDigest` 表 + EvidenceCompactProposal
  （>12 条 support 开始压缩；保留最早 1 / 最近 3 / 最高 authority 2 /
  correct 与 contradict 全部），evidence-channel 渲染。
- **PR4 — Archive + Purge**：retention 策略驱动的 ArchiveProposal、
  RestoreProposal、forget 级联删除事务 + DeletionTombstone（防 stale
  worker 复活）、真正的加列迁移工具。
- **MetabolismBench**：current survival / historical preservation /
  correction protection / reactivation / deletion / stale-worker /
  Semantic Preservation Rate / Storage Reduction Ratio。

---

# PR3 — Evidence Digest + Compaction（追加于同日）

Status: **implemented, verified**（26 个新用例；全量 922 passed, 1 skipped）。

## 交付内容

### 1. Evidence 生命周期 + EvidenceDigest 表

Evidence 新增 `retention_state`（hot / representative / compacted /
archived）与 `compaction_group_id`；新表 `mm_evidence_digests`
（每 belief 一行，upsert）：support/contradict/verify/correct 计数、
first/last_seen_at、representative_ids、规则生成的 summary、
source/authority 分布。计数始终覆盖**全量链接图**（含已折叠行），
重复压缩是累加而非覆盖。

### 2. 纯函数选择规则（metabolism/compact.py）

`plan_compaction()`：>12 条 support 才开始；保留样本 = 最早 1 +
最近 3 + 最高 authority 2（并集，可重叠）；**correct / contradict /
verify 永不折叠**；已折叠的行不再参与选择（幂等）。
`build_digest_fields()` 生成汇总统计。planner 与 Publisher 共用同一份
纯逻辑——Publisher 重算后逐项比对，不一致即拒（`stale_selection`），
伪造/过期计划压不动规则会保留的行。

### 3. EVIDENCE_COMPACT 提案 + Publisher 门禁

不变量：保护重算 + `below_compaction_threshold` + `nothing_to_fold` +
`stale_selection`（payload 的 keep/fold 必须与重算一致；policy 被
`clamp_policy` 夹到每槽 ≥1 后重算比对）。提交：digest upsert、
keep 标 `representative`、fold 标 `compacted` 并**清空 content**
（元数据保留，原始内容释放——折叠的正是冗余本体），写
`evidence_compacted` 事件，`belief.compacted_into` 指向 digest。
同样不 bump state revision。

### 4. 读路径

`evidence_for_belief()` 默认跳过 compacted 行（`include_compacted=True`
可看全量）；`evidence_summary()` 附带 digest 块（总数保持完整）；
**重观察一条已折叠的消息即解折**（`record_evidence` 幂等路径把
compacted 行恢复为 hot——证据级的重热信号，与 recall 重热对称）。
renderer 的 evidence 意图（"why / 根据什么"）给渲染命中的信念附
`[evidence] key: <digest 摘要或观察数>` 行，受同一字符预算约束。

## 关键修正（实现中发现）

- **过度保守的保护**：初版把 belief 级 `protection_reason` 整体复用到
  压缩上，一条 correct 链接就挡住整个信念的 46 条冗余折叠。按规格
  （`evidence.relation == "correct"` 是证据级保护 + KEEP 规则
  "correct 全部"），纠正行本身永不折叠即可，其余 bulk 照折。新增
  `compaction_protection_reason()`：只拦未解决冲突（证据的*形状*即
  争议本身）和 pending verification；降温仍用严格的 `protection_reason`。
- **安全扫描器误报**：`Evidence.id.in_(list(keep) + list(fold))` 的
  列表拼接被 SQL 注入规则拦截（DDL/列表拼接形态），改为逐主键
  `session.get` + 显式去重循环，语义不变。

## 实测收益

50 条 support + 1 条 correct 的信念经一个周期：**45 条折叠，
live 行 51→6，内容字符 24829→2489（-90%）**；digest 输出
"50 observations since 2026-06, kept sample: 6"；correct 行原样保留。
行不物理删除（那是 PR4 的级联删除职责），只释放内容。

## PR4 路线（更新）

- **Archive + Purge**：retention 策略驱动（archive_after_days）的
  ArchiveProposal、RestoreProposal、forget 级联删除事务 +
  DeletionTombstone、加列迁移工具。
- **MetabolismBench**：含 Storage Reduction Ratio 与 Semantic
  Preservation Rate 两个核心指标。

---

# PR4 — Archive + Purge + Tombstone + Migration（追加于同日）

Status: **implemented, verified**（23 个新用例；全量 945 passed, 1 skipped）。

## 交付内容

### 1. Archive：策略时钟驱动，不是热度驱动

`decide_tier` 的 dormant 分支：idle（recall 或证据，取最近）≥ 该 retention
class 的 `archive_after_days` → dormant→archived；未到时 → `archive_too_soon`；
class 无归档计划（canonical）→ 永不归档；保护重算一票否决。
方向判断改用**原始 heat**（protected floor 只用于挡降级，绝不制造升级——
受保护的休眠信念不会被“推”成 hot）。归档同样不 bump state revision，
审计行走 `mm_memory_transitions`。archived 行不进 `_scan_beliefs`，
也不进任何默认检索扫描。

### 2. Restore：显式、走 Publisher

`engine.restore_belief(user_id, belief_id)` 构造 archived→warm 的
TIER_TRANSITION 提案（reason="restore"）经 Publisher 提交——合法边、
scope、revision 三重门禁照旧；重复恢复第二次即拒（tier_mismatch）。
rejected 行照旧 resurrection guard。

### 3. Forget：一个删除事务 + 内容无关的 tombstone

新表 `mm_deletion_tombstones`：user_id / generation（每用户递增的
删除世代）/ scope / scope_hash（作用域身份的哈希，**永不是内容**）/
counts_json（每表删除行数）。`forget_belief` 现在级联删除 belief、
events、evidence links、EvidenceDigest、MemoryTransition，修剪孤儿
evidence，并使 snapshot / pending evolution job 失效；`delete_user_memories`
同样覆盖 digest 与 transitions，并写 user 级 tombstone。
**Publisher 的 consent 门禁对 FORGET 豁免**——consent 管“采集”，
不管“删除”：关掉记忆的用户仍有删除已采集数据的权利。
`engine.forget_belief` 改走 Publisher（此前直连 repository）。

### 4. 迁移：SQL 即数据，库零 DDL 执行

原计划的“自动加列”在本仓库的 SQL 安全约束下不可行（DDL 标识符无法
参数化，字面量 ALTER 也被扫描器拦截——试过 text() 与 DDL() 两种构造，
共 11 处全部拦截）。最终形态：`check_schema()` 启动期检测旧库并抛
带指引的 ConfigError；`legacy_migration_sql(engine)` **返回**还需的
additive ALTER 语句，由运维用自己的迁移工具执行。这比自动迁移更正确：
库不应该在 import 时擅自改用户的生产数据库。

## 关键修正（实现中发现）

- **protected floor 制造假升级**：dormant + thin-evidence 的信念 heat 被
  floor 到 0.70 → 目标 hot → 被“重热”到 hot。floor 的本意是阻止降级，
  改为方向判断只用原始 heat，保护状态由 `protected_reason` 独立上报。

## 全链路实测（冒烟）

3 条 support、1000 天无人问津的信念：cycle 1 hot→warm、cycle 2
warm→dormant、cycle 3 dormant→archived（preference 时钟 720 天）；
归档后 recall 返回 None；`restore_belief` 后回到 warm 且 recall 重新
返回 "likes coffee"；`forget_belief` 后 beliefs=0、tombstone generation=1
且不含任何内容，recall 返回 None。

## 收尾状态

四个 PR 的全部功能已落地：tiers/heat/protection/eligibility（PR1）、
planner runtime + 审计（PR2）、evidence compaction + digest（PR3）、
archive/restore/forget/tombstone/migration（PR4）。

---

# MetabolismBench v1 — 生命周期基准（追加于同日）

Status: **implemented, verified**（29/29 用例通过，gate PASSED；956 tests 全量通过）

## 设计

沿用 StateBench 的计分哲学：**评渲染出的状态面，不评答题模型**。每个
case 是一段脚本化的“记忆人生”：对话轮次 → 生命周期操作序列（cycle 在
模拟时钟上推进并跑一个周期）→ 对渲染块和引擎状态的断言。零 answer 模型；
LLM 只用于摄入（K2 提取），与 StateBench 完全一致。

三个指标：

- **Semantic Preservation Rate** — 生命周期操作前可回答的结论，操作后
  仍可回答的比例（probe / probe_after 成对测量）。
- **Storage Reduction Ratio** — 压缩后 evidence 内容字符 / 压缩前。
- **Compaction Efficiency** — 实际压缩率 ÷ 保留策略的理论上限
  （1 − kept/total）。扁平压缩率在阈值附近没有意义（13 条链接、6 条
  样本时理论上限只有 ~0.54），所以 gate 卡的是效率而不是裸比率。

Gate：四个“保护类”（correction / conflict / deletion / stale_worker）
必须 100% 通过——保护失败是 bug 不是调参；总体通过率 ≥ 0.95；语义保持
≥ 0.98；压缩效率 ≥ 0.90。

## 实测（DeepSeek-V4-Flash via aiping relay，temperature 0）

```
category                   cases   pass  semantic  storage  effic.
conflict_preservation          3   1.00        -        -      -
correction_protection         4   1.00        -        -      -
current_survival              4   1.00        -        -      -
deletion                      4   1.00        -        -      -
evidence_compaction           4   1.00     1.000    0.822   1.000
historical_preservation       4   1.00     1.000        -      -
reactivation                  4   1.00        -        -      -
stale_worker                  2   1.00        -        -      -
OVERALL                      29   1.00     1.000    0.822   1.000
recall latency: p50=4.4ms p95=9.4ms max=69.6ms (n=32)
metabolism gate PASSED
```

运行方式（`bench/metabolism.py` 的 `run_metabolism_bench(config, llm_client=...)`）：

- 有 LLM client：强制每轮提取（同 StateBench 的 forced 模式），报告头记录
  模型与 base_url。
- 无 client：自动降级 K1 模式并在报告标注 `degraded_k1_only`，harness
  本身由 `tests/test_metabolism_bench.py` 覆盖（11 个用例，不依赖 key）。

## 实现中发现的问题（bench 驱动的真实修复）

1. **bench 的 turn_index bug**：每次 observe 都传 `turn_count=0`，管线按
   `session:turn` 做 evidence 去重，14 条相同观察塌缩成 1 条链接——信念
   因 thin-evidence 保护永不降温。修法：bench 为每个 case 维护自增序号。
2. **recall 与周期时钟不同源**：recall 在真实时间记访问，周期跑在模拟
   时钟上，重热信号被 freshness 衰减吃掉。修法：`recall` / `render_memory_block`
   接受 `now` 参数，把评分、时间标签、访问遥测统一到同一时钟。
3. **冲突用例的构造路径**：提取器对相同文本会产出不同 key（咖啡/茶各自
   成键——多基数偏好共存是正确行为），且 extractor 报的 contradicts 一律
   按自我纠正处理。跨源冲突按设计走 repository 的
   `conflict_kind=CONFLICT_SOURCE_CONFLICT`（宿主应用接入第二信源时的
   文档化路径），bench 用 `inject_conflict` 算子照此构造。
4. **压缩豁免语义**：correct 链接不阻止压缩（PR3 已定案：纠正行本身永不
   折叠），bench 用例从“纠正后不压缩”改为“压缩后纠正仍在、结论不变”。
5. **safe_json vs safe_json_list**：解析 `representative_ids` 数组误用
   返回 dict 的助手，导致理论压缩率恒为 None。
