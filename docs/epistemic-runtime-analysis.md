# Mirror 底层能力提炼：从 Belief 功能集合到 Epistemic Memory Runtime

> 分析基线：2026-09-22 工作区。当前分支包含未提交的 StateBench、Temporal、Evidence、
> Publisher 与 ingestion bugfix；本文把它们视为 WIP，不把文档中已经声明的能力等同于
> 端到端已经成立。

## 1. 结论

分享讨论对 Mirror 的定位是准确的：Mirror 最有价值的能力不是“长期保存一些事实”，
而是管理 AI 对用户形成的、带证据且可演化的信念状态。

当前代码已经开始形成这个方向所需的五类组件：

1. `Belief` 与 `BeliefEvent`；
2. 一等 `Evidence` 与带类型的证据边；
3. validity interval 与 Relation/Lifecycle 分离；
4. `StateTransitionProposal -> Publisher`；
5. query-aware retrieval 与 StateBench 三轨评测。

但目前它们仍然是并排加入的功能，还没有收敛成一个统一内核。最关键的缺口不是再加
一个模块，而是确立一条所有写入、查询和维护都遵守的协议：

```text
Observation/Evidence (发生了什么，谁说的，何时看到)
        ↓
Claim/Proposition (这句话规范化后在断言什么)
        ↓
Relation Judgement (与已有状态是什么关系)
        ↓
Transition Proposal (建议发生什么，基于哪个 revision)
        ↓
Atomic Publisher (校验权限、不变量和并发后提交)
        ↓
Belief State + Transition Ledger (当前状态及可重放历史)
        ↓
Retrieval / Snapshot / Graph / Maintenance projections
```

这条链才是 Mirror 可以对外复用的底层能力。Hybrid Retrieval、Graph、Snapshot 和
Forgetting 都应当消费这条链的投影，而不应绕过它成为新的事实来源。

## 2. 当前代码到了哪里

| 能力 | 当前实现 | 判断 |
|---|---|---|
| Belief Core | `Belief`、`BeliefEvent`、confidence、layer、status | 已有基础，但一个 `status` 混合了认知、时间和发布状态 |
| Evidence | `Evidence`、`BeliefEvidenceLink`、typed relation | 方向正确；Evidence 仍可被原地刷新，写入与 belief 变更也不是同一原子操作 |
| Temporal | `observed_at`、`valid_from`、`valid_to`、scope、window relation | 模块成立；真实 CREATE 路径没有把这些字段写入列，round trip 还会覆盖旧阶段 |
| Relation/Lifecycle | `judge_relation()` + `LifecyclePolicy` | 分层正确；subject 比较仍被硬编码为相同，polarity/termination 语义未建模 |
| Transition Authority | `StateTransitionProposal` + `Publisher` | K3 使用了；observe、correct、forget、verify 等主路径仍直接调用 repository |
| Retrieval | lexical admission + symbolic score + temporal marker | 是 query-aware lexical retrieval，还不是 Hybrid Retrieval，也缺少显式查询契约 |
| Maintenance | evolution worker + K3 snapshot | 是 synthesis worker；不是 contradiction/stale/dormancy maintenance runtime |
| Evaluation | StateBench v1.1 的 state/recall/answer 三轨 | 方向正确；应以 forced/production state 先验约束 retrieval 结论 |

因此最准确的阶段判断是：**Mirror 已经拥有 Epistemic Runtime 的若干原语，但还没有完成
Runtime closure。**

## 3. WIP 中暴露的关键结构问题

### 3.1 “Publisher 是唯一写入者”目前只是目标

`extraction/pipeline.py` 仍然直接调用 `support_belief_by_id()`、
`update_belief_by_id()` 和 `record_claim()`。公开 API 的 correct/forget 也直接调用
repository。现在只有 K3 snapshot 真正走 Proposal/Publisher。

这会产生两个后果：

- Publisher 的 consent、revision、authority 和 invariant gate 没有覆盖主要写入流量；
- 同一种状态变化有两套语义：一套由 pipeline/repository 决定，另一套由 proposal/publisher
  决定。

在所有 mutation 都经过 Publisher 前，文档不应把“single writer”描述为已完成能力。

### 3.2 Publisher 还不具备原子发布语义

当前 gate 先读取 revision，随后执行写入。这是 read-check-write，数据库并发下仍有
TOCTOU 窗口。`claimed_revision=0` 还会跳过 revision 检查。

`publish()` 捕获 commit handler 的异常后直接返回 `commit_error`，没有 savepoint/rollback。
如果 handler 在多次 flush 之间失败，调用方后续 commit 可能看到失败事务或部分 mutation；
“拒绝后数据库 byte-for-byte 不变”目前只对 gate refusal 有测试意义，不能覆盖所有提交失败。

底层协议需要：

1. mutation proposal 必须带明确的 expected revision；
2. 用原子 compare-and-swap/fencing claim revision；
3. handler 在 savepoint 内执行，任何失败完整回滚；
4. transition event、belief state、evidence edge 和 outbox 在同一事务内提交；
5. proposal 有 idempotency key，重复投递返回同一结果。

### 3.3 时间字段存在于模型，但 CREATE 丢失

Pipeline 构造 `CandidateAtom` 时已经读取 `observed_at/valid_from/valid_to`，UPDATE 也会把
它们交给 `update_belief_by_id()`。但 CREATE 最终调用的 `record_claim()` 没有这些参数，
新建 `Belief` 时也没有写这些列。

`valid_from/valid_to` 的原始字符串会由 semantic parser 保留在 `value_json`，但后续 temporal
查询只读取 Belief 的独立列，不会把它们从 JSON 恢复；`observed_at` 则完全丢失。结果是第一段
事实通常仍然是 undated，只有发生 UPDATE 后，新段才可能带 interval。时间模型的单元测试
通过，不代表实际 extraction -> CREATE -> recall 链已经使用该模型。

### 3.4 `(user_id, key)` 唯一约束与 round trip 冲突

A -> B -> A 应产生三个状态阶段：

```text
A [t0, t1)
B [t1, t2)
A [t2, infinity)
```

当前 update 路径找到旧的 superseded A 后，会把同一行重新激活并覆盖其 `valid_from`。
这样 A 的第一段历史无法同时保留。这个行为来自“一条 key 对应一行”的旧模型，无法仅靠
补充 `valid_to` 修好。

当前回归测试还明确要求 “Shanghai -> Berlin -> Shanghai must revive, not accumulate rows”，并
断言最后一个 Shanghai 与第一个 Shanghai 使用同一 row id。所以下面的版本化方案会有意改变
现有契约：实施时必须同步改测试、迁移历史数据，而不能把它当成 repository 内部重构。

需要把稳定身份和状态版本拆开：

- `slot_id` 表示同一个状态槽，例如 `user/lives_in`；
- `claim_id` 表示规范化命题，例如 `user/lives_in/shanghai/positive`；
- `belief_version_id` 表示一次有效期，例如 Shanghai 的 2023-2025 阶段；
- current view 只是一份物化投影，不是历史的唯一存储位置。

### 3.5 polarity、termination 和 contradiction 仍被混为一类

当前 `relation="contradicts"` 的默认行为是降低旧 belief confidence 并标记
`needs_clarification`。以下三种情况却需要不同语义：

| 输入 | 应有语义 |
|---|---|
| “我不再喜欢咖啡” | 关闭正向偏好的当前阶段，创建/表达当前负向或中性状态 |
| “刚才说错了，我其实喜欢茶” | 高权限 correction，保留被纠正历史并开启新版本 |
| 两个来源对同一事实意见不一 | unresolved conflict，保留双方证据并进入 verification |

`polarity` 必须成为结构字段；termination/correction/conflict 必须成为不同 transition intent。
文本里的否定词不应继续承担事实模型的职责。

### 3.6 Evidence 还不是严格不可变的账本

`Evidence` 以 `(user_id, ref)` 去重，但重新观察相同 ref 时会原地刷新 session、
extraction_method、authority 和 observed_at；content 采用“更长文本覆盖”，source_type 不刷新。
这样 provenance 仍会随重新 ingestion 改写，只是并非所有字段都按最后一次输入覆盖。

建议把两层拆开：

- `Observation`：来源事件，不可变，按 source namespace + source ref 幂等；
- `Extraction/Assertion`：某个 extractor 在某版本 prompt 下对 Observation 的解释，可有多个；
- `EvidenceEdge`：一个 assertion 对某个 claim/belief version 的 support/contradict/correct/verify
  关系。

同时，Publisher 当前只检查给定 evidence id 中是否有外部用户记录；不存在的 id 不会被
拒绝，且 proposal 的 evidence ids 没有在 commit handler 中自动建立 evidence edge。

### 3.7 subject identity 尚未真正进入 RelationReasoner

`judge_relation()` 的 `same_subject` 目前固定为 `True`。虽然 K2 会过滤明确的
`third_party`，但这不等于 subject identity 已经成立。未来支持家庭成员、团队、设备或多个
persona 时，不同主体的相同 predicate 会被当成同一状态槽。

Subject 应当是规范化实体 ID，并参与 slot identity；“过滤第三方”应是产品策略，不应替代
内核身份判断。

### 3.8 Session Summary fallback 会绕过信念治理

当 belief block 没覆盖查询词时，renderer 会直接追加 session summary。这个通道没有经过：

- rejected/superseded/validity 过滤；
- contradiction 和 authority 判断；
- targeted belief forget 的同步清理；
- current/history query mode。

因此一条已经被纠正、拒绝或定向忘记的内容，仍可能从 summary 回到上下文。Dual-channel
应该保留，但 summary/evidence recall 必须经过同一个 governance filter，并在输出中标明它是
source context 还是 accepted belief。

### 3.9 Retrieval 需要从 renderer 中拆成查询编译器

当前 temporal intent、query-to-predicate table、diversity cap 和文本拼装位于 renderer；
lexical admission 位于 repository，score 位于 retrieval。职责在文件层已经部分拆开，但缺少
贯穿三者的显式查询契约，因此运行时仍把四个不同问题耦合在一起：

1. 查询在问哪个 subject/predicate/object/time；
2. 哪些候选语义相关；
3. 哪些候选在认知上允许使用；
4. 怎样在字符预算内呈现。

建议建立显式 `RecallQuery`：

```text
RecallQuery
├── user_id / subject_ids
├── predicates / objects / free_text
├── temporal_mode: current | history | at_time | all_occurrences
├── epistemic_floor: confirmed | supported | hypothesis
├── include_disputed / include_rejected
├── evidence_scope / authority_floor
└── top_k / context_budget
```

查询编译后并行进入 lexical、semantic、structured/temporal，以及未来的 graph channel；融合后
再经过 epistemic gate 和 context packer。这样 Hybrid Retrieval 是可替换投影，不会污染
Belief SoT。

### 3.10 当前 schema 缺少升级路径

`MemoryEngine` 只调用 `Base.metadata.create_all()`。它能建新表，不能给已有表添加
`observed_at/valid_from/valid_to/temporal_scope` 等列。对已有 Mirror 数据库，当前 WIP 可能在
首次查询新列时直接失败。

Temporal/Evidence/Publisher 进入公开版本前，需要正式 schema version 与 migration；否则
底层模型越丰富，升级风险越高。

## 4. 应该固化的五个内核能力

### 4.1 Epistemic Ledger

职责：不可变地回答“我们看到了什么、如何解释、为什么相信、状态如何变化”。

最小实体：

```text
Observation       immutable source event
Assertion         extractor interpretation of an observation
Claim             canonical proposition + polarity + qualifiers
EvidenceEdge      assertion -> claim/version relation
TransitionEvent   append-only accepted state change
BeliefVersion     materialized epistemic + temporal state
```

`Belief` 可以继续作为兼容 current projection，但不再承担全部历史。

### 4.2 Temporal State Engine

职责：同时管理 valid time 和 transaction/observation time。

```text
valid_from / valid_to       现实中何时成立
observed_at                 Mirror 何时看到
recorded_revision           Mirror 何时接受为状态
closed_revision             何时被新状态关闭
```

核心不变量：每个 `current_state slot` 在任一 valid time 最多有一个 accepted current version；
round trip 创建新版本；event occurrence identity 包含规范化时间或原始 qualifier fingerprint。

### 4.3 Transition Authority

职责：让 extractor、用户 correction、verification、maintenance 和 import 都只能“提案”，
不能直接改 truth。

```text
Interpret -> Relate -> Propose -> Authorize -> Commit -> Emit
```

Publisher 需要根据 intent 使用不同权限，而不是把所有变化压成 CREATE/SUPPORT/UPDATE/
CONTRADICT。建议至少有：

```text
ASSERT, REINFORCE, REPLACE_CURRENT, END_CURRENT,
CORRECT, DISPUTE, VERIFY, REJECT,
FORGET, SYNTHESIZE, ARCHIVE
```

### 4.4 Retrieval Compiler

职责：把自然语言查询编译成结构查询，并把多个召回通道融合为可解释结果。

推荐执行顺序：

```text
parse RecallQuery
  -> lexical + semantic + temporal/structured (+ graph later)
  -> RRF or calibrated fusion
  -> epistemic/authority/consent filter
  -> current/history/occurrence selection
  -> diversity and budget packing
  -> RecallResult with provenance and score breakdown
```

这比直接在现有 score 上再加 embedding 更重要；否则语义相似度会把已过期或被否定的事实
更高效地召回。

### 4.5 Maintenance Planner

职责：发现问题并产生 proposal，不直接修改 belief：

```text
stale scan
contradiction cluster
duplicate/alias scan
interval gap/overlap scan
snapshot drift
low-confidence verification candidate
dormancy candidate
projection rebuild
```

Maintenance 必须复用 Relation/Lifecycle 与 Publisher。Dormancy 是 retrieval/retention policy，
不能等同于用户要求的 Hard Forget。

## 5. 状态维度必须拆轴

当前单一 `Belief.status` 无法同时表达认知、时间、保留和发布状态。建议至少拆成：

| 轴 | 建议状态 |
|---|---|
| epistemic | hypothesis / supported / disputed / confirmed / rejected |
| validity | current / historical / future / unknown |
| retention | active / cooling / dormant / archived / forgotten |
| publication | pending / committed / superseded / failed |

confidence 是证据聚合结果，不应代替状态；`disputed` 也不应通过把 confidence 乘 0.8 来隐式
表达。

## 6. 基于当前 WIP 的优先级调整

分享讨论给出的 `Hybrid Retrieval -> Temporal -> Maintenance -> Graph -> Lifecycle` 顺序，建立
在 Belief State 已经正确的假设上。当前代码和 StateBench 显示，这个前提还没有完全满足。

建议调整为：

### P0 — Runtime Closure

1. 所有 mutation 统一走 Proposal/Publisher；
2. Publisher 使用 atomic revision claim + savepoint + idempotency；
3. CREATE 持久化 temporal、polarity、subject identity 和 evidence edges；
4. 使用 BeliefVersion/interval，而不是复活旧 row；
5. 建立 schema migration；
6. summary、snapshot、index 的 forget/invalidation 与主状态一致。

### P1 — Recall Contract + Hybrid Retrieval

1. 抽出 `RecallQuery`/`RecallResult`；
2. lexical + embedding + structured temporal 三路召回；
3. 融合分和 epistemic trust 分分开；
4. 支持 current/history/at_time/all_occurrences；
5. graph 先做可重建 projection，不做 SoT。

### P2 — Maintenance Runtime

扫描只产生 proposal；verification、stale、duplicate、drift、interval repair 都通过相同门禁。

### P3 — Retention Lifecycle

在正确性和 recall 稳定后，再引入 active -> cooling -> dormant -> archived。自动策略只改变
可见性/成本层级；物理删除只来自明确 Hard Forget 或治理策略。

## 7. 可以直接作为发布门槛的不变量

### 写入与审计

- 生产代码中没有 Publisher 之外的 belief/snapshot mutation；
- 每个 committed transition 有 proposal id、expected revision、actor、evidence 与结果事件；
- commit handler 任意一点失败后，数据库无部分写入；
- 重放同一 proposal 不产生第二次变化。

### 状态正确性

- current-state slot 在同一时间最多一个 current version；
- preference 的同一 object 在同一时间最多一个 current polarity；
- A -> B -> A 保留三个 interval，current 只有最后一个 A；
- 同一 event occurrence 重复输入为 SUPPORT，不同 occurrence 为 CREATE；
- correction、termination、source conflict 产生不同 transition；
- rejected belief 不因普通 extraction 复活。

### Governance

- foreign/missing evidence id 均被拒绝；
- targeted forget 后，belief、summary、snapshot、semantic index 和 graph 都不可召回该内容；
- memory/feature consent 在 compute 期间变化时，旧 proposal 无法发布；
- migration 可从上一公开 schema 带真实数据升级并回滚。

### Retrieval

- state track 不通过的 case 不计入 retrieval 能力结论；
- 分别报告 current/history/all-occurrences；
- recall 结果返回 channel、relevance score、epistemic score 和 provenance；
- semantic channel 不得越过 rejected、forgotten、authority 和 validity gate。

## 8. 当前验证结果如何解读

本次在 WIP 上得到：

- Temporal、Relation/Lifecycle、Evidence、Publisher、runtime invariant、query-aware retrieval、
  StateBench schema 和 LongMemEval regressions 八个定向测试文件：`233 passed`；
- 完整测试：`702 passed`，无失败、1 个第三方 deprecation warning，用时 `56.73s`；
- Ruff：`63 errors`，其中 49 项可自动修复；
- StateBench v1.1 deterministic：state `25/200 = 0.125`，recall `2/200 = 0.010`；
  replacement `0/30`、round-trip `0/30`、stale-state `0/20`。其余 state 分类并非全为 0：
  contradiction `5/30`、multi-value `4/30`、preference-evolution `12/30`、
  temporal-event `4/30`，合计正好为 25。

deterministic 模式只跑 K1，不能代表带 K2 extractor 的 forced/production 质量。因此这个分数
不能用来否定 Temporal 或 Publisher 模块；它说明的是：**端到端能力仍然高度依赖 K2，且底层
状态模型的关键路径尚未通过统一协议闭合。** 下一次路线决策应以 forced 和 production 的
state track、两者差距，以及按 transition 的失败漏斗为准。

## 9. 最终架构定义

```text
Mirror = Epistemic Memory Runtime

                    ┌────────────────────┐
Observe / Import -> │ Epistemic Ledger   │
                    │ evidence + claims  │
                    └─────────┬──────────┘
                              v
                    ┌────────────────────┐
                    │ State Engine       │
                    │ relation + time +  │
                    │ polarity + policy │
                    └─────────┬──────────┘
                              v
                    ┌────────────────────┐
                    │ Transition Authority│
                    │ proposal + publish │
                    └─────────┬──────────┘
                              v
           ┌──────────────────┼──────────────────┐
           v                  v                  v
     Recall Projection   Graph Projection   Snapshot Projection
           │                  │                  │
           └──────────────────┼──────────────────┘
                              v
                    Context / Agent Runtime

Maintenance Planner ---------------> Transition Proposal
```

Mirror 的护城河不应是某一种数据库、某个 embedding 模型或一条遗忘曲线，而是：任何来源的
记忆变化都能被规范化、解释、授权、审计、按时间查询，并安全投影到不同的 retrieval runtime。
