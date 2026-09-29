# Runtime Closure v1 — PR1: Freeze Publish Protocol

Date: 2026-09-28
Status: **implemented, verified**（18 个新协议用例；全量 974 passed, 1 skipped）

## 背景

Metabolism 的 Proposal/Publisher 设计已经比主认知写链更先进：metabolism 的
每个动作都是提案、都过 Publisher，而 extraction / verification / API 仍在
直接调 repository 写入。Runtime Closure 的目标是把 Publisher 本身先变成
一个真正可靠的协议，再迁入口（PR2），最后让 versioned ledger 成为真相源
（PR3）。

PR1 不动任何调用方，只加固协议本身。

## 交付内容

### 1. 协议字段（core/proposal.py）

`StateTransitionProposal` 新增：

- `proposal_id`（默认 uuid4 hex）与 `idempotency_key`（默认等于
  proposal_id）——提案的身份；
- `actor_type` / `actor_id`——谁在要求这个变更（user / worker /
  metabolism / system），进审计日志；
- `expected_revision`——替代原来的 `claimed_revision`（后者允许 0 =
  跳过检查，这是漏洞）。

`TRUTH_TRANSITIONS` 元组显式列出七个真值变更（CREATE / SUPPORT /
UPDATE / CONTRADICT / VERIFY / CORRECT / FORGET）；它们的
`expected_revision` **不得为空**，否则 Publisher 拒绝而不是跳过陈旧
检查。存储类变更（TIER_TRANSITION / EVIDENCE_COMPACT / SYNTHESIZE）
不推进 revision，可省略。

### 2. 四个能力（core/publisher.py）

**CAS revision。** 新增 `cas_bump_state_revision(session, user_id,
expected)`：单条
`UPDATE ... SET revision = revision + 1 WHERE user_id = ? AND revision = ?`
，`RETURNING` 不到行即失败。原来的"读→比→写"三步可以被并发插入击穿，
现在不能：两个并发发布者只有一个能让 UPDATE 落地。真值变更由
Publisher 在 handler 之前 CAS 推进，handler 通过新的
`bump_revision=False` 开关不再自行 bump——一次提交推进且只推进一次。

**幂等。** 新表 `mm_proposal_log`（proposal_id UNIQUE，另索引
idempotency_key）。publish 先查库：proposal_id 命中（字面重放）或
idempotency_key 命中（调用方超时后重建了提案对象）都返回
`already_committed`，不重放写入。行与变更同事务写入，回滚不会留下
幽灵记录。拒绝的提案不记录——它什么都没改，同 id 重试必须能重新
评估。

**SAVEPOINT。** `publish()` 的提交包在 `session.begin_nested()` 里：
revision CAS、handler 写入、提案日志行要么全落要么全滚。handler 在
"belief 已写、event 失败"处抛异常时，整个变更回滚，数据库保持字节
一致，且调用方的事务仍然可用（下一个提案照常提交）。

**完整 authority + scope。** evidence_ids 现在必须：存在（数量不符 →
`evidence_missing`，以前 `[999999]` 会被当成"没有外部证据"放行）、
属于同一用户（`evidence_authority_mismatch`）、且来自提案声明的
session（`evidence_session_mismatch`，补上 README 宣称但从未执行的
scope 检查）。

### 3. 删除仍绕过 consent

`FORGET` 豁免 consent 门禁：consent 管"采集"，不管"删除"——关掉记忆的
用户仍有删除已采集数据的权利。

## 测试（tests/test_publish_protocol.py，18 个）

- expected_revision 必需：CREATE / SUPPORT / FORGET 不带即拒；
  SYNTHESIZE 可省略
- CAS：陈旧拒绝且库不变；一次提交只推进一次；**双 session 并发同一
  expected_revision 只有一个赢**
- 幂等：字面重放 → already_committed 且无二次写入/二次 bump；
  共享 idempotency_key 的重试同样折叠；日志记录 actor 与 revision
- savepoint：handler 失败后库字节一致、revision 不动、无日志行；
  回滚后同一 session 仍可提交
- authority/scope：缺失证据、跨用户证据、跨 session 证据、跨用户目标
- 端到端：CREATE → SUPPORT → FORGET 每个都恰好推进一次 revision

## 实现中发现的问题

1. **幂等查询用错主键**：`session.get(ProposalLog, proposal_id)` 按自增
   `id` 查，永远查不到——改为按 `proposal_id` / `idempotency_key` 列查。
2. **bump_revision 参数漏插**：脚本只处理多行签名，`confirm_belief` /
   `reject_belief` 两个单行签名漏掉，条件 bump 引用未定义名——补齐后
   全绿。
3. **旧测试构造提案不带 revision**：这正是新协议要拦的行为，测试补上
   `expected_revision` 后测到它们本来要测的门禁。

## 下一步（PR2 — Kill Write Bypasses）

生产代码中的直写点（grep 指标：repository.py / publisher.py / tests/
之外必须为 0）：

- `extraction/pipeline.py`：`support_belief_by_id`（585）、
  `update_belief_by_id`（627）、`record_claim`（707），以及 A→B→A
  round-trip 直接改 `middle.valid_to` / `middle.status` 的 ORM mutation；
- `extraction/verification.py`：`confirm_belief`（346）、
  `reject_belief`（349）——建议新增 `TRANSITION_REJECT`；
- `api.py`：`correct_belief`（723）。

迁移顺序：pipeline → verification → API → worker。
