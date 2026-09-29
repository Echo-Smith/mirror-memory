# Runtime Closure v1 — PR3: Versioned Belief Ledger

Date: 2026-09-28
Status: **implemented, verified**（9 个 ledger 用例；全量 984 passed, 1 skipped）

## 目标

``Belief`` 此前同时承担四件事：事实、当前状态、历史、数据库行身份。
PR3 把历史真相迁入 versioned ledger，``Belief`` 降为 compatibility
read model——所有读路径（retrieval / renderer / metabolism / bench）不变。

## 交付内容

### 1. 两张表（core/models.py）

``mm_belief_identities``：稳定的“槽”。(user_id, subject, predicate,
slot_key) 唯一。

``mm_belief_versions``：槽的一个值，**两个时钟分开记录**：

- ``valid_from`` / ``valid_to``——事实本身何时为真；
- ``epistemic_from`` / ``epistemic_to``——引擎何时学会、何时不再相信；
- ``created_by_proposal``——打开这个区间的提案 id（可回溯到确切提交）；
- ``superseded_by_version``——替换它的版本。

### 2. 槽的派生规则（_slot_key_for）

单值 current_state 谓词（``lives_in`` / ``works_at``，或 cardinality=
single）同时只持一个值，所以它持有过的每个值是**同一个槽**的一个
version——槽键就是谓词。多值谓词（``likes``）的多个值共存，每个值是
自己的槽，键为值特定的读模型键。这直接实现了规格里的例子：

```
lives_in
Shanghai [2024-01, 2026-05)
Beijing  [2026-05, 2026-09)
Tokyo    [2026-09, ∞)
```

### 3. A→B→A = 三个 version

``update_belief_by_id`` 的 revival 分支（重新打开已 superseded 的旧行）
**已删除**。返回值现在走普通路径：读模型得到消歧键的新行
（UNIQUE 不关心 status），ledger 得到同一 identity 的第三个 version。
第一段上海史保持关闭。

### 4. 哪些 transition 写 ledger（VALUE_TRANSITIONS）

- CREATE / UPDATE / CORRECT：值改变 → 追加 version（并把同槽仍打开的
  旧 version 关闭：epistemic_to=now、superseded_by_version 指向新的）；
- SUPPORT / VERIFY / CONTRADICT：深化或争议**同一个值** → 不追加
  （SUPPORT 曾错误地追加第二版本，被专用测试抓住）；
- REJECT：拒绝是用户决策 → 关闭该值的 version（epistemic_to），不追加。

追加发生在 Publisher 的 savepoint 内，与产生它的变更同事务——区间
不存在于打开它的提交之外。

### 5. 读取

- ``repository.belief_history(user_id, belief_id)``：版本链（旧→新），
  含两个时钟与提案溯源；
- ``MemoryEngine.belief_history(user_id, belief_id)``：公开 API；
- 读模型不变：``list_active_beliefs`` / 渲染 / metabolism 全部照旧工作
  （有测试钉住这点）。

## 测试（tests/test_belief_ledger.py，9 个）

- CREATE 追加首版本（含 proposal 溯源与 belief_id 投影）；
- SUPPORT 不产生新版本；
- UPDATE 关闭前一版本（superseded_by_version 指向新版本，同一 identity）；
- **A→B→A = 三版本**，每段的 created_by_proposal 各自对应打开它的提案，
  一个 identity；
- CORRECT 是同槽的新版本；
- **两个时钟分离**：valid_from/valid_to（事实）≠ epistemic_from（学会的
  时间）；
- 读模型兼容（list_active_beliefs 照旧）；
- 越权与不存在的信念返回空历史。

## 实现中发现的问题

1. **槽键最初误用值特定的读模型键**（``lives_in:shanghai`` vs
   ``lives_in:berlin`` 成了两个 identity）——改为按单值/多值派生。
2. **Python ``or`` 混进 SQL 表达式**（``BeliefIdentity.predicate == p or ""``）
   让 SQLAlchemy 对子句取布尔值而抛 ArgumentError——改为括号内取值。
   （同一错误在 PR1 也出现过一次；是本次迁移的高频笔误。）
3. **SUPPORT 追加了重复版本**——引入 VALUE_TRANSITIONS 区分“值改变”与
   “深化同一值”。
4. **epistemic_from 取值顺序**：原取 ``first_seen_at``（行创建默认值），
   改为优先 ``observed_at``（调用方给的观测时间），否则 UPDATE 路径的
   学习时间永远是“现在”。

## 下一步（PR4 — StateBench v1.1 Gate Closure）

固定 same model / same commit / same dataset / same config / same
answerer，跑 forced + production，目标：

- State overall >= 0.90，每类 >= 0.85
- Recall overall >= 0.85
- Production-forced gap <= 0.05

需要可用的 LLM key（当前 relay key 已过期）。
