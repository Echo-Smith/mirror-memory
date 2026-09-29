# Runtime Closure v1 — PR2: Kill Write Bypasses

Date: 2026-09-28
Status: **implemented, verified**（全量 975 passed；生产代码直写点 grep = 0）

## 目标

PR1 把 Publisher 变成可靠协议后，PR2 把所有生产写入入口迁到提案通道。
硬指标：``record_claim(`` / ``update_belief_by_id(`` /
``support_belief_by_id(`` / ``correct_belief(`` / ``confirm_belief(`` /
``reject_belief(`` 在 ``core/repository.py``、``core/publisher.py``、
``bench/``、``tests/`` 之外的出现次数为 **0**。已达成。

## 交付内容

### 1. 新增 TRANSITION_REJECT

用户否认验证问题是一个有自己不变量的用户决策（status → rejected，
key 永不复活），不再借用其他 transition 隐式表达。它与 VERIFY 一起
进入 TRUTH_TRANSITIONS（必须带 expected_revision）。

### 2. Publisher 决策携带 belief_id

commit handler 从“返回 revision”改为“返回 belief_id”，``_commit``
统一在末尾取 revision；``PublishDecision.detail["belief_id"]`` 让调用方
能把证据链接到**新创建**的信念（UPDATE/CORRECT 的行事前不存在）。

### 3. round-trip closes 进入 savepoint

A→B→A 的 revival 原先在 pipeline 里直接改 ORM：

```python
middle.valid_to = ...
middle.status = "superseded"
middle.superseded_by = new.id
```

现在 ``close_belief_ids`` + ``close_at`` 随 UPDATE 提案的 payload 传递，
由 Publisher 的 ``_commit_update`` 在**同一 savepoint 内**关闭。 revival
与它的 closes 成为一个原子 transition——不可能再出现“新值已活、中间值
仍 active”的半成品 round-trip。

### 4. pipeline 三处直写全部迁入提案

- SUPPORT → ``TRANSITION_SUPPORT`` 提案；
- UPDATE → ``TRANSITION_UPDATE`` 提案（含 closes）；
- CREATE / CONTRADICT → 对应提案（conflict_kind 随 payload 传递）。

证据顺序随之调整：``_ensure_evidence_rows`` 先建行并返回 ids（Publisher
的 authority 门禁需要证据在提案发布前就已属于该用户和 session），发布
成功后再 ``_link_evidence_rows`` 链接。原 ``_attach_evidence_rows``
保留为两者组合，供“无信念也要留观测”的路径使用。

每份提案发布前重读 revision：同一轮多个 claim 各自推进它，Publisher
的 CAS 会拒绝用旧 revision 计算的提案。``_publish`` 助手统一构造提案
（actor_type="extraction"）、发布、trace 与日志。

### 5. verification 的 confirm/deny 迁入提案

用户的回答是真值决策，走 VERIFY / REJECT 提案（actor_type="user"），
同样受 revision 门槛与提案日志约束。被拒时记录日志、不改变状态。

### 6. api.correct 迁入提案

``MemoryEngine.correct`` 改发 ``TRANSITION_CORRECT`` 提案，用
``decision.detail["belief_id"]`` 取回新行返回给调用方。

## 验证

- 全量 **975 passed, 1 skipped**（含 integration、longmemeval collision
  storm——即“同一 key 反复 UPDATE”路径、round-trip、verification、
  api correct 的全部既有用例）。
- 生产代码直写点 grep = 0（上述六个函数）。
- MetabolismBench harness 端到端跑通、指标计算正常（semantic 1.000 /
  storage 0.822 / efficiency 1.000 的用例子集）；LLM 全量复跑被 relay
  key 过期（401）阻挡——需要新 key 后重跑，见文末。

## 实现中发现的问题

1. **``allowed_dimensions`` 透传**：非严格模式下该值为 None，迁移时代码
   做了 ``sorted(allowed)`` 直接炸掉 CREATE 路径——改为原样透传
   （None 即“任意维度”）。
2. **证据必须先于提案存在**：Publisher 的 authority 门禁要求 evidence_ids
   真实存在且属于该用户/ session，所以 pipeline 的“先写信念再挂证据”
   顺序必须反转为“先建证据行 → 发提案 → 链接”。
3. **REJECT 的不变量**：``_check_invariants`` 对非 FORGET/TIER/COMPACT 的
   transition 会先查 rejected/superseded 守卫；REJECT 的目标必须是
   active（拒绝一个已关闭的信念无意义），现有守卫顺序恰好正确。

## 下一步（PR3 — Versioned Belief Ledger）

Belief 目前同时承担事实、当前状态、历史与行身份。按既定架构决策：

- 新表 ``belief_identity``（user_id / subject / predicate / slot_key /
  cardinality）与 ``belief_version``（object / polarity / lifecycle_state /
  valid_from / valid_to / epistemic_from / epistemic_to / confidence /
  created_by_proposal / superseded_by_version）；
- 现有 ``Belief`` 降为 **compatibility read model**，写路径变为
  Ledger → project → Belief；
- A→B→A 必须产生三个 version（三条区间），而不是复活覆盖旧 version。
