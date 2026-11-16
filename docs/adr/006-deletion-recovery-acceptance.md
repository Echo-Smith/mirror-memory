# ADR-006: 删除恢复验收设计（草案 · 待独立复核）

状态: DRAFT — 设计文档，未包含任何代码改动。
基线: `6f7a840`（M5 第一批补丁 12）。
证据分级: **R** = 本仓库真实运行过并有记录的测试；**P** = 部分覆盖（真实运行但场景组合不完整）；**N** = 从未运行（明确未跑）。

---

## 1. 验收目标

对"删除 → 备份 → 恢复 → 重放"这条最危险的路径，给出**可证明**的安全性质，并区分哪些性质已被真实运行证据支撑、哪些只是设计意图。

### 1.1 必须成立的安全性质（判定标准）

| # | 性质 | 反例即失败 |
|---|---|---|
| S1 | 已删除的记忆在任何恢复路径上**不复活** | 恢复旧备份后启动，`mm_beliefs` 中重新出现已删除行 |
| S2 | 恢复后主库水位**单调前进**到日志 tail，绝不回退 | `applied_seq` 小于日志 tail 而引擎正常服务 |
| S3 | 见证 tail 与日志 head 满足 `witness.tail_seq ≤ head_seq`，且 hash 一致 | 见证指向一个日志中不存在的 hash |
| S4 | 任何被拒绝的路径（缺见证 / 不连续 / 认证失败 / 分叉）**三份证据字节不变** | 拒绝的同时日志、见证或注册表被改写 |
| S5 | 并发交错（append/prune/恢复发布窗口）下，日志哈希链**始终可验证**，见证**永不回退** | 交错后 `verify_chain` 失败或见证 tail_seq 下降 |
| S6 | 首次创建与"证据丢失"可区分；不可区分时 fail-closed | 无凭据无见证但 applied>0 的库被铸出新历史 |
| S7 | 受限角色下全链路可运行，且运行时角色**无法**改写历史 | `mm_p10_logapp` 成功 DELETE/DROP 日志表 |

### 1.2 明确**不承诺**的性质（本地不可证明）

| 性质 | 为什么 |
|---|---|
| 全套证据（日志+见证+注册表+bound+密钥）同时回滚 | 无第一方锚点可区分"从未存在"与"完美擦除"；需离线外部锚点（未实现） |
| 密钥泄露后的**完整性/防伪造** | witness/registry 是 MAC 签名而非加密：能读密钥的攻击者可重签，本不存在机密性承诺可"丢失"（与 integration.md 威胁模型一致） |
| 跨设备实时性 | 恢复检测是启动/重放时的，不是实时的 |

---

## 2. 场景矩阵与证据分级

按 6 个阶段 × 2 个后端组织。每格标注证据级别。

### 阶段 A：首建与身份（S6）

| 场景 | 文件后端 | PG 后端 |
|---|---|---|
| A1 健康首建（O_EXCL 胜者） | **R** `test_deletion_witness` 首建系列 | **R** `test_p10_pg_chain_runs_on_restricted_roles` |
| A2 并发首建（确定性 Barrier/锁） | **R** `test_p10_creation_lock_blocks_a_second_creator_until_release` | **N**（复核降级：p12 combined 的子进程是 append/prune，genesis 在 spawn 前单进程完成——**PG 并发首建从未被测试**，见 G10） |
| A3 创建者持锁被 SIGKILL 后的残留判定 | **R** `test_p10_creator_killed_while_holding_the_lock_leaves_safe_state` | **N** PG 无对应崩溃注入 |
| A4 bound 标记 + 密钥存活、其余全缺、applied=0 | **P**（复核更正引用：原引测试实为 applied>0 的 A5 场景。applied=0 由 `test_p10_creator_killed...`（残留后拒绝）与 `test_p8_bound_and_key_present_but_all_evidence_gone` 覆盖，但**仅到 log-API 层，未经真实 MemoryEngine 启动路径**） | **R**（凭据缺失拒绝，`..._without_owner`） |
| A5 applied>0 且无凭据（不可区分） | **R** fail-closed 拒绝 | **R** 同上语义 |

### 阶段 B：删除与见证推进（S1/S3）

| 场景 | 文件 | PG |
|---|---|---|
| B1 真实引擎删除（记忆消失 + 日志推进） | **R** brief-repro 系列 | **R** `test_p10_pg_predeletion_backup_restore_replays_the_deletion` |
| B2 日志提交后、见证发布前崩溃 | **P** 无直接注入，但**模拟崩溃窗口**已有测试：`test_crash_between_log_write_and_witness_advances_witness`（把见证记回旧值再 heal）与 `test_crash_heal_still_works_when_the_chain_is_genuine` | **P** 同左 |
| B3 见证发布后日志被外部截断 | **R** witness 系列 | **R** `test_p10_pg_prune_refuses_a_broken_chain` 等 |

### 阶段 C：恢复与重放（S1/S2）

| 场景 | 文件 | PG |
|---|---|---|
| C1 删除前备份 → 删除 → 恢复 → 启动重放 → 记忆不复活 | **R** brief-repro ①（sqlite3.backup） | **R** `test_p10_pg_predeletion_backup_restore_replays_the_deletion`（pg_dump/pg_restore） |
| C2 同一主库 + 同一日志标识 + 受限角色 + Barrier 交错 + 重放 | **N** 文件后端无此联合场景 | **R** `test_p12_pg_joint_restore_then_concurrent_append_prune` |
| C3 恢复后水位落在日志 tail | **P**（在 C1 中隐式断言） | **R** 显式断言 `landed == tail` |
| C4 三存储（主库+日志库+见证）同时恢复 | **N** | **N** 明确未做 |
| C5 恢复的是**更旧**的备份（跨多个删除代） | **N** 只有单代回退 | **N** |
| C6 恢复到**全新**机器/密钥轮换 | **N** | **N** 密钥丢失路径仅在文档中描述 |

### 阶段 D：拒绝路径的证据不变性（S4）

| 场景 | 文件 | PG |
|---|---|---|
| D1 见证缺失 + 有凭据 → append 拒绝且证据不变 | **R** `test_p8_*` | **R** `test_p8_pg_append_refuses_missing_witness` |
| D2 等高分叉 | **R** | **R** |
| D3 更长分叉（见证 hash 非祖先） | **R** | **R** `test_p10_pg_prune_refuses_when_witnessed_hash_not_ancestor` |
| D4 见证认证失败（MAC 错） | **R** | **R** |
| D5 CLI dry-run 零副作用（逐字节） | **R** `test_p8_cli_dry_run_has_no_persistent_side_effects` | **R** `test_p11_adopt_cli_on_pg_backend_dry_run_and_reconcile` |
| D6 CLI 分叉拒绝（即使 --yes） | **R** | **R** `test_p11_adopt_cli_on_pg_backend_refuses_divergence` |

### 阶段 E：并发与崩溃窗口（S5）

| 场景 | 文件 | PG |
|---|---|---|
| E1 多进程 append 交错 | **P**（复核降级：multiprocess 测试均为首建/锁场景；文件后端 append 交错只有**线程版** `test_concurrent_appends_keep_log_and_witness_consistent`——跨进程 flock 语义未测，见 G9） | **R** `test_p11_pg_prune_and_append_interleave_safely` |
| E2 append/prune 进程交错（Barrier） | **P**（补丁 12 用 Event 非严格 Barrier；文件后端无） | **R** p12 系列 |
| E3 二次压缩锚点（连续两次 prune 后链完整） | **R** `test_second_compaction_continues_the_chain` | **R** 同名场景经 interleave 触发 |
| E4 prune 与恢复发布窗口交错 | **N** 未单独注入 | **P** p12 联合场景间接覆盖 |
| E5 重放与用户写入并发 | **N** 未测 | **N** |
| E6 恢复过程中的新删除写入 | **N** | **N** |

### 阶段 F：角色与权限（S7）

| 场景 | PG |
|---|---|
| F1 受限角色跑 genesis→append→verify→prune 全链 | **R** `test_p10_pg_chain_runs_on_restricted_roles` |
| F2 运行时角色 DELETE/DROP 被拒 | **R** 同上 |
| F3 未 provision 的库对运行时 fail-closed | **R** `test_p10_pg_issue_genesis_on_restricted_role_without_owner` |
| F4 运行时角色无法伪造见证（MAC） | **R** 补丁 7 系列 |
| F5 日志/见证/注册表/密钥分属不同 OS 凭据 | **N** 明确未实现（运行时与见证写入者同登录） |

---

## 3. 缺口清单（按风险排序）

| # | 缺口 | 风险 | 建议 |
|---|---|---|---|
| G1 | **C4 三存储同时恢复**未测——生产备份很可能把主库+日志库一起恢复，此时水位与日志同时回退，重放幂等性（补丁 12 修复）从未在该形态下验证 | 高 | 补 1 个 e2e：dump 三个库 → 全部恢复 → 启动 → 断言水位 ≥ 日志已对齐部分且记忆不复活 |
| G2 | **C5 更旧备份**（跨多代删除）未测——恢复到第 1 代删除之前，而日志已有 3 代 | 高 | 在 G1 的测试中多删两轮，恢复到最早 dump |
| G3 | **B2 崩溃窗口注入**（日志提交后见证发布前 kill）只有设计推断，无直接测试 | 中 | 文件后端可用 SIGKILL 注入；PG 可用 `pg_terminate_backend` 在事务提交后、见证 UPDATE 前终止 |
| G4 | **E6 恢复期间的新写入**未测——恢复停机窗口外，是否有写入绕过停机 | 中 | 明确运维约束（恢复必须停写）并写入 runbook；或加测试证明恢复期间的 append 会被水位检查拒绝 |
| G5 | **C6 密钥丢失/轮换**只写在文档，无测试 | 中 | 至少补"密钥文件被替换后见证认证失败 → fail-closed"的单测 |
| G6 | **E5 重放与用户写入并发** | 低-中 | 依赖 SQLite/PG 事务隔离；可推迟 |
| G7 | **F5 角色隔离** | 低（已如实标注未实现） | 长期项 |
| G8 | 共享测试目录密钥残留（夹具根因） | 低（测试卫生） | 建议单独修，不属验收范围 |
| G9 | **文件后端多进程 append 交错**未测（复核新增）：现有 multiprocess 测试均限于首建/锁；跨进程 flock 语义与线程不同 | 中 | spawn 多进程 append 交错测试 |
| G10 | **PG 并发首建**未测（复核新增）：combined 测试为避免 CREATE TABLE 竞态而特意让 genesis 在 spawn 前单进程完成，因此该竞态从未被注入 | 中 | provision 后 spawn 多进程同时 `ensure_genesis`（凭据/witness 发布竞态仍可注入） |

---

## 4. 独立复核要求（复核者须知）

1. **不要把本文件的"R"当真**——逐条打开对应测试文件确认存在且断言的就是该性质；任何一条对不上即降级为 P/N。
2. 复核 §3 的风险排序是否正确，特别是 G1/G2 是否真的从未被覆盖。
3. 复核 §1.2 的"不承诺"边界是否与 `docs/integration.md` 一致。
4. 复核后输出：必须修的项（G 前缀）、可不修的项及理由。
5. 复核者**不应**修改本文件，也不应先写代码——只输出复核结论。

---

## 4b. 独立复核结论（已完成，2026-11-16）

复核者逐条 grep 验证了全部 R 格子与缺口清单，结论：

**降级 3 处**：A2/PG（R→N：PG 并发首建从未测过）、A4/文件（R→P：引用错测试，applied=0 仅 log-API 级）、E1/文件（R→P：append 交错只有线程版）。B2 补引两个模拟崩溃窗口测试；D2/D4 补点名。

**缺口修正**：G1、G2、G5 经全量 grep 确认**零覆盖**，成立；新增 **G9**（文件多进程 append 交错）与 **G10**（PG 并发首建）。G3–G8 描述属实。

**一致性**：§1.2 与 integration.md 逐行核对一致；"密钥泄露"行措辞已按复核意见从"机密性"改为"完整性/防伪造"。

**复核者判定的必须修复项**（发布前）：
- **G1**（高）：生产备份最常见的形态恰是"主库+日志一起回滚"，补丁 12 的重放幂等修复正是在该形态下最需要被证明的性质——零覆盖。
- **G2**（高）：跨多代回退命中 checkpoint/锚点边界，与单代回退走不同代码路径；在 G1 的 e2e 里多删两轮即可覆盖，成本极低。
- **G5**（中偏高）：密钥替换后 fail-closed 是 S4/S6 的直接推论，一个单测可钉住。

**可不修**：G3（模拟崩溃窗口已覆盖语义）、G4/G6（可推迟）、G7（已如实标注）、G8（测试卫生）。

**总结论**：草案可作为**可信的验收基线**——证据分级总体诚实，绝大多数 R 格子经查属实（含真实 pg_dump/pg_restore、真实受限角色、MAC 伪造拒绝）；但 G1/G2/G5 须在发布前补测。

## 5. 证据汇总

- 本文件所有"R"标注的测试均为**真实运行**过（本地 PG 16 容器 + 真实受限角色 + 真实 pg_dump/pg_restore），最近一次全量：
  SQLite 1657 passed / PG 1765 passed（补丁 12 报告，`/tmp/m5p12_pg2.txt`）。
- "N"项为**从未运行**，如实标注。
- CI：**未跑**（无法触发）。
