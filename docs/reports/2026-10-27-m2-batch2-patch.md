# M2 第二批补丁报告（2026-10-27 追加）

分支 `feat/m2-postgres`，补丁起点 `94fd942`，HEAD `ed45525`。**未合并 main。**

## 一、结论

五项全部完成，验收通过：dual-db CI（run **40474261760**，`ed45525`）两条腿全绿——SQLite **1210 passed / 47 skipped**（187.9 s），PostgreSQL **1250 passed / 3 skipped / 4 xfailed**（477.0 s），审计器与写入守卫常开。任务 1 改变了 worker 的事务结构，按任务书**只补跑一次 A 模式 longrun 与 C 同 run 对比**：run **40474266537**（`m2-rls-perf-longrun`），3 轮全 VALID，C 74.3 / 73.4 / 73.2 s，A 80.8 / 82.0 / 82.0 s → **A/C 最小 1.104 ≤ 1.15，无回归**（与批内终验的 1.107 一致），全量 C/A/B 终验未重跑。

## 二、各任务设计决定与证据

**任务 1——worker 计算期间不持有事务**（`a9a65e7`）。先写测试确认现状（`tests/test_worker_pg_transaction.py`）：`MM_PG_IDLE_IN_TRANSACTION_TIMEOUT_MS=2000`、假 LLM `generate()` sleep 3 s——旧代码里从 `_renew_lease` 的 commit 之后到 `_set_job_state` 是**一个事务**，两次模型调用（formulate + K3）都在其中空等，实测后端被 PG 以 idle-in-transaction 终止，发布阶段 `PendingRollbackError`。修法三段：
- **快照事务**：续租 → consent 门禁 → 读 revision → `_load_evidence`（本就返回纯 dict）→ **K3 的读半段**（`Synthesizer` 拆成 `prepare(session)`（只读、纯数据）与 `produce(prepared)`（只模型调用），`synthesize()` 组合二者保持兼容）→ `_end_snapshot_transaction(session)`：commit 且**临时挂起 `expire_on_commit`**——rollback 必然过期所有实例，而计算期间任何属性访问都会懒加载、即"在模型调用背后悄悄重开事务"，正是要避免的（有第二条测试：`before_cursor_execute` 记录器证明模型运行期间零 SQL）；job 需要的字段（id / user_id / attempt_count / watermark）先拷成普通局部量。
- **计算**：formulate / validate / compile_policy / K3 produce，不触碰 session，连接已归还池（测试断言 LLM 调用期间 `pool.checkedout() == 0`）。
- **发布事务**：先 `session.expire_all()`——快照读到的身份映射对象一律作废，Publisher 的 consent / revision / authority 重检读到当下状态；`_renew_lease`、revision 校验、授权重检的"快照可能过时"语义未改（仍以 revision CAS 与围栏为准）。
- 15 个并发测试与 5 个租约测试全部通过（PG 74 passed + 4 xfail；SQLite 78 passed）。

**任务 2——迁移不受 statement_timeout 限制**（`89da6ef`）。`upgrade_database` 的主体包进 `_unbounded_statement_timeout(engine)`：迁移期间每个被借出的连接在 checkout 时 `SET statement_timeout = 0`（**会话级**——`CREATE INDEX CONCURRENTLY` 在事务外，`SET LOCAL` 覆盖不到）并立即 commit 让连接回到空闲态（否则 SQLAlchemy 之后套 AUTOCOMMIT 隔离会报错）；checkin 时 `RESET statement_timeout`——恢复的是 libpq options 建连时写下的会话默认值，即应用自己的配置。`lock_timeout` 刻意不动（迁移沿用 `SetLockTimeout` + `MM_MIGRATION_LOCK_TIMEOUT`，快速失败而非排队）。分批回填每批是一条 UPDATE 独立事务，Python 循环在事务**之间**运行，`idle_in_transaction` 不会在回填中累积（已写进 `docs/postgres-deployment.md`："迁移用 owner 登录、不受应用超时约束"）。测试：`build_engine` 建 statement_timeout = 1 s 的 owner 引擎跑迁移，探针步骤断言普通连接与 AUTOCOMMIT 连接的 `SHOW statement_timeout` 均为 0、`pg_sleep(1.3)` 不被取消；迁移后新借出的连接恢复为 1s，同一条 sleep 被取消。先写测试确认了现状失败（迁移语句被应用超时取消）。

**任务 3——`(tenant_id, app_id, user_id)` 替换单列 `user_id` 索引**（`388a622`）。**替换而非新增**：模型层一条规则（`tenancy.install_tenant_user_indexes`，模型 import 后运行一次）对全部 **14** 张带 `user_id` 的租户表撤掉 `ix_<t>_user_id`、挂上 `ix_<t>_tenant_user`——未来新表按构造自动受益；新库（含 SQLite）`create_all` 直建。存量库走迁移**第 5 步**（LATEST_VERSION = 5）：每表先 `CREATE INDEX CONCURRENTLY` 新索引（失败构建留下的 INVALID 先删再建），再 `DROP INDEX CONCURRENTLY` 旧索引；事务外执行、可重入（有测试，含 INVALID 残留重建）；SQLite 普通建删。EXPLAIN 前后对比（`tools/explain_review.py --compare-user-index`，同一 500 万行数据集先换回单列索引、再换复合索引各跑一遍；工件 `docs/artifacts/2026-10-27-m2-batch2-explain/user_index_before_after_local.md`）：

| 查询（C 模式） | before（单列 user_id） | after（tenant, app, user_id） |
|---|---|---|
| 按用户删事件（200 万行表） | BitmapAnd `ix_*_user_id` ∧ 主键，8.4 ms，**1036 次磁盘读** | 单索引，0.44 ms，**0 读** |
| 按用户删证据 | `uq_mm_evidence_user_ref` 1.74 ms，207 读 | 0.35 ms，0 读 |
| /stats 按用户 ORDER BY created_at LIMIT 20 | Index Scan on `user_id`，**过滤掉 297 行他租户** | 0 行过滤 |
| K3 快照摘要按用户 | BitmapAnd `user_id` ∧ 主键，23 页 | Index Scan，9 页 |
| 按用户删信念 / 证据重查找 (user, ref) | 复合唯一约束本就带租户前缀 | 同级不变 |

**非租户查询清单**：`system_scope` 内三类调用点——队列扫描（按 status / created_at，另有 Step4 的 pending 部分索引）、单租户启动探针（DISTINCT tenant_id, app_id）、迁移内部（按主键回填 / 全表计数）——**没有任何按 `user_id` 过滤的查询**（grep 校验）；所有按用户的语句都在租户上下文内发出、带租户谓词，复合索引前缀精确匹配。删除路径的 `uq_mm_beliefs_user_key`、`uq_mm_evidence_user_ref` 复合唯一约束本就带租户前缀，不受影响。

**任务 4——RLS 租户取值与 Session 绑定一致**（`4a2ee05`）。先写测试确认现状：绑定 A 的 Session 在 `tenant_scope("B")` 里 `session.execute(text("select 1"))` 与 `session.connection()` **不报错**——ORM 语句有 `_check_session_tenant` 拦着，但裸 text / 显式 connection() 不经过 ORM 钩子，而 `begin` 钩子直接读 ContextVar，于是以 B 的租户执行了 set_config。修法按任务书第二案：`Engine` 的 `begin` 监听器只把新事务标记为「待设置」（不发 SQL）；Session 路径由 `SessionEvents.after_begin` 取 **Session 绑定的租户**，先经 `_check_session_tenant` 与 ContextVar 比对——不一致或未绑定在**任何 SQL 之前**抛 `TenantMismatchError`（无租户上下文时不设 GUC，与 RLS fail-closed 对齐）；非 Session 的裸连接（`engine.connect()` / Core）在首条语句前经 `before_cursor_execute` 取 ContextVar，行为不变。三条测试：A 绑定 Session 在 B 下三种入口全部报错且录不到任何 B 的 set_config、回到 A 下正常工作且 set_config 全是 A；未绑定 Session 在租户下拒绝裸 SQL；裸连接仍取环境租户。

**任务 5——SQLite 腿 skip 清单（47 条，全部 PG 专属机制）**：

| 文件 | 数量 | 原因 |
|---|---:|---|
| tests/test_rls.py | 20 | `needs MM_DATABASE_URL pointing at PostgreSQL`（行级安全是 PG 机制） |
| tests/test_concurrency_ported.py | 16 | real-concurrency tests require PostgreSQL（真实并发 + 行锁争用） |
| tests/test_pg_pool.py | 4 | 同 rls（`pg_stat_activity` 计数、libpq options 超时） |
| tests/test_pg_migration_locks.py | 3 | 同 rls（PG 锁级别与并发读阻塞） |
| tests/test_worker_pg_transaction.py | 2 | `idle_in_transaction_session_timeout` 是 PG 设置（本补丁新增） |
| tests/test_tenant_pk.py | 1 | physical composite PK only exists on PostgreSQL |
| tests/test_longmemeval_ingest.py | 1 | LongMemEval oracle not available |

sqlite_only 标记仍为空：skip 全部经 `pytest.mark.skipif` 显式声明且原因如上，无掩盖失败。四个 xfail 均为 M3 占位（strict=True）。

## 三、实际耗时；git diff --stat；提交号

实际耗时：约 **5.5 小时**（2026-10-27 12:00–17:30 本地，含 2 次 CI 等待）。git diff --stat（`94fd942..ed45525`）：**16 files changed, 1095 insertions(+), 65 deletions(-)**。提交：`a9a65e7`（任务 1）、`89da6ef`（任务 2）、`388a622`（任务 3）、`4a2ee05`（任务 4）、`ed45525`（longrun-ac 验收工具与工作流）。CI：dual-db run **40474261760**、longrun-ac run **40474266537**（A/C 最小 1.104，3 轮 VALID）。
