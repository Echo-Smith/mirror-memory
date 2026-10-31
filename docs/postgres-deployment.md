# PostgreSQL 部署说明（M2）

> SQLite 是端侧形态，零配置即可用；本文只涉及托管形态（PostgreSQL 16）。
> 隔离机制的使用规范见 `docs/tenancy-context.md`。

## 1. 三个角色

| 角色（默认名） | 环境变量 | 用途 |
|---|---|---|
| `mm_owner` | `MM_PG_OWNER_ROLE` | 表 owner；只在建库 / 迁移时使用 |
| `mm_app` | `MM_PG_APP_ROLE` | 应用运行时登录；`MM_DATABASE_URL` 必须用它 |
| `mm_maintenance` | `MM_PG_MAINTENANCE_ROLE` | `system_scope` 调用点专用；`MM_MAINTENANCE_DATABASE_URL` |

迁移步骤 3 会在角色缺失时创建它们（需要 `CREATEROLE`），但**不设口令**：
口令或外部认证由运维方配置（`ALTER ROLE mm_app PASSWORD '...'`）。
`mm_app` 不是任何表的 owner、没有 `BYPASSRLS`、不能 `ALTER TABLE`；
`mm_maintenance` 由命名策略放行而非 `BYPASSRLS`（理由见 tenancy-context.md）。

## 2. 上线顺序

1. **以 owner / 超级用户登录运行迁移**（唤起 `upgrade_database(engine,
   default_tenant=..., default_app=..., confirm_backup=True)`）。它建表、
   补租户列、改复合主键，并在第 3 步安装 RLS（所有权转给 `mm_owner`、
   GRANT、`ENABLE`+`FORCE ROW LEVEL SECURITY`、两条策略）。每一步可重入。

   **升级前选择（M3）**：迁移第 6 步给 `mm_evolution_jobs` 加 `purpose`
   列。不提供 `legacy_job_purpose`（或 `MM_LEGACY_JOB_PURPOSE`）时，存量
   任务的 purpose 回填为空——worker 会在运行时以 `missing_purpose` 取消
   它们（数量在迁移日志和 `upgrade_database` 返回值
   `legacy_jobs_to_cancel` 中给出）；提供该参数时，存量任务被打上这个
   purpose——它们仍只在存在该 purpose 的有效授权时运行。若要「升级后
   继续跑存量任务」，升级前先 `grant_purpose(purpose=<legacy>, ...,
   operations=["extract", ...])`。
   **第 7 步（M3 第二批）**给 `mm_evidence`、`mm_belief_events`、
   `mm_snapshots` 加 `source_purpose` 列——该行是在哪个 purpose 的授权下
   收集的（observe 写当前 purpose，worker 写任务的 purpose）。存量行回填
   为空标记 `''`（含义：来源未记录）；提供**同一个** `legacy_job_purpose`
   / `MM_LEGACY_JOB_PURPOSE` 时，存量行改打该 purpose——一个选择同时标注
   存量任务与存量记忆。列的服务端默认值始终是 `''`，legacy purpose 只标注
   已存在的行，不会泄漏到之后写入的行上。本批只记录、不按它过滤召回。
   **第 8 步**建访问审计表 `mm_access_audit`（租户表：复合主键、RLS、
   写入守卫类别 `audit`）：每次读取记忆的入口（recall / panel / beliefs /
   belief_by_id / stats / explain / belief_history / 验证候选）在**同一事务**
   里追加一行——user_id、purpose、授权版本、入口名、返回的记忆数量与 id
   列表（只存 id，超过 `access_audit.max_ids` 截断并标记）；被拒绝的读取
   也记一行（decision=denied + reason；缺 purpose 的拒绝仍在 Python 层零 SQL
   拒绝，不记）。审计写不进去则读取失败、不返回数据。`delete_memories`
   不删审计。保留期 `access_audit.retention_days`（默认 90 天）只由显式的
   `MemoryEngine.purge_access_audit()` 按租户清理，宿主对每个租户各跑一次；
   引擎不会替宿主枚举租户。
   **第 9 步（M3 第三批）**替换两张出处表的 `belief_id` 单列索引为租户
   前缀复合索引，并给 `mm_session_summaries` 补上 `source_purpose`
   （recall 的会话摘要回退会返回原始消息文本，摘要与证据一样是出处）。
   **`''` 的语义（M3 第三批起）**：M3 之后所有经授权入口的写入都不会再
   产生 `''`（有测试锁定）；它只出现在两类行上——迁移前的存量行、绕过
   授权入口的直接写入（测试播种等）。可见性上 `''` 在**用户全局记忆开关
   为开时视为可见**（这正是它被收集时的同意制度；开关本身仍在两层授权
   里单独检查，关掉开关整个读取都被拒绝）。要把存量记忆绑定到某个具体
   purpose（从而受该 purpose 的撤销约束），必须在**升级前**决定
   `legacy_job_purpose`——迁移完成后补标需要另行处理，引擎不提供事后
   打标工具。
   **迁移不受应用超时约束**：迁移期间它借出的每个连接都设为会话级
   `statement_timeout = 0`（会话级而非 `SET LOCAL`，因为 `CREATE INDEX
   CONCURRENTLY` 在事务外执行），连接归还连接池时 `RESET` 回建连时的应用
   默认值；`lock_timeout` 不动——迁移沿用自己的 `MM_MIGRATION_LOCK_TIMEOUT`
   （默认 3 s）并失败重试，而不是排队堵住后面的读者。分批回填的每一批是
   独立事务里的一条 UPDATE，Python 循环在事务**之间**运行，所以
   `idle_in_transaction_session_timeout` 不会在回填期间触发。建议迁移用
   专用的 owner 引擎，不与请求流量共享连接池（监听器作用于整个引擎）。
2. **应用进程用 app 登录**：`MM_DATABASE_URL=postgresql+psycopg://mm_app:...@host/db`。
   引擎启动时校验：schema 已建好、每张租户表 RLS 在位、当前登录不是
   超级用户 / owner / BYPASSRLS——任一不满足即拒绝启动（fail-closed）。
3. **设置维护登录**：`MM_MAINTENANCE_DATABASE_URL=postgresql+psycopg://mm_maintenance:...@host/db`。
   worker 的队列扫描与单租户启动探针都走它；未设置时这些路径**报错**而
   不是静默看到 0 行。

**删除记忆 ≠ 撤销授权（M3）**：`delete_memories` 的级联范围不含
`mm_consent_grants`、`mm_memory_preferences`、`mm_purpose_grants(_events)`
——功能开关、全局偏好与 purpose 授权在删除后原样保留；「删除并撤销」
由调用方显式两步（先 delete_memories，再 revoke_purpose / set_consent）。

`MM_PG_RLS=0` 关闭 RLS 相关行为（不再设置事务级租户、不做姿态校验），
**仅用于性能测量基线**，生产禁用。

## 3. 连接池与超时（任务 2）

全部 PostgreSQL 引擎（server 共享引擎、宿主 `MemoryEngine`、维护引擎）都
经 `core.pg_engine.build_engine` 构造，配置项与默认值：

| 环境变量 | 默认 | 理由 |
|---|---|---|
| `MM_PG_POOL_SIZE` | 5 | SQLAlchemy 默认；单进程 ≤ 5+10=15 连接，默认 `max_connections=100` 下可容纳多个进程 + 维护池 |
| `MM_PG_MAX_OVERFLOW` | 10 | 同上 |
| `MM_PG_POOL_PRE_PING` | 1 | 每次借出一次廉价探活，把被服务端 / 代理掐掉的连接变成透明重连而不是请求失败 |
| `MM_PG_POOL_RECYCLE` | 1800 s | 低于常见 NAT / 代理空闲上限；服务端角色或参数变更无需重启即可到达连接池 |
| `MM_PG_STATEMENT_TIMEOUT_MS` | 30000 | observe/recall 的语句是毫秒级；跑到 30 s 的语句是失控查询或等不到的锁，池只有 ≤15 条连接，几条被钉住就是整体不可用 |
| `MM_PG_LOCK_TIMEOUT_MS` | 5000 | claim/publish 的行锁争用毫秒级即可解决；5 s 兜住排在卡死写者后面的情况。迁移另有 `MM_MIGRATION_LOCK_TIMEOUT` |
| `MM_PG_IDLE_IN_TRANSACTION_TIMEOUT_MS` | 60000 | 空闲的开放事务持锁并阻塞 VACUUM；引擎会话是请求级的，60 s 远超任何合法请求，又能在泄漏会话伤到集群前释放。worker 在模型调用期间**不持有事务**（快照事务先结束、发布前再开新事务），所以这个上限对计算时长没有约束 |
| `MM_PG_MAINTENANCE_POOL_SIZE` | 2 | 维护流量只有队列扫描与启动探针 |

三个超时通过 libpq `options`（`-c statement_timeout=...`）在**建连时**设置，
不增加往返；值在进入连接串前校验为非负整数。超过 `statement_timeout`
的语句被取消（`57014`），事务回滚，连接保持可用。

**server 的共享引擎**：PostgreSQL 上所有租户共用一个 SQLAlchemy Engine /
连接池（租户由事务级 `set_config` 与读注入区分）；`MM_TENANT_ENGINE_CACHE`
限制的是轻量 `MemoryEngine` 包装的数量，淘汰包装**不会** dispose 共享池。
新租户包装的创建在缓存锁之外完成（锁内只放占位符），同一租户的并发首
请求等待同一个结果，别的租户不被冷启动阻塞。SQLite 保持每租户一引擎。

## 4. 测量开关（生产不用）

`MM_TENANT_READ_FILTER=rls|none`、`MM_PG_RLS=0`、`MM_ALLOW_UNFILTERED_MEASUREMENT=1`、
`MM_LONGRUN_DATABASE_URL`（longrun 基准指向 PG）只服务 `.github/workflows/m2-rls-perf.yml`
与 `tools/m2_mode_perf.py` 的三模式测量；server 在 `none` 下拒绝启动。

## 5. 备份

PostgreSQL 迁移前要求 `confirm_backup=True`；库不复制文件，缺参数时打印
`pg_dump` 命令提示。

## 6. 删除日志运维（M4）

主库的单行水位 `mm_deletion_watermark`（`db_id`, `applied_seq`）与独立删除
日志按 `db_id` 配对：备份恢复后水位随库回滚，启动对账发现落后即按日志重放
（指纹只删当初删掉的行，id 复用不会误删）。

**部署**：M4 迁移（Step 11）在部署时创建水位行（`db_id` 现铸）并写日志创世
条目（seq 0）。部署前的备份都是「M4 之前」形态。

**M4 部署前的备份恢复后必须先认领**：这种备份带着租户数据但没有水位行，
启动会拒绝（`WatermarkMissing`，报错含 `PRE-M4 backup` 与认领命令）。认领：

```bash
python -m mirror_memory.deletion_adopt --database-url <主库 URL> \
    --db-id <该库历史所在的日志 db_id>
```

认领写入水位行（applied_seq = 0），下次启动对账重放该 `db_id` 的全部条目。

**日志位置必须显式**：`backend=file` 只在端侧单租户 SQLite（含内存库）允许
由 URL 推导目录；server / 多租户部署必须配置 `deletion_log.directory`。
`backend=postgres` 必须配置 `deletion_log.url`（独立于主库的库，app 角色
只有 INSERT/SELECT）。`backend=none` 只允许在从未应用过删除条目的库上使用。

**对账失败即停工**：worker 每个 tick 先对账；断链（`LogChainBroken`）、
日志不可达（`DeletionLogUnavailable`）、水位缺失（`WatermarkMissing`）等
任何异常都跳过本轮——不回收租约、不认领、不发布——按退避重试并累加
`deletion_reconcile_failures` 告警计数。日志比主库旧（`DeletionLogStale`）
同样拒绝：日志的恢复点绝不能落后于主库。
