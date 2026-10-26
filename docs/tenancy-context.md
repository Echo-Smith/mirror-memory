# 租户上下文传递规范（M1）

> 本文档是 mirror-memory 多租户隔离的**使用规范**。实现见
> `src/mirror_memory/core/tenancy.py`。违反本文档的写法会被测试期
> SQL 审计器或写入守卫拦截。

## 核心概念

- **租户**：`(tenant_id, app_id)` 二元组。`user_id` 保持原义（租户内的主体）。
- **上下文**：租户存放在 `ContextVar` 中，SQL 执行时才读取。传错上下文
  等于串租户，因此上下文的设置点必须显式、可审计。
- **会话绑定**：Session 在**创建时**绑定当时的租户（`session.info`）。
  会话是单租户对象；在别的租户上下文里使用它会抛
  `TenantMismatchError`。

## 正确写法

### 1. 设置租户：只用 `tenant_scope`

```python
from mirror_memory.core.tenancy import tenant_scope

with tenant_scope(tenant_id, app_id):
    engine.observe(user_id=..., ...)
```

- 这是**唯一**的设置方式；没有裸 setter（禁止 `set_tenant` 式用法）。
- 退出（含异常路径）自动恢复原值。
- 禁止空字符串租户；`tenant_scope("", ...)` 直接抛 `ValueError`。

### 2. 创建会话：在租户作用域内创建

```python
with tenant_scope("tA", "app"):
    session = Session(engine)   # 绑定到 ("tA", "app")
```

- 一个会话只为一个租户服务；需要切租户就开新会话。
- 在租户作用域外创建的会话绑定值为 None，只能做非租户工作
  （或在 `system_scope` 内做队列扫描）。

### 3. 跨租户系统操作：只用 `system_scope(reason)`

```python
from mirror_memory.core.tenancy import system_scope

with system_scope(reason="evolution worker: crash-recovery sweep and pending-job scan"):
    ...  # 队列扫描等
```

- `reason` 必填，会被测试期 SQL 审计器与日志记录。
- 第二批会加使用点白名单 + CI 检查；新增使用点必须评审。

### 4. 线程池：必须传上下文

```python
from mirror_memory.core.tenancy import TenantAwareExecutor, submit_with_tenant

wrapped = TenantAwareExecutor(pool)          # 包装整个 executor
wrapped.submit(work, arg)                    # 上下文随调用走
# 或单次：
submit_with_tenant(pool, work, arg)
```

裸线程池里的租户查询会因上下文缺失抛 `TenantContextMissing`（fail-closed）。
底层用 `contextvars.copy_context()`；不要自己在目标线程里"猜"租户。

### 5. asyncio：直接用

`asyncio.create_task` / `gather` 自动继承当前上下文，每个协程应在自己的
`tenant_scope` 内创建并使用自己的会话（不要跨 `await` 持有别的租户的会话）。

### 6. 后台 Worker：从任务行读租户

```python
with tenant_scope(job.tenant_id, job.app_id):
    session = factory()          # 绑定到任务行上的租户
    ...                          # 处理；队列扫描在 system_scope 里做
```

Worker **不允许**继承调度方的上下文——租户的唯一来源是任务记录本身
（见 `worker/evolution.py: _claim_and_process`）。

## 禁止写法（会被拦截）

| 写法 | 后果 |
|---|---|
| 裸 set / 绕过 `tenant_scope` 设置租户 | 没有这个 API |
| 同一会话跨租户复用 | `TenantMismatchError` |
| 无上下文查询/写入租户表 | `TenantContextMissing` |
| 修改已有行的 `tenant_id` / `app_id` | `TenantMismatchError`（租户列不可变） |
| 裸线程池跑租户查询 | `TenantContextMissing` |
| 在 Publisher / 仓储认可入口之外写 `mm_beliefs` / `mm_belief_events`（引擎代码路径） | `WriteGuardViolation` |
| 租户表的 SQL 缺 `tenant_id = ?` / `app_id = ?` 绑定条件（测试期） | `AuditViolation` |
| 为性能关闭 `propagate_to_loaders` 或缩小注入范围 | 评审拒绝（硬原则） |

## 表写入分类（任务 2，唯一事实源 `core.tenancy.TABLE_WRITE_CLASSES`）

新表必须先在此分类（映射缺表 → 测试报错），再写代码。

| 类别 | 表（18） | 允许的写入作用域 / 入口 |
|---|---|---|
| engine（记忆内容） | mm_beliefs, mm_belief_events, mm_evidence, mm_evidence_digests, mm_belief_evidence_links, mm_belief_identities, mm_belief_versions, mm_snapshots, mm_memory_transitions, mm_deletion_tombstones, mm_proposal_log | `engine_write_scope()`：Publisher 发布与仓储认可 claim mutator（`@_sanctioned_writer("engine")`） |
| queue（任务队列） | mm_evolution_jobs | `queue_write_scope()`：worker claim/complete/recycle、enqueue |
| consent（授权配置） | mm_memory_preferences, mm_consent_grants, mm_users | `consent_write_scope()`：偏好/授权/用户设置函数（`@_sanctioned_writer("consent")`） |
| telemetry（运行记录） | mm_extraction_stats, mm_session_summaries, mm_intervention_events | `telemetry_write_scope()`：各记录函数（`@_sanctioned_writer("telemetry")`） |

### engine 类的登记写入函数清单（`@_sanctioned_writer("engine")`）

| 函数（core/repository.py） | 写入的表 | 合法调用方 |
|---|---|---|
| `record_claim` | mm_beliefs, mm_belief_events, （经 record_evidence/link_evidence 的 evidence/links） | 抽取管线（extraction/pipeline）、bench、测试夹具 |
| `record_evidence` | mm_evidence | **抽取管线**（extraction/pipeline）写入证据，不经过 Publisher；record_claim |
| `link_evidence` | mm_belief_evidence_links | 抽取管线；record_claim |
| `get_or_create_identity` | mm_belief_identities | 身份解析（memory/identity），经管线与 Publisher 的 ledger 投影 |
| `append_belief_version` | mm_belief_versions | Publisher（`_append_version`，VALUE_TRANSITIONS 提交后投影） |
| `set_belief_tier` | mm_beliefs（tier 列）, mm_memory_transitions | Publisher（`_commit_tier_transition`） |
| `apply_evidence_compaction` | mm_evidence_digests, mm_evidence, mm_beliefs（compacted_into） | Publisher（`_commit_evidence_compact`） |
| `confirm_belief` / `reject_belief` / `support_belief_by_id` / `update_belief_by_id` / `update_belief_value` / `correct_belief` / `forget_belief` | mm_beliefs, mm_belief_events | **仅 Publisher**（各 `_commit_*` handler） |
| `mark_belief_accessed` | mm_beliefs（访问遥测列） | recall 渲染路径（render/renderer） |
| `_write_deletion_tombstone` | mm_deletion_tombstones | forget 链路（Publisher 的 `_commit_forget`） |

注意：证据表（mm_evidence / links）由**抽取管线**直接写入而不经过
Publisher——这是设计事实（证据是观察记录，不是状态变迁），已在收尾
补丁中显式登记；其余 engine 表的状态变更全部经 Publisher。

### 主键与审计豁免（M2 任务 3 + 补丁）

每张租户表的主键在 ORM 层都是 `(tenant_id, app_id, ...)` 复合键，因此
ORM 工作单元的 UPDATE / DELETE / 按 PK 刷新天然带租户谓词。据此，SQL
审计器对「主键定位写语句」的豁免已在**两个方言上全部撤销**：一条不带
租户谓词的 PK 语句（手写 SQL、对他租户行的 merge-by-id）现在是违规，
而不再是豁免。SQLite 的物理主键仍为单列（自增限制，DDL 由编译钩子收
窄），但映射与语句形态两库一致。

仍然保留的运行时防线（与审计相互独立）：租户列构造期写守卫、
`session.merge()` 跨租户拒绝、flush 期租户校验、读路径的
`with_loader_criteria` 注入。

### 调用方白名单（M2 任务 4）

上表的"合法调用方"不是注释而是可执行的约束：`tenancy_writer_callers.yaml`
按函数列出允许调用它的模块，`tools/check_writer_callers.py` 用 AST 扫描
比对，CI 与本地均可运行：

```
python tools/check_writer_callers.py          # 双向检查，有漂移则非零退出
python tools/check_writer_callers.py --list   # 打印当前调用方
```

两个方向都判失败：**未登记的调用方**（多出来的写路径）与**幽灵条目**
（登记了但已不再调用）。后者同样致命——一份高于现实的清单不再是代码的
描述。`@_sanctioned_writer` 是唯一能打开写入类作用域的装饰器，因此它的
调用方就是能力的持有者；新增调用方必须与调用代码同一个提交里更新
`tenancy_writer_callers.yaml`。

测试侧：fixture **setup** 阶段默认在 `fixture_write_scope` 内（autouse
hookwrapper）；**测试体**不再有全局豁免——直写表的测试体必须声明
`@pytest.mark.fixture_writes(reason="...")`（reason 必填）。守卫的真实
拦截力由金丝雀测试（`@pytest.mark.no_write_fixture_scope`）证明：逐类
反例、跨树插件模块反例、映射完备性检查都在金丝雀套件中。

### PostgreSQL 行级安全：第二道防线（M2 第二批任务 1）

实现见 `src/mirror_memory/core/rls.py`，作为迁移**第 3 版本步骤**
（`Step3RowLevelSecurity`，可重入）安装。SQLite 不受影响（无 RLS，
版本号照常推进）。

**角色**（名字可用 `MM_PG_OWNER_ROLE` / `MM_PG_APP_ROLE` /
`MM_PG_MAINTENANCE_ROLE` 配置，默认如下）：

| 角色 | 用途 | 关键属性 |
|---|---|---|
| `mm_owner` | 建表、迁移、DDL | 表 owner；`FORCE` 使其**同样受策略约束**，由维护策略放行 |
| `mm_app` | 应用日常读写 | 不是 owner、无 `BYPASSRLS`、不能 `ALTER TABLE`；只被租户策略放行 |
| `mm_maintenance` | 仅 `system_scope` 白名单调用点 | **独立引擎/连接池**；不是 `BYPASSRLS`，由命名策略放行 |

维护角色选**专门策略而不是 BYPASSRLS**：策略是表级、可在 `pg_policies`
里逐表看见和撤销的对象，角色本身仍是普通登录；`BYPASSRLS` 则会无声地
绕过这个库里现在和将来每一张表的每一条策略。

**策略**（每张租户表 `ENABLE` + `FORCE ROW LEVEL SECURITY`）：

- `mm_tenant_rows` FOR ALL TO `mm_app`：`USING` 与 `WITH CHECK` 都是
  `tenant_id = current_setting('mm.tenant_id', true) AND app_id =
  current_setting('mm.app_id', true)`。未设置时 `current_setting(..., true)`
  为 NULL，谓词为 NULL：**读到 0 行、写入报错**（fail-closed）。
- `mm_maintenance_rows` FOR ALL TO `mm_maintenance`, `mm_owner`：
  `USING (true) WITH CHECK (true)`。

**租户设置**：`Engine` 类上的 `begin` 事件监听器在**每个事务开始**执行
`SELECT set_config('mm.tenant_id', %(tenant_id)s, true),
set_config('mm.app_id', %(app_id)s, true)`——两个值是绑定参数，
`is_local = true` 为事务级：PostgreSQL 自己在 COMMIT / ROLLBACK 时丢弃，
池里复用的物理连接不可能把上一个租户的设置带进下一个事务。SQLAlchemy 2.x
对 autobegin 也会触发 `begin`，Session 与 `engine.connect()` 两条路径走
同一个钩子，所以**不经过 Session 的 Core 语句同样被覆盖**；Session commit
后继续查询会开新事务并再次触发。例外：`AUTOCOMMIT` 隔离级别的连接没有
事务可挂设置，不得用于租户工作（策略会 fail-closed）。

**`system_scope` 在 PG 上的路由**：在 `system_scope` 内创建的 Session，
若其 bind 是 PG 引擎且 RLS 开启，会被重新绑定到注册的**维护引擎**
（`register_maintenance_url` / 环境变量 `MM_MAINTENANCE_DATABASE_URL`，
按"去掉口令的应用库 URL"为键）。未注册时是硬错误而不是静默看到 0 行
（否则队列扫描会"看见空队列"）。不走 Session 的调用点
（`api._verify_single_tenant_startup` 的裸连接）改用
`rls.system_engine_for(engine)`；迁移内部的 5 个调用点以迁移登录
（owner / 超级用户）运行——DDL 需要所有权，owner 由维护策略放行。
`process_pending` 的扫描会话移到 `system_scope` 内创建。

**运行期姿态校验**（`MemoryEngine._ensure_db`，PG 且 `MM_PG_RLS != 0`）：
应用引擎不再 `create_all`；它要求 schema 已由 owner 侧
`upgrade_database` 建好、每张租户表 ENABLE+FORCE 且两条策略在位、
**当前登录不是超级用户、无 BYPASSRLS、不是表 owner、是 app 角色**——
超级用户会无声通过所有策略，让第二道防线看起来开着实际是空的。

**读过滤模式** `MM_TENANT_READ_FILTER`（任务 5 的 A/B/C）：`app`（默认，
应用层注入 + RLS）、`rls`（跳过注入，RLS 独自过滤读；非 PG bind 直接
拒绝）、`none`（无任何读过滤，**仅测量**：还需
`MM_ALLOW_UNFILTERED_MEASUREMENT=1`，server 拒绝启动）。

**审计器新规则**：引擎自己的那条 set_config 语句精确放行；其他任何设置
`mm.*` GUC、`SET/RESET ROLE`、`SET SESSION AUTHORIZATION`、`row_security`
操作一律**硬性违规**（带内 RLS 绕过向量）；GRANT / CREATE|DROP POLICY /
CREATE ROLE 单语句形态作为不读行的 DDL 放行（真正阻止 app 角色发出它们
的是所有权——数据库报 `must be owner`）。

## 工具速查

| 工具 | 场景 |
|---|---|
| `tenant_scope(t, a)` | 一切入口（API、server、worker 按 job、测试） |
| `system_scope(reason)` | 跨租户队列扫描/维护（唯一旁路，必须写 reason） |
| `no_tenant_scope()` | 仅测试：模拟入口缺失，验证 fail-closed |
| `TenantAwareExecutor` / `submit_with_tenant` | 线程池 |
| `engine_write_scope()` | 引擎认可写入口内部标记（Publisher、仓储认可 mutator）；**应用/插件代码不得使用** |
| `fixture_write_scope(reason)` | 仅测试夹具：直接造信念数据时显式进入（reason 必填） |
| `no_audit_scope()` | 仅测试：机制类测试里故意发原生 SQL 时使用 |
| `rls.system_engine_for(engine)` | PG 上 `system_scope` 内不走 Session 的裸连接：取维护引擎 |
| `rls.register_maintenance_url(app_url, maint_url)` | 宿主启动时注册维护登录（或设 `MM_MAINTENANCE_DATABASE_URL`） |
| `read_filter_override(mode)` | 仅测试/测量：临时切换读过滤模式 |

## 错误类型

- `TenantContextMissing`：拿不到租户（fail-closed，最常见）。
- `TenantMismatchError`：上下文与会话/对象不一致（会话跨租户、
  租户列被改、写入了别的租户的行、`session.merge()` 试图并入他租户行）。
- `WriteGuardViolation`：信念类表被写在认可入口之外（白名单语义，
  **任何来源**的越界写都拦——包括未来的 server/脚本/插件）。
- `AuditViolation`：SQL 审计器规则违规（测试期）。
- `RowSecurityError`：PG 行级安全姿态缺失或错误（角色/策略/所有权、
  运行登录为超级用户或 owner、`system_scope` 缺维护引擎）——全部 fail-closed。
