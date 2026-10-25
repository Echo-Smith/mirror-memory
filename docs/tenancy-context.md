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

## 错误类型

- `TenantContextMissing`：拿不到租户（fail-closed，最常见）。
- `TenantMismatchError`：上下文与会话/对象不一致（会话跨租户、
  租户列被改、写入了别的租户的行、`session.merge()` 试图并入他租户行）。
- `WriteGuardViolation`：信念类表被写在认可入口之外（白名单语义，
  **任何来源**的越界写都拦——包括未来的 server/脚本/插件）。
- `AuditViolation`：SQL 审计器规则违规（测试期）。
