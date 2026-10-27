# 方案：B 模式（RLS 承担读过滤）下 PG 审计器的读规则

状态：**方案，未实施**（M2 第二批任务 5 要求"如果建议 B，先写方案"；终验按预设规则建议保持 A，本方案留作切换 B 时的实施依据）。

## 问题

审计器现行读规则：SELECT / UPDATE / DELETE 触及的每张租户表都必须带 `tenant_id = ?` 与
`app_id = ?` 绑定谓词。A 模式由 `with_loader_criteria` 注入满足它；B 模式跳过注入，
读语句形态上不带谓词，规则会把每条合法读都判为违规。测量时只能关审计器，这正是
B 不能直接上生产的缺口。

## 规则

B 模式下（`MM_TENANT_READ_FILTER=rls` 且 bind 为 PostgreSQL）：

1. **读语句（SELECT）**：触租户表时，要求「当前连接的**当前事务**已设置租户」。判定
   依据不是语句文本，而是事务上下文——`core.rls` 的 `begin` 钩子在执行 `set_config`
   后，在 `Connection.info` 记一个本事务标记；`commit` / `rollback` / `close` 事件清除。
   审计器在 `before_cursor_execute` 读该标记：无标记且语句触租户表 → `AuditViolation`
   （与 RLS 的 fail-closed 一致：这样的语句在数据库里也只会读到 0 行）。零额外往返。
2. **`system_scope` 内的语句**：照旧记日志放行，但新增一项核对——连接必须属于**维护
   引擎**（`conn.engine is rls.maintenance_engine_for(...)`）；在 app 引擎上出现的
   system_scope 读语句是违规（它会静默读到空集）。
3. **写语句**：规则不变。INSERT 仍要求列清单含两列租户列；UPDATE / DELETE 仍要求谓词
   ——B 保留全部写侧校验，复合主键让 ORM 工作单元语句天然带谓词；RLS 的 WITH CHECK
   是第二道。
4. **判词缓存**：键加入模式维度 `(mode, sql)`；A 与 B 对同一条读语句判词不同。
5. **非 PG bind**（B 下 SQLite 回退注入）：沿用 A 规则。

## 反例（实施时必须有）

- B 模式、app 引擎、无租户事务中的 `SELECT ... FROM mm_beliefs` → 违规。
- B 模式、租户事务中同一语句 → 放行；同一连接 commit 后新事务未重设（人为构造）→ 违规。
- B 模式、`system_scope` 在 **app** 引擎上的读 → 违规；在维护引擎上 → 放行并记日志。
- B 模式、UPDATE 不带谓词 → 仍违规（写规则未放松）。

## 不做的事

不把 RLS 的存在当作"审计器可以关掉"的理由：审计器是测试期对**应用层**行为的证明，
RLS 是数据库层的第二道；两者判定的对象不同。
