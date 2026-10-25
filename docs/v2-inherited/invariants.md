# Mirror Memory v2 — 不变量与事务设计

## 6 条不可破坏的不变量

| # | 不变量 | 违反后果 | 验证点 |
|---|---|---|---|
| **I1** | 一个 Job 在任意时刻最多被一个 Worker 持有 | 双写、数据竞争、重复记忆 | Claim 阶段 CAS |
| **I2** | 已删除（revoke/delete）的 Scope 不能产生新 Atom | 数据复活、隐私泄露 | Publish 阶段屏障 |
| **I3** | complete/fail 只能由当前 lease holder 提交 | 旧 Worker 覆盖新 Worker 结果 | Publish 阶段 fencing |
| **I4** | 同一 source_event_id 在同一 Scope 内全局唯一 | 幂等性破坏、重复记忆 | Claim 阶段唯一约束 |
| **I5** | 授权 Purpose 与请求 Purpose 必须匹配 | 越权操作 | Claim + Publish |
| **I6** | Atom 的 source_evidence_id 必须属于同一 Scope | 跨 Scope 数据引用 | Claim 阶段校验 |

## 三阶段模型：Claim → Compute → Publish

```
┌─────────────────────────────────────────────────────────────────┐
│                        Claim (事务 T1)                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ 检查授权     │  │ CAS 获取租约  │  │ 加载 Evidence          │  │
│  │ version>0   │  │ state+token  │  │ scope_id 匹配          │  │
│  │ not expired │  │ scope_id     │  │ not deleted            │  │
│  │ op allowed  │  │ +1, owner    │  │                        │  │
│  │ purpose ok  │  │ RETURNING    │  │                        │  │
│  └─────────────┘  └──────────────┘  └────────────────────────┘  │
│  事务提交 → 获取 lease_token, evidence_id                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Compute (无事务)                            │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ 来源校验     │  │ 规则提取      │  │ 模型调用（可选）        │  │
│  │ role=user?  │  │ 确定性/M2    │  │ 事务外执行              │  │
│  │ injection?  │  │ patterns     │  │                        │  │
│  └─────────────┘  └──────────────┘  └────────────────────────┘  │
│  纯计算，无 DB 写入，可安全失败                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Publish (事务 T2)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ 发布前屏障     │  │ 写入 Atom    │  │ CAS 更新 View + 完成   │ │
│  │ 重检授权      │  │ + ViewHead   │  │ lease_token 匹配       │ │
│  │ 重检删除代际   │  │              │  │ state='running'        │ │
│  │ 重检 lease    │  │              │  │ → 'ready'              │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
│  事务提交 → Job 状态变为 ready                                    │
└─────────────────────────────────────────────────────────────────┘
```

## 各阶段的 SQL 条件

### Claim 阶段 (T1)

```sql
-- CAS 租约获取
UPDATE jobs SET
    lease_token = lease_token + 1,
    lease_owner = :worker_id,
    lease_expires_at = :now + :ttl,
    state = 'running'
WHERE id = :job_id
  AND scope_id = :expected_scope_id        -- I6: Scope 归属
  AND (state = 'pending'
       OR (state = 'running'
           AND lease_expires_at < :now))    -- I1: CAS 原子性
RETURNING lease_token;

-- 授权检查
SELECT auth_version, auth_expires_at, auth_allowed_operations, auth_purpose
FROM scope_control
WHERE tenant_id = :t AND app_id = :a AND subject_id = :s
  AND auth_version > 0
  AND auth_expires_at > :now               -- I2: 未过期
  AND :operation = ANY(auth_allowed_operations)  -- I5: 操作允许
  AND (auth_purpose IS NULL OR auth_purpose = :purpose);  -- I5: Purpose 匹配

-- Evidence 加载
SELECT * FROM evidence
WHERE id = :evidence_id
  AND scope_id = :scope_id                 -- I6: Scope 归属
  AND deletion_generation <= :scope_generation;  -- I2: 未删除
```

### Publish 阶段 (T2)

```sql
-- 发布前屏障：重检授权
-- (同 Claim 阶段的授权检查，但必须在 T2 事务内)

-- 发布前屏障：重检删除代际
SELECT deletion_generation FROM scope_control
WHERE id = :scope_id
  AND deletion_generation <= :evidence_deletion_generation;  -- I2

-- 发布前屏障：重检 lease
SELECT lease_token FROM jobs
WHERE id = :job_id
  AND lease_token = :my_token              -- I1: 仍持有 lease
  AND lease_owner = :my_worker_id;

-- 写入 Atom
INSERT INTO memory_atoms (...) VALUES (...);

-- CAS 更新 ViewHead
UPDATE view_heads SET
    revision = revision + 1,
    content_hash = :hash
WHERE scope_id = :scope_id
  AND revision = :expected_revision;       -- CAS

-- 完成 Job (fencing)
UPDATE jobs SET
    state = 'ready',
    reason = :reason
WHERE id = :job_id
  AND state = 'running'
  AND lease_token = :my_token;             -- I3: fencing
```

## 事务边界总结

| 阶段 | 事务 | 写入 | 失败语义 |
|---|---|---|---|
| Claim | T1 (短事务) | jobs.state, jobs.lease_token | lease_failed / auth_denied |
| Compute | 无事务 | 无 | source_invalid / extraction_failed |
| Publish | T2 (短事务) | atoms, view_heads, jobs.state | cancelled / lease_lost |

关键约束：
- T1 和 T2 是独立事务，中间是无状态的 Compute
- T2 必须在 T1 的 lease_token 仍然有效时提交
- 如果 T2 失败，Job 保留在 "running" 状态，等待 lease 过期后可被 reclaim