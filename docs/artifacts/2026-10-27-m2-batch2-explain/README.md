# M2 第二批任务 4：EXPLAIN 复核与索引决定

数据：100 租户 × 1 万信念（100 用户/租户，用户 ID `u000..u099` **跨租户重复**——多租户
真实命名空间，也是 M1 保留的单列 `user_id` 索引的最坏情形），事件 2:1、证据 1:1、
链接 1:1、任务 1/用户。工具 `tools/explain_review.py`，逐条 `EXPLAIN (ANALYZE, BUFFERS)`，
三种形态：**C** 应用谓词 + 超级用户（无 RLS）、**A** 应用谓词 + RLS（app 角色，今日生产
形态）、**B** 仅 RLS 过滤（任务 5 候选）。本机 Docker PG16 原始表格见
`explain_review_local.md`（计划形状是决策输入；耗时仅作参考，CI 工件见任务 5 工作流）。

## 计划摘要（基线 = 当前模型索引）

| 查询 | C / A / B 计划 | 走的索引 | 判定 |
|---|---|---|---|
| recall 相关切片（ILIKE，ORDER BY last_evidence_at LIMIT 150） | Bitmap Index/Heap | `uq_mm_beliefs_user_key` (tenant, app, user, key) 前缀 | ✅ 已走复合唯一约束；三模式一致 |
| recall 最近切片（LIMIT 25） | Bitmap Index/Heap + Sort | 同上 | ✅ 100 行取 25，0.1 ms |
| 按 id 取信念（session.get） | Index Scan | `mm_beliefs_pkey` (tenant, app, id) | ✅ |
| 按信念取事件 | Index Scan | `ix_mm_belief_events_belief_id` | ✅ belief_id 全局唯一，选择性足够 |
| 按信念取证据（links ⋈ evidence） | Index Scan ×2 | `uq_mm_belief_evidence_link` + `mm_evidence_pkey` | ✅ |
| 任务 claim（条件 UPDATE） | Index Scan | `mm_evolution_jobs_pkey` | ✅ |
| **pending 任务扫描（维护角色，无租户）** | **Seq Scan**（1 万行丢 9000，随规模线性） | 无 | ❌ → 采纳部分索引 |
| 删除路径：按用户删信念 | Bitmap | `uq_mm_beliefs_user_key` | ✅ |
| 删除路径：按用户删事件 | BitmapAnd | `ix_mm_belief_events_user_id` ∧ `mm_belief_events_pkey` | ⚠️ 用户 ID 跨租户重复时单列 user_id 命中 1 万行再与主键求交，3 ms @ 100 万行；可接受 |

RLS 的 `tenant_id = current_setting('mm.tenant_id', true)` 谓词可走索引（函数 STABLE），
A/B 与 C 的计划形状相同；B 模式（无应用谓词）同样命中租户前缀索引。

## 决定

1. **新增** `ix_mm_evolution_jobs_pending`：`(created_at) WHERE status = 'pending'`，
   声明在模型上（新库 `create_all` 直接建，SQLite 同样受益），已有 PG 库由迁移
   **第 4 步**用 `CREATE INDEX CONCURRENTLY` 补建（INVALID 残留先删再建，可重入，
   `lock_timeout` 生效）。效果：Seq Scan → Index Scan，过滤丢弃 9000 → 0，缓冲
   462 → 34 页。
2. **不补** M1 删掉的 36 个单列租户索引，也**不加**按租户前缀的复合索引：所有
   按租户+用户/主键的查找已由现有复合主键和唯一约束服务，补索引只增加写放大
   （M1 的教训）。
3. **记录为可选、未采纳**（工件里有前后对比）：
   - `(tenant, app, user_id, last_evidence_at DESC)` on beliefs：最近切片变纯 Index
     Scan（316 → 56 页，0.11 → 0.03 ms）；但基线已是 0.1 ms 的 100 行索引取数，
     recall 不是瓶颈，不值一个 100 万行表上的额外索引。
   - 用 `(tenant, app, user_id)` **替换**各表单列 `ix_*_user_id`：若用户 ID 命名空间
     确认跨租户共享，这是结构性修法（索引数不变、选择性更好）；当前只在按用户删除
     路径可测到（3 ms），留待命名空间策略明确后决定。
   - `(tenant, app, belief_id)` on links / events：计划未改善（单列 belief_id 已足够）。

## 顺带修复

`_pkey_build_index`（第一批）把主键构建索引挂在 `Base.metadata` 上：PG 上做过第 2 步主键
换索引后，同进程再 `create_all` 会在事务内发 `CREATE UNIQUE INDEX CONCURRENTLY … __pkey_build`
而失败。改为挂在分离的表副本上，加回归测试。

## 补丁任务 3（2026-10-27 追加）：单列 `user_id` 索引替换为 `(tenant_id, app_id, user_id)`

决定 3 中的"可选未采纳"项，用户评估后要求**替换**（不新增）。模型层用一条规则
（`tenancy.install_tenant_user_indexes`）对全部 14 张带 `user_id` 的租户表统一生效，
迁移**第 5 步**对存量 PG 库 `CREATE INDEX CONCURRENTLY` 新索引 → `DROP INDEX CONCURRENTLY`
旧索引（INVALID 残留先删再建，可重入）。原始表格见 `user_index_before_after_local.md`
（同一 500 万行数据集上先换回单列索引跑一遍，再换成复合索引跑一遍）。

C 模式关键行（before → after）：

| 查询 | before | after |
|---|---|---|
| 按用户删事件（200 万行表） | BitmapAnd `ix_*_user_id` ∧ 主键，8.4 ms，1036 次磁盘读 | `ix_*_tenant_user` 单索引，0.44 ms，0 读 |
| 按用户删证据 | `uq_mm_evidence_user_ref` 1.74 ms，207 读 | 0.35 ms，0 读 |
| /stats：extraction_stats 按用户 ORDER BY created_at LIMIT 20 | Index Scan on `user_id`，**过滤掉 297 行他租户** | 0 行过滤 |
| K3 快照摘要按用户 | BitmapAnd `user_id` ∧ 主键，23 页 | Index Scan，9 页 |
| 按用户删信念 | `uq_mm_beliefs_user_key`（本就带租户前缀） | 同级（0.53 → 0.11 ms，噪声量级） |
| 证据按 (user, ref) 重查找 | `uq_mm_evidence_user_ref` | 不变（唯一约束已覆盖） |

非租户查询是否依赖旧的 `user_id` 索引：**没有**。`system_scope` 内的三类调用点——队列扫描
（按 status / created_at）、单租户启动探针（DISTINCT tenant_id, app_id）、迁移内部（按主键
回填 / 全表计数）——都不按 `user_id` 过滤（grep 校验）；所有按用户的语句都在租户上下文内
发出，带租户谓词，复合索引前缀精确匹配。
