# M3 第三批补丁完成报告（2026-10-29）

分支 `feat/m3-purpose`，起点 `2251b66`（第三批验收版），HEAD `cff9301`。**未合并 main。**

## 一、结论

四项全部完成，验收通过。**M3 关闭。**

| 项 | 状态 | 验收证据 |
|---|---|---|
| 1 会话摘要按 purpose 分行 | ✅ `f9c9566` | 同一会话在 Q、P 下 observe 产生两行（各行单一 purpose）；撤销 P 后 Q 的 recall 回退路径零 P 原文 |
| 2 快照/摘要类合并文本记录输入出处并集 | ✅ `25e683c` | 快照 source_purposes_json = 输入并集（含前一版快照与作业 purpose）；K3 按 AND 门控：P 撤销后含 P 输入的快照不再被使用；digest 同规则 |
| 3 残余入口 | ✅ `64dcee2` | correct/restore 对不可见信念返回与不存在完全相同的形态（None/False → server 404）、零事件；rejected 按规则过滤；验收跨用途场景逐 id 1..N 扫描 |
| 4 先 limit 后过滤的饥饿 | ✅ `64dcee2` | P 200 条较新 + Q 5 条较旧：撤销 P 后 recall / get_beliefs / panel / 摘要回退 / export 全部返回 Q 内容、零 P 哨兵——规则改为 SQL 内先过滤再 limit |

**dual-db CI（run 40904363924，`cff9301`）两条腿全绿**：SQLite **1327 passed / 70 skipped**，PostgreSQL **1394 passed / 3 skipped**（两侧 tests=1397，0 failed）。性能 run **40899119830**（6/6 VALID）：SQLite 全量（交集 1200）最小 **1.069** / 中位 **1.073**（预算 ≤ 1.10 ✅），PG longrun（mode A，18 ID）最小 **1.008** / 中位 **1.008**（预算 ≤ 1.05 ✅）。

## 二、各任务设计决定与证据

**任务 1——会话摘要分行**（`f9c9566`）：`append_session_summary` 以 `(user, session, source_purpose)` 取原行——同 purpose 追加、跨 purpose 新行。**无需唯一约束调整**（表本无 (user, session) 级唯一约束，只有主键；任务书「必要时调整」不适用），也无需新迁移步骤。存量行保持原状（`''` 或 legacy 标签）；旧行与新行天然按 purpose 隔离。测试：Q 先 P 后同一会话 → 两行；撤销 P 后 Q 回退零 P 原文。

**任务 2——合并文本的输入出处并集**（`25e683c`）
- 原则落地：离散条目 OR（任一出处可见即可见，第三批已实现）；**合并文本 AND（全部输入 purpose 可见才可见）**。`mm_snapshots` 与 `mm_evidence_digests` 新增 `source_purposes_json`（迁移第 10 步，LATEST_VERSION=10，NOT NULL DEFAULT '[]'，可重入）。
- 快照并集的构成（worker 发布 payload 携带）：**可见性过滤后实际读取的信念**的出处（链接 + created/corrected 事件，一次批量探针）、**前一版活跃快照的并集**（K3 把它作为上下文读过——它的输入也是输入）、**作业自身 purpose**。`_load_evidence` 返回可见信念行，并集从真实输入计算。
- digest 盖章：`apply_evidence_compaction` 在折叠时把图中全部证据的 source_purpose 与压缩入口的 purpose 求并集写入（增量累积）。
- AND 门控：K3 两处快照读取按 `merged_visible`（列表内全部 purpose 可见）；**空列表（存量行）回落到单 purpose**——升级不造成无声消失。explain 的 digest 披露与 recall 的证据意图提示按 digest 自身并集门控（预补丁 digest 回落整图检查）。
- **worker 其他派生写入排查**：SYNTHESIZE 只写快照（publisher 在该转换下从不创建信念——代码注释即此约束）；TIER_TRANSITION / REJECT 只移动或关闭既有行；另一处派生内容就是 digest（已覆盖）。无遗漏。
- 测试：Q 作业在 P 有效时生成快照（并集 = {P, Q}）；P 撤销后 K3 拒用该快照（Q 仍有效也不行——AND 规则）；digest 提示按并集门控；Step10 从 v9 升级。

**任务 3——残余入口**（`64dcee2`）：correct / restore 在引擎层先判可见性（规则在引擎手里），不可见 → 返回与不存在**完全相同**的 None / False（server 404），**零事件零写入**（事件计数断言）；rejected 过滤加进 `list_rejected_beliefs` 并从 renderer / panel 传入。验收跨用途场景补齐三入口：对 id 1..N 逐一 correct / restore / explain / belief_history，与幽灵 id 逐字节对照，且事件数不动。

**任务 4——饥饿**（`64dcee2`）：测试先复现（P 200 条较新 + Q 5 条较旧，撤销后 post-limit 过滤让 Q 完全不可见——get_beliefs 空列表）。修法选**方案 A：SQL 先过滤再 limit**——A2 规则表达为单条 SQL 谓词 `_visibility_sql_predicate`（三个 EXISTS：可见链接 / 存在链接 / created-corrected 可见事件 / 存在事件 + 无出处→空标记分支，全部带复合租户键，命中第 9 步的租户前缀复合索引），接入 `list_active_beliefs` 与 recall_candidates 两个切片；摘要回退在 WHERE 里过滤。Python 后置过滤删除（recall 槽逻辑现在只见过滤后的信念）。顺带修正 `provenance_purposes` 的探针——原先按裸 belief_id 匹配（与 export 连接同类的跨租户同号缺陷），现带完整复合键。

**过程中修复的回归**：① digest 盖章初版引用了尚未构建的 `evidence_rows`（UnboundLocalError，12 个压缩测试抓到）——移到行加载之后；② `evidence_summary` 重构时丢了无规则路径的 digest 获取——恢复；③ 快照并集初版**逐信念**探测（publish 事务内 2N 条语句，CI 的 50 轮撤销竞速发布线程因此超过 20 s join 上限，dual-db PG 腿失败一次）——改为一次批量调用（`cff9301`），发布路径固定 +2 条语句与 N 无关。

## 三、与任务书不同的决定

1. 摘要分行**无需**迁移第 10 步做约束调整（表无 (user, session) 级唯一约束）；第 10 步改用于任务 2 的两列。
2. 快照并集包含**前一版快照的并集**（任务书列了「前一版快照的出处」，按并集语义展开为累积并集）。
3. correct / restore 的门控放**引擎层**而非 repository（正确性路径经 publisher，规则在引擎已解析；行为与 404 等价不变）。
4. 饥饿选方案 A（SQL 半连接）而非分批补取：谓词一次过滤、无二次查询循环，且第 9 步索引正好服务。
5. 第三批遗留项全部关闭：correct/restore/rejected 三个入口已门控并纳入验收场景。

## 四、性能数据（run 40899119830，@ `27069e8`；其后的 `cff9301` 只减语句，下列数字是最终 HEAD 的上界）

**SQLite 全量**（交集 1200，main 403f38d vs HEAD 同 run 交错 ×3，6/6 VALID）：

| 轮 | main | HEAD |
|---|---|---|
| 1 | 159.6 | 171.2 |
| 2 | 159.8 | 169.3 |
| 3 | 158.4 | 171.7 |
| 最小 | 158.4 | 169.3（**1.069**，预算 ≤1.10 ✅） |
| 中位 | 159.6 | 171.2（**1.073**） |

**PG longrun**（mode A，18 ID）：

| 轮 | main | HEAD |
|---|---|---|
| 1 | 168.0 | 168.3 |
| 2 | 165.9 | 169.3 |
| 3 | 167.0 | 167.3 |
| 最小 | 165.9 | 167.3（**1.008**，预算 ≤1.05 ✅） |
| 中位 | 167.0 | 168.3（**1.008**） |

夹具成本：与第三批相同（懒签发 memo，零新增）。产品成本：本批在 publish 事务新增固定 +2 条批量出处探针 + 1 条前快照读取；读路径的可见性从「limit 后 Python 过滤」改为「SQL 内谓词」——语句数不变（第三批已计入），单查询变重但走第 9 步索引。SQLite 剩余 +6.9~7.3% 主要是饥饿修复把可见性成本移进了被测查询本身（第三批的数字把它算在查询之后、测不到的行数上）；PG 持平（1.008）。

## 五、新增 system_scope 使用点；待评审白名单完整清单（M3 合入 main 前一并评审）

**本批新增 system_scope 使用点：0 个。**

完整待评审清单（前批累计 + 本批）：
- **system_scope（3 条）**：`_count_legacy_jobs`（批 1，迁移计数）；`purge_expired_audit`（批 3，审计维护租户枚举）；`_backfill_column`（M2 已评审，批 2/3 扩展了复合主键行值匹配与 marker 参数——重审知悉）。
- **写入函数（writer 段，5 条）**：`grant_purpose`（批 1）、`record_purpose_invocation`（批 1）、`record_access`（批 2，audit 类）、`purge_access_audit`（批 2，audit 类）、`enqueue_job` 批注（批 2 wrapper 段，host-facing 空调用方）。
- **包装函数（wrappers 段，2 条）**：`grant_bench_purpose`（批 2）、`enqueue_job`（批 2）。
- **写入类别（1 条）**：`audit`（`mm_access_audit` 专属写入类与作用域，批 2）。
- 第三批补丁对白名单**零新增**（未引新 system_scope 点、未加写入函数）。

## 六、实际耗时；git diff --stat；提交号

实际耗时：约 **5 小时**（2026-10-29 21:00 – 2026-10-30 02:00 本地，含 3 轮 CI 等待与一次 Docker PG 容器重启）。git diff --stat（2251b66..cff9301）：**17 files changed, 651 insertions(+), 44 deletions(-)**。提交：`f9c9566`（任务 1）、`25e683c`（任务 2）、`64dcee2`（任务 3+4）、`27069e8`（perf 触发分支）、`cff9301`（批量探针修复）。CI：dual-db run **40904363924**（`cff9301`，全绿）；性能 run **40899119830**（6/6 VALID；27069e8）。首轮 dual-db（40899114495）PG 腿 1 失败（竞速线程超时，见第二节），`cff9301` 修复后全绿。
