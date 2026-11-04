# M4 任务 0：删除覆盖面清单（表 × 范围 × 处理方式）

日期：2026-10-30　基线：main @ aed0639（M3 已合入）　状态：核对完成，无缺口

## 结论

1. **快照失效现状无缺口**：`forget_belief` 与 `delete_memories` 都在**同一删除事务**里删除该用户的全部快照（`Snapshot.user_id == user` 批量删）——比「读时跳过、由 worker 重建」更强（重建 = 下一个完成作业自然重建）。`forget_session` 无需动快照：快照由信念合成（K3 读取的是快照与信念，不是会话摘要），会话摘要是独立的原始文本层。本批以哨兵测试锁定两条路径（信念删除后 worker 重建物零哨兵；会话删除后回退与快照零哨兵）。
2. 授权行（consent / preference / purpose grants + events）、访问审计（只存 id）、users 行、墓碑本身：**全部保留**——M3 的「删除 ≠ 撤销授权」决定不变。
3. M4 在此清单上加两件东西：**独立删除日志**（任务 1）的指纹来自这里的「删除 / 保留」划分；**删除代数**（任务 2）接替墓碑行上「每用户 max(generation)」的临时计数（墓碑保留为审计）。

## 覆盖面矩阵

| 表 | forget_belief（belief） | forget_session（session） | delete_memories（user） | 理由 |
|---|---|---|---|---|
| mm_beliefs | 删该行 | 保留 | 删全部 | 信念本体 |
| mm_belief_events | 删该信念的 | 保留 | 删全部 | 账本含 detail 与 evidence ids |
| mm_belief_evidence_links | 删该信念的 | 保留 | 删全部 | 图边 |
| mm_evidence | 仅删孤儿（无链接的） | 保留 | 删全部 | 有信念指向的证据是其余信念的出处 |
| mm_evidence_digests | 删该信念的 | 保留 | 删全部 | 折叠文本 |
| mm_memory_transitions | 删该信念的 | 保留 | 删全部 | 分层审计 |
| mm_belief_versions / identities | 删该槽整链 | 保留 | 删全部 | 版本链存明文 object |
| mm_proposal_log | 删 target/created 指向该信念的 | 保留 | 删全部 | payload 可能含内容 |
| mm_snapshots | **删该用户全部**（= 作废，worker 重建） | 保留（快照不含摘要内容） | 删全部 | 合成自由文本，AND 门控无法覆盖删除 |
| mm_evolution_jobs | pending/running → cancelled | 保留 | 删全部 | 队列任务的输入是被删记忆 |
| mm_session_summaries | 保留 | **删 (user, session) 全部 purpose 行** | 删全部 | 原始消息文本；M3 补丁后按 purpose 分行 |
| mm_extraction_stats | 保留 | 保留 | 删全部 | 运行遥测（无信念级归属）；user 范围整删 |
| mm_intervention_events | 保留 | 保留 | 删全部 | 遥测；user 范围整删 |
| mm_deletion_tombstones | **追加**（审计，generation+1） | —（见下） | **追加** | 内容无关的「删过」记录 |
| mm_access_audit | 保留 | 保留 | 保留 | 只存 id，不是内容；审计不是记忆 |
| mm_users / 授权行 / purpose grants + events | 保留 | 保留 | 保留 | 删除 ≠ 撤销授权（M3 决定） |
| mm_memory_preferences | 保留 | 保留 | 保留 | 同上 |

注：forget_session 今天只删摘要（其 API 语义即「忘记该会话的原文」）；任务 3 的重放按同一覆盖面执行。

## M4 之后的增量（本批实现）

- 每次删除的**指纹列表**（表名 + id + created_at 的哈希，不含内容）随墓碑一同写入独立删除日志——上表所有「删」动作的行都要进指纹。
- 墓碑上的 generation 由新的**每用户代数表**（CAS）接替权威来源（日志为准，重放对齐）。
- 快照增加合成时代数：低于用户当前代数的快照读时跳过（备份恢复 / 重放后的兜底，与「删全部」互为冗余防线）。
