# 删除日志运维手册（M4）

独立删除日志（`core/deletion.py`）、恢复后自动重放（`core/replay.py`）、墓碑同步（`core/tombstones.py`）的运维口径。配置项见 `docs/config-guide.md`；本文只写"出事了怎么办"。

## 1. 备份与恢复顺序：日志的恢复点不能早于主库

- **备份顺序**：先备主库，后备日志（或同时）。日志永远只能**比主库新**——它记录的是主库将来要补做的删除。
- **恢复顺序**：先恢复主库，再核对日志。主库水位 `(db_id, applied_seq)` 随备份回滚，与日志尾部比对：
  - 水位 = 尾部 → 正常；
  - 水位 < 尾部 → 启动时自动重放（fail-closed，重放完成前拒绝服务）；
  - 水位 < 检查点 seq → **拒绝启动**（见 `DeletionLogPruned`，第 5 节）——备份比日志保留期还老，被裁掉的删除无法重放；
  - 水位 > 尾部 → 拒绝启动（`DeletionLogStale`）——日志比主库旧，违反恢复点规则。
- 端侧整机恢复：**主库和日志目录必须一起恢复**（`deletions/` 与库文件同源）；只恢复主库会让新日志与新 db_id 配对失败。

## 2. adopt 与 --new-lineage

| 场景 | 用什么 |
|---|---|
| 主库是「M4 部署前的备份」恢复出来的（有数据、无水位行），日志里有这条部署线的历史 | `python -m mirror_memory.deletion_adopt --database-url <url> --db-id <日志的 db_id>`（schema 低于 11 时先跑迁移、跳过铸行，再写入指定 db_id） |
| 运维**明确**要开一条全新部署线：旧日志里的删除**不会**重放到新库 | 迁移时传 `new_lineage=True`（adopt 命令加 `--new-lineage`）；会打印 `WARNING: NEW-LINEAGE: ... The old log's deletions will NOT be replayed onto this database.` |

不要用 --new-lineage 来"消除"报错——它是业务决定，不是恢复手段。

## 3. 一个日志只服务一条部署线

日志按 db_id 配对：一个 db_id = 一次数据库生命。**不要**让两个主库共用一个日志目录/日志库：

- 启动对账会按创世条目认亲（`DeletionLogUnavailable`：日志没有这个 db_id 的创世条目）；
- 迁移铸行前会检查日志里是否已有别的部署线的创世条目（拒绝并列出，见第 5 节）；
- 端云同步的对端关系也存在 `mm_sync_peers`（按对端 db_id 记游标）。

## 4. 端侧整机恢复的局限与离线宽限期

- 恢复后端侧**先追平云端再开门**（启动追平，游标存在 `mm_sync_peers`，逐批推进，中断可续）。
- 连不上云端时按 `sync.offline_grace_days`（默认 7 天，从**上次成功追平**算起，时间存在主库里——恢复会把它带回过去，宽限期更早到期，这是有意的）：
  - 宽限期内：可服务（状态 offline_grace）；
  - 剩余 ≤ `sync.reminder_days`（默认 2 天）：`sync_status().reminder = True`，产品层应提示用户联网；
  - 超期：读入口抛 `SyncCatchupRequired`（`user_message` 可直接展示），写入口照常（本地删除稍后随同步上行）；
  - 一旦联网追平成功，**立即恢复服务，无需重启**。
- 局限：端侧删除若尚未上行（导出前）就整机恢复，这些删除还在本地日志里，恢复后照常重放/上行，不丢。

## 5. 每一种拒绝启动报错：含义与处理

以下原文可直接 `grep` 日志定位。所有 `WatermarkMissing` / `DeletionLogUnavailable` / `DeletionLogPruned` 均为 fail-closed：先修复，再启动。

| 报错（原文片段） | 含义 | 处理 |
|---|---|---|
| `the main database holds tenant data but has no deletion watermark row: it looks like a restore from a PRE-M4 backup ... run the adopt command ... python -m mirror_memory.deletion_adopt --db-id <db_id>` | M4 部署前的备份恢复了 | 按 `--db-id`（日志里这条部署线的 db_id）走 adopt；确认日志位置正确 |
| `the deletion log already holds genesis entries -- this deployment line has run M4 before: {db_id}... (genesis {时间}) ...`（迁移时） | 迁移铸行前发现日志已有别的部署线 | 走 adopt 认领对应 db_id；确要开新线才用 new_lineage |
| `no deletion log found for database {db_id}... (the log has no genesis entry for this db_id): deletions cannot be reconciled.` | 日志里没有这个库的创世条目——配错日志了（或 backend=none 但已有已应用条目：`...the watermark shows N applied deletion entries...`） | 把 `deletion_log.directory / url` 指回正确的日志；确认没删错日志文件 |
| `the deletion log is OLDER than the main database (watermark {n} > log tail {m} ...)` | 日志比主库旧（违反恢复点规则） | 找回较新的日志备份；**不要**删主库数据迁就日志 |
| `the deletion log was compacted through seq {cp}, but this database's watermark is {applied}: the backup is older than the log's retention ...`（`DeletionLogPruned`） | 备份比日志保留期还老，被裁掉的删除无法重放 | 恢复一个检查点**之后**的备份并 adopt；或业务决定开新部署线（new_lineage） |
| `another instance is replaying the deletion log; entries are refused until it finishes`（`ReplayPending`） | 重放正在别处进行 | 等待；持续不结束再查持锁进程 |
| `the deletion log is unreachable ({异常}): refusing to mint ...`（迁移时） | 迁移读不到日志（配置错误/网络） | 先修日志配置再迁移 |
| `deletion_log.backend=file with a non-SQLite database requires deletion_log.directory ...` / `deletion_log.backend=postgres requires deletion_log.url` | 配置缺失 | 补配置后重启 |
| `deletion_log.retention_days ({n}) is shorter than backup_retention_days ({m}) ...` | 保留期短于备份保留期 | 调大 retention 或调小 backup_retention |
| `memory sync is catching up with the peer; entries are refused until it finishes`（`SyncCatchingUp`） | 端侧正在追平云端 | 重试即可；持续数分钟不结束再查对端连通 |
| `the last successful sync catch-up was at {时间}: the offline grace period has expired...`（`SyncCatchupRequired`） | 端侧超期未追平 | 联网后任意一次调用会自动追平并恢复；无需重启 |

## 6. 保留期与压缩

- `deletion_log.retention_days`（默认 35）≥ `backup_retention_days`（默认 30）——启动强制；压缩由 worker 按 `prune_interval_seconds`（默认 1 天）后台执行。
- 压缩 = `prune(before_seq)`：删除保留期外的条目，写**检查点条目**（seq=被裁到的最大 seq，哈希=当时的链哈希），哈希链从检查点接续，创世条目永不裁。校验（`verify_chain`）对检查点只验衔接、不重算哈希。
- 云端 PG 日志的 DELETE 由 **maintenance 连接**执行（`deletion_log.maintenance_url`）；app 角色仍只有 INSERT/SELECT。
- 与同步的交互：对端游标落在检查点之前 → `export_tombstones` 抛 `TombstoneCursorPruned`，**要求对端全量重建**，绝不静默跳过。

## 7. 指标与告警阈值建议

| 指标 | 含义 | 告警建议 |
|---|---|---|
| `deletion_replay_current_seq` / `deletion_replay_tail_seq` | 重放进度 | `tail - current` 持续 > 0 超 10 分钟告警（重放卡住） |
| `deletion_replay_active` | 是否处于重放模式 | =1 持续 > 30 分钟告警（重放模式是拒绝服务的） |
| `sync_catching_up` | 是否处于追平模式 | =1 持续 > 30 分钟告警 |
| `sync_since_last_catchup_seconds` | 距上次追平 | > `offline_grace_days - reminder_days`（默认 5 天）提醒；> `offline_grace_days` 页面已拒绝，运营侧告警 |
| `deletion_reconcile_failures` | worker 对账失败次数（按退避重试） | **任何一次**都告警：对账失败=worker 停工 |
| `tombstones_applied` / `tombstones_skipped` | 墓碑应用成功 / 幂等跳过 | 参考值，无告警 |
| `tombstones_rejected_<reason>` | 整批拒绝次数（format / hash / tenant / user / sequence / apply） | **任何一次**都告警：正常链路不应出现，出现即对端实现有 bug 或有伪造流量 |
| `tombstones_export_skipped` | 导出时跳过的不可迁移条目（batch 2 之前的信念删除） | 首次出现告警（说明存在跨代日志） |

指标为进程内 gauge/counter（`core.metrics.snapshot() / gauges()`），由维护端点暴露；重启归零，用于告警而非审计。
