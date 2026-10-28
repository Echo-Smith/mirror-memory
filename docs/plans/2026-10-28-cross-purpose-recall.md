# 跨用途召回方案（M3 第二批任务 4，只写方案、不实施）

日期：2026-10-28　分支：feat/m3-purpose　状态：待用户确认

## 0. 问题

证据、账本事件、快照自第二批起带 `source_purpose`（在哪个 purpose 的授权下收集；`''` = 来源未记录）。要决定三件事：

1. 以 purpose **Q** 召回时，能否用到在 purpose **P** 下收集的记忆？
2. 撤销 P 之后，**只由 P 的证据支撑**的信念在 Q 下是否隐藏？
3. **混合证据**（P 和 Q 都有）的信念怎么办？

附带：来源未知（`''`）的证据怎么处理；对召回查询、索引、性能、端云同步的影响。

## 1. 现状（第二批结束时）

- 可见性只看两层授权：`authorize(user, Q, recall)` 通过 + 功能开关开 → 该用户**全部**活跃信念都是候选。`source_purpose` 只记录、不参与任何过滤。
- 一个信念的"来源 purpose 集合"不存列，由证据链接推导（`core.export.source_purposes_for_beliefs`），explain / export 已输出。
- 用 `correct` 产生的信念没有证据链接，但它的 `created` / `corrected` 账本事件带 purpose；K3 快照带任务 purpose。
- 撤销是即时的（无缓存），但撤销 P 只影响"以 P 调用"的路径；以 Q 调用仍能读到 P 收集的内容——这正是本方案要决定的事。

## 2. 两个方案

### 方案 A：按证据来源过滤（召回时判定）

召回（及 panel / get_beliefs / explain / export）在 Q 下只接纳**至少有一条"可见出处"**的信念。出处 = 证据链接指向的证据行，或该信念的账本事件；一条出处可见，当且仅当它的 `source_purpose` 属于 **可读来源集合 R(Q)**。

两种 R(Q) 定义：

- **A1 严格**：R(Q) = {Q}。每个 purpose 是一个筒仓，记忆完全不跨用途。隐私最强，但产品价值大减——support 用不到 api 学到的任何东西；多数部署不会要这个。
- **A2 活跃授权**：R(Q) = 该用户**当前有效**（未撤销、未过期）的全部 purpose。含义：撤销 P = 撤回"在 P 下收集"的同意，P 单独支撑的记忆在**所有**用途下消失；只要 P 仍有效，Q 可以用 P 收集的记忆。这符合用户的直觉模型："我收回了 P 的授权，P 收集的东西就不该再被用"。

混合证据：采用 **OR 规则**——任一出处可见即信念可见。理由：这条信念同时也是通过 Q（或其他仍有效的来源）合法得知的，一个被撤销的来源不应抹掉共享事实；但 **P 独有的那些证据不得再展示**，所以 explain / export 在 Q 下只列可见出处（证据 id 列表、事件）。AND 规则（任一来源被撤销就隐藏）会让一次撤销抹掉大量共同事实，过度隐藏，不推荐。

来源未知（`''`）：它表示"M3 之前、按全局记忆开关的同意收集"或"未经授权入口直接写入"。建议：**`''` 在用户全局开关为开时视为可见**（这就是它被收集时的同意制度，开关本身仍在两层授权里检查）；想把存量记忆绑定到某个具体 purpose 的运维，用迁移第 6/7 步的 `legacy_job_purpose` 打标签——打标后它就是普通 purpose，受撤销约束。不建议默认把 `''` 判为不可见：升级后全部存量记忆会无声消失。

### 方案 B：授权里声明可读取的来源 purpose 集合

给 `mm_purpose_grants` 加 `readable_sources_json`：签发 Q 时声明 Q 可以读哪些来源 purpose 的记忆（`["*"]` = 全部，`["self"]` = 只读自己，或显式列表）。召回时 R(Q) = Q ∪ readable_sources(Q)。

单独的 B 有一个漏洞：Q 被声明可读 P，之后 P 被撤销——纯 B 下 Q 仍能读到 P 收集的记忆，因为"Q 的读权限"没有被撤。这违反"撤销后任何路径都拿不到记忆"。所以 B **必须叠加 A2 的活跃来源门**：R(Q) = (Q ∪ readable_sources(Q)) ∩ 当前有效 purpose（∪ `''` 按上节规则）。

B 相对 A 多出的能力是**限制**而不是放开：签发方可以让 Q 读不到某些仍然有效的来源（数据最小化，例如"广告用途不得读健康对话收集的记忆"）。代价：授权行多一个字段（签发 / 更新的 CAS、事件表、v2 的 `sync_authorization` 同步字段都要带上），签发方要会填；默认值决定一切——`["*"]` 时 B 退化为 A2，`["self"]` 时退化为 A1。

### 对比

| 维度 | A1 严格 | A2 活跃授权 | B（叠加 A2 门） |
|---|---|---|---|
| 撤销 P 后 P 独有记忆在 Q 下 | 本来就不可见 | 隐藏 ✅ | 隐藏 ✅ |
| P 有效时 Q 能否用 P 的记忆 | 否 | 是 | 由签发方声明 |
| 混合证据 | 只看 Q 的证据 | OR：可见，但只展示可见出处 | 同 A2 |
| 来源未知 `''` | 按全局开关 | 按全局开关 | 按全局开关 |
| 授权表改动 | 无 | 无 | +1 列 + CAS/事件/同步 |
| 签发方负担 | 无 | 无 | 要理解来源集合语义 |
| 召回额外查询 | 同下 | +1 条（取有效 purpose 集）+ 半连接 | 同 A2 |
| 端云同步 | 同下 | 需同步 `source_purpose` + 授权 | 还要同步 `readable_sources` |

## 3. 对召回查询、索引、性能、端云同步的影响（A2 / B 共同）

**查询**。`recall_candidates`（及 panel / get_beliefs 的 `list_active_beliefs`）加一个半连接谓词：

```sql
EXISTS (SELECT 1 FROM mm_belief_evidence_links l
        JOIN mm_evidence e ON e.tenant_id = l.tenant_id AND e.app_id = l.app_id AND e.id = l.evidence_id
        WHERE l.tenant_id = b.tenant_id AND l.app_id = b.app_id AND l.belief_id = b.id
          AND (e.source_purpose IN (:live) OR e.source_purpose = ''))
OR EXISTS (SELECT 1 FROM mm_belief_events ev
        WHERE ev.tenant_id = b.tenant_id AND ev.app_id = b.app_id AND ev.belief_id = b.id
          AND (ev.source_purpose IN (:live) OR ev.source_purpose = ''))
```

`:live` 每次请求取一次：`SELECT purpose FROM mm_purpose_grants WHERE user_id = :u AND revoked_at IS NULL AND (expires_at IS NULL OR expires_at > :now)`——主键前缀精确命中，+1 条语句/召回；不缓存到事务之外（撤销即时生效的要求不变）。explain / export 在 Q 下按同一谓词过滤证据 id 列表与事件。

**索引**。`mm_belief_evidence_links(tenant_id, app_id, belief_id)` 与 `mm_belief_events(tenant_id, app_id, belief_id)`——两表的 `belief_id` 目前是单列 `index=True`（M2 批 2 的租户前缀规则只处理了 `user_id`）；实施时按同一规则换成租户前缀复合索引（迁移一步，CONCURRENTLY）。`mm_evidence` 走主键。不建议在 `source_purpose` 上建索引：谓词是按信念半连接，不是按 purpose 扫表。

**性能**。候选集经查询感知准入后是几十到几百条，半连接每条两次索引探查；估计每次召回 +1 条语句、+5–15% 查询耗时，具体以实施批的 m3-perf-acceptance 为准。另一条路是**写时物化**：给 `mm_beliefs` 加 `source_purposes_json`，在 link_evidence / _append_event 时维护，召回只做一次 `json` 包含判断——读快、写慢、要回填；第一版不建议，等测量说话。

**端云同步**。可见性是 (有效授权集, 出处) 的确定函数，两端各自算就能得到一致结果，前提是：① `source_purpose` 随证据 / 事件 / 快照同步（第二批列已加，同步协议要带上）；② 授权（含撤销）已在同步范围（v2 `sync_authorization`）；③ B 方案还要同步 `readable_sources`。端侧撤销先于云侧到达时，端侧先隐藏、云侧随后隐藏，中间窗口云侧 recall 仍可见——与现在撤销传播的窗口相同，不是新问题。

## 4. 代理推荐

**推荐：A2 的语义作为默认规则，用 B 的数据模型承载**——即给授权行加 `readable_sources`（默认 `["*"]`），召回可见性 = (Q ∪ readable_sources(Q)) ∩ 当前有效 purpose，`''` 按全局开关，混合证据 OR 规则 + explain/export 只展示可见出处。

理由：
1. 活跃来源门是撤销语义的硬要求，A2 和 B 都必须有，它决定"撤销后拿不到"；这一点没有第三种选择。
2. 默认 `["*"]` 让第一版行为就是 A2（最符合用户直觉、不需要签发方学新东西），而需要数据最小化的部署可以在不改代码的前提下收紧——改默认值比日后加列便宜得多。
3. 召回查询与索引改动两方案完全一样，B 只多一个字段，边际成本低。

**需要用户决定的一点**：export 是否也按 Q 过滤。按同一规则过滤最一致（Q 拿不到的，Q 也不该导出）；但这意味着 P 被撤销后，P 独有的证据**没有任何 purpose 能导出**——"用户本人的数据访问权由产品侧另行处理"（M3 页的第 2 个决定）若要靠 export 实现，需要一个特殊的"数据主体"purpose 对 `readable_sources` 免除活跃门。建议先按统一规则实施，数据主体导出作为单独的产品决定。

## 5. 实施清单（确认后）

- `core/provenance.py`：`visibility(session, user_id, purpose) -> VisibilityRule`（有效 purpose 集 + `''` 规则），每请求一次。
- `repository.recall_candidates` / `list_active_beliefs` / `explain_belief` / `core.export.read_page`：接受规则并过滤（信念半连接；证据与事件按行过滤）。
- 迁移：links / events 的 `belief_id` 索引换租户前缀复合索引；B 的 `readable_sources_json`（默认 `["*"]`）+ CAS/事件/同步。
- 测试：P 独有信念在 P 撤销后于 Q 下隐藏、混合信念仍可见但 explain 不列 P 的证据、Q 独有不受影响、`''` 按开关、`correct` 产生的无证据信念按事件出处、readable_sources 显式收紧生效、撤销并发 50 轮下无"撤销已提交却仍可见"。
- 性能：m3-perf-acceptance，SQLite 全量 + PG longrun，夹具成本与产品成本分开。
