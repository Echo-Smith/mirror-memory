# K2 语义分类问题详单

引擎规则修不了、需要改提示词或接受的抽取器语义缺陷。按可修性分三档：

## A 档：提示词可修（改 k2_system.txt 即可）

### A1. 跨域歧义动词分类错误
**现象**：`moved_to` 在雇佣语境被归为居住。
"I worked for Northstar. I moved to Redwood." — 第二句是换雇主，但抽取器
产出 `moved_to → lives_in`，Redwood 变成住址，与 Atlas（雇佣）并列渲染。
**同病**：`returned_to`（居住/雇佣）、`based in`（居住/工作）、
`went to`（事件/教育）。
**为何引擎修不了**：动词本身无歧义，歧义来自宾语类型。引擎没有宾语类型
推理，同义词表只能钉死一个领域，钉哪个都会错另一个。
**提示词修法**：在 `k2_system.txt` 加域判定规则——当上文已出现
employment 谓词（work/employ/job/company）时，`moved_to/returned_to/based`
的谓词归为 `works_at` 而非 `lives_in`；反之归 `lives_in`。要求模型输出
`predicate` 时声明 `domain: residence | employment | education | ...`。

### A2. 裸介词作谓词
**现象**：`at` 作谓词（"I am at Cedar Bank"）。已映射到 `works_at`，但
"I am at home" 就错了。
**为何引擎修不了**：`at` 不含语义，全靠宾语类型。
**提示词修法**：禁止输出介词作谓词；必须从 policy 的规范谓词集里选。
提示词已要求 `predicate: the relationship verb (likes, has, went_to ...)`，
但模型仍自由造词。改成闭集列举 + 违规时要求模型重新生成。

### A3. 教育槽位动词变体
**现象**：`study at` / `attend` / `go to [school]` / `enrolled in` 同指
教育属性，但 `attend` 被归为 `attended`（event）而非 `studies_at`
（current_state）。"I attend Lakeview High" 变成事件而非当前状态。
**为何引擎修不了**：`attend` 语义上既可作事件（"attended a workshop"）
也可作持续状态（"attend Lakeview High"）。判断依据是宾语是否为机构名。
**提示词修法**：宾语是机构名（学校/大学）时用 `studies_at`；宾语是
活动/事件名时用 `attended`。在示例里加对比对。

### A4. 多部 claim 未拆分
**现象**："I live in Berlin and work at Globex." 抽取器可能产出单条
claim，object="Berlin and work at Globex"，或只取前半。
**为何引擎修不了**：拆分是抽取器的职责，引擎拿到的是已合并的 object。
**提示词修法**：强化 "Each claim = one atomic piece of information" 规则，
加复合句示例要求输出两条 claim。

### A5. 否定辖域
**现象**："I don't like coffee but I love tea." 抽取器可能产出
`likes coffee and tea`（否定辖域错）。
**为何引擎修不了**：否定辖域是句法解析问题，发生在抽取阶段。
**提示词修法**：加否定辖域示例，要求拆成 `dislikes coffee` +
`likes tea` 两条。

## B 档：需模型能力（提示词难改，可能要接受）

### B1. 代词消解到错误主体
**现象**："My sister lives in Rome. She works at a bank." `she` 应归
third_party，但模型有时归 user，导致 sister 的事实被存成用户事实。
**现状**：`subject: user | third_party` 字段已存在，但模型判定不稳。
**可能修法**：提示词强化代词消解规则 + 多轮示例。若仍不稳，接受为已知
失败率并在报告中单列。

### B2. 相对日期解析
**现象**："I moved here last spring." 需解析为具体日期（如 2026-03-01）。
已注入当前日期到 payload，但"last spring"这类模糊相对日期模型仍可能
解析偏差（偏早/偏晚几个月）。
**现状**：`valid_from/valid_to` 字段已要求 ISO 日期，注入了当前日期。
**可能修法**：接受为日期精度损失；或加后处理把 `last spring` 映射到
季节区间（3/1–5/31）而非单点日期。

### B3. 隐含状态终止
**现象**："I stopped drinking coffee." 模型可能只产出 `drinks coffee` 的
否定而非明确的 `terminates(coffee)`。引擎的撤回标记过滤只认字面标记词，
模型换了说法（"coffee is off the table"）就漏。
**现状**：读侧撤回标记表（avoided/gave up/lost interest/stopped…）已覆盖
常见表达。
**可能修法**：提示词要求抽取器对状态终止输出显式
`relation: terminates` 或 `polarity: negative`，引擎据此处理而非靠词表。

### B4. 年龄表述归一化
**现象**："I just turned 29" / "I am 29 now" / "my birthday last week, now 29"
应统一为 `age=29`。模型对 `turned N` 类表述的 predicate/object 不一致
（可能产出 `turned` / `became` / `is` + `29_years_old` / `29`）。
**现状**：`became` 已映射到 `profession`（可能误伤）；`turned` 未映射。
**可能修法**：提示词加年龄表述规范，要求 `predicate: is, object: 29` 固定。

## C 档：接受为架构边界

### C1. 同义词表的覆盖天花板
**现象**：抽取器造词无上界（`transitioned_to`、`took_a_role_at`、
`stepped_into`…），同义词表追不上。本轮已按观测补了 14 个变体 + 时态/
前缀归一化，但新动词仍会出现。
**为何修不了**：无 embedding 或语义相似度就无法泛化到未见动词。
**建议**：接受为已知失败模式，报告中单列 "unmapped predicate rate"；
后续若上 embedding，此项自动消失。

### C2. 宾语粒度不一致
**现象**：同一实体的表述粒度不一（`acme_corp` / `acme` / `Acme Corporation`），
`canonicalize_object` 只做小写和去冠词，无法归一化。
**为何修不了**：实体归一化是独立问题（entity linking），不在抽取范畴。
**建议**：接受；或后续引入实体归一化层。

### C3. 隐含事实不抽取
**现象**："My home is in Toronto." 隐含 `lives_in toronto`，模型可能产出
`home_is_in` 而非 `lives_in`。冷启动逃逸已保证这类 turn 被 K2 尝试，
但谓词仍可能不准。
**建议**：提示词加隐含事实映射表（home → lives_in, job → works_as 等），
可移入 A 档。

---

## 优先级建议

| 档 | 项 | 预期收益 | 成本 |
|---|---|---|---|
| A1 | 跨域动词分类 | stale_state/replacement 泄漏 | 低（改提示词） |
| A2 | 裸介词禁用 | replacement 泄漏 | 低 |
| A3 | 教育槽位变体 | replacement 泄漏 | 低 |
| A4 | 复合句拆分 | extraction recall | 低 |
| A5 | 否定辖域 | contradiction | 低 |
| B3 | 显式 polarity | preference_evolution | 中 |
| B4 | 年龄归一 | replacement | 低 |
| C1 | 同义词天花板 | 持续小量泄漏 | 高（需 embedding） |

A 档五项都改 `config/prompts/k2_system.txt`，一轮提示词迭代可全试。
B 档需要提示词 + 引擎 schema 双改。C 档建议接受并在报告中披露。
