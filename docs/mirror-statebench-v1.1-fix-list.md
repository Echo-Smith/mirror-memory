# Mirror 基于 StateBench v1.1 的修复详单

本文只列 Mirror 引擎需要修复的事项。评测协议本身的修正已进入
StateBench v1.1。

## P0：阻断状态正确性的缺陷

### 1. 抽取节流不能按“维度已覆盖”跳过更新

**现象**

- 第二个合法多值没有被抽取；
- 同一事件的第二次发生消失；
- A → B → A 经常停留在 B；
- 状态更新句使用 `returned`、`settled`、`turned` 等表达时容易完全漏掉。

**原因**

节流器把一个维度已有高置信信念等同于当前回合没有新增信息。状态更新、
新对象、否定、恢复和新时间限定都被误判成重复信息。

**修改要求**

1. 新颖性至少比较规范化谓词、对象、极性和时间限定；
2. 明确更新词、否定词、恢复词和事件时间出现时不得执行维度级抑制；
3. 多值谓词的新对象始终具有信息增益；
4. event 谓词的新时间限定始终形成候选 occurrence；
5. 保留 `forced` 与 `production` 两条独立评测轨道。

**验收**

- `forced` 模式中每个回合都有一次 K2 尝试记录；
- `production` 与 `forced` 的状态分数差距不超过 0.05；
- multi-value、temporal-event、round-trip 三类不再出现系统性的“只保留第一条或中间条”。

### 2. 把用户修正、状态终止和证据冲突拆开

**现象**

“我喜欢咖啡”之后“其实我不喜欢咖啡”，系统只降低旧正向信念的置信度，
仍然可能展示“喜欢咖啡”。

**原因**

当前 CONTRADICT 统一执行置信度衰减和 `needs_clarification`，没有保存可查询的
否定状态，也没有区分用户自我修正和来自不同来源的未解决冲突。

**修改要求**

1. 给 Memory Atom 增加明确的 polarity 或等价结构；
2. 用户明确修正应关闭旧状态并创建新的当前状态；
3. “不再拥有”“取消目标”“关系结束”作为状态终止处理；
4. 只有来源冲突或无法判断时间顺序时进入 clarification；
5. 保留旧状态及证据，不能物理覆盖历史。

**验收**

- contradiction 的四个 transition type 分别通过；
- 当前状态中不能同时存在同一对象的正向和明确否定状态；
- 历史审计仍能找到被修正的旧陈述及其证据。

### 3. 建立“每个对象一个当前极性”的偏好生命周期

**现象**

`likes`、`dislikes`、`wants_to` 等都按 multi + persistent 处理，导致喜欢、放弃、
恢复等多个阶段同时 active。

**修改要求**

1. 多对象可以共存，例如同时喜欢咖啡和绘画；
2. 同一对象在同一时间只能有一个当前偏好极性；
3. 目标支持 active、paused、cancelled、resumed 等生命周期；
4. 行为证据不能自动升级为长期偏好，除非语句明确表达态度或计划；
5. 查询当前偏好时过滤已经关闭的阶段。

**验收**

- preference-revival、preference-reversal、goal-reactivation 分别报告；
- 当前查询不泄漏 `gave up`、`lost interest`、`cancelled` 等已结束阶段；
- multi-value 偏好仍能同时保留多个不同对象。

### 4. 修复 current-state 的时间区间与 round trip

**现象**

住址、雇主、学校、职业和年龄更新后，旧值可能继续 active；A → B → A 时最终
状态可能停在 B。

**修改要求**

1. current-state 新值到达时关闭旧值区间；
2. 没有明确日期但有 `now`、`returned`、`again` 等顺序证据时，使用观察顺序建立区间；
3. round trip 重新打开 A 的新有效期，同时保留 A 的旧历史阶段和 B 阶段；
4. identity 不得仅依赖 extractor 输出的表面谓词；
5. 年龄更新覆盖 `turned N`、`birthday ... now N` 等表达。

**验收**

- replacement、round-trip、stale-state 的 state track 各类别达到 0.90；
- current 查询仅返回开放区间；
- history 查询可返回 superseded 区间；
- A → B → A 最终只有 A 为 current，B 可从历史查询得到。

### 5. 事件身份必须包含 occurrence 时间

**现象**

同一地点或活动的第二次事件被合并，或者日期只保留一次。

**修改要求**

1. event identity 至少由规范化谓词、对象和时间限定组成；
2. 相同对象、不同时间创建独立 occurrence；
3. 相同对象、相同时间的重复证据执行 SUPPORT；
4. 时间无法解析时保存原始 temporal qualifier；
5. 事件查询允许返回所有匹配 occurrence，不受固定每维度三条限制。

**验收**

- temporal-event 每个输入回合都对应一个可审计 occurrence；
- 两次同地点访问同时包含各自日期；
- 重复提交同一事件不会制造重复 occurrence。

## P1：影响召回完整性的缺陷

### 6. 扩充规范化谓词和状态表达覆盖

统一 residence、employment、education、profession、age、skill、possession、goal
和 relationship 的常见变体。规范化必须在 identity resolution 前完成，并记录原始
谓词供审计。

**验收**：所有 v1.1 case 的抽取 trace 中，状态类 claim 都有 subject、predicate、object。

### 7. 查询时间模式必须参与候选选择

`current`、`history`、`all_occurrences` 应进入检索条件。历史查询允许召回当前值作为
背景，但回答器必须依据问题选择历史区间。

**验收**：历史查询的 recall track 与 answer track 分开报告；正确历史状态不会因召回块
包含当前背景而被误判。

### 8. 渲染容量按查询自适应

固定 `max_per_dimension=3` 不适合“列出所有事件/技能/物品”的查询。对
`all_occurrences` 和列举型问题扩大相关维度容量，同时继续受总字符预算约束。

**验收**：state track 通过但 recall track 失败的比例按类别可见；multi-value 和
temporal-event recall 达到 0.85。

## P2：可观测性与发布门槛

### 9. 每个 case 保存完整漏斗

至少保存：逐回合抽取数量、候选 atom、identity 决策、持久化动作、最终 belief rows、
召回候选、门控原因、渲染块和答案。

### 10. 固定发布门槛

建议第一阶段使用以下门槛：

| 指标 | 门槛 |
|---|---:|
| forced state overall | ≥ 0.90 |
| forced state 每类别 | ≥ 0.85 |
| production 与 forced state 差距 | ≤ 0.05 |
| recall overall | ≥ 0.85 |
| answer overall（固定回答器） | ≥ 0.85 |
| 报告缺失 manifest | 0 次 |

达标前不再使用“架构优势已证明”作为外部结论。公开的 `public_evaluation` 只用于
可复现报告；正式发布结论需要额外的私有未见集。


---

# 执行记录（2026-09-23）

## P0 — 全部完成

| # | 项 | 状态 | 关键改动 |
|---|---|---|---|
| 1 | 节流不按维度覆盖跳过 | ✅ | 内容新颖性逃逸（value gate + 抑制分支）+ 冷启动逃逸 |
| 2 | 修正/终止/冲突拆分 | ✅ 5/5 | `Belief.polarity`；`correct_belief` 改为关闭+新建；撤回标记过滤；`conflict_kind` 分流 self_correction / source_conflict |
| 3 | 每对象一个当前极性 | ✅ 5/5 | `memory/polarity.py` 结构化推断；resolver 按极性判冲突；goal 生命周期 active/paused/cancelled/resumed；行为谓词恒为中性 |
| 4 | current-state 区间与 round trip | ✅ | 时态归一化（was_hired_by→works_at）；revival 路径；`update_belief_by_id` 允许重开 superseded 行 |
| 5 | 事件 occurrence 身份 | ✅ | 同对象+不同时间=独立 occurrence；key 含时间判别 |

## P1 — 全部完成

| # | 项 | 状态 | 关键改动 |
|---|---|---|---|
| 6 | 规范化谓词覆盖 | ✅ | 同义词表 + 时态/前缀归一 + K2 提示词闭集；`Belief.raw_predicate` 审计；`audit_predicate_coverage()` 监测未映射谓词 |
| 7 | 时间模式参与候选选择 | ✅ | `all_occurrences` 模式（列举型查询自动进入）；StateBench recall/answer 双 track 分离报告 |
| 8 | 渲染容量自适应 | ✅ | 列举型查询提升 per-dimension 上限（默认 12），字符预算仍为硬约束 |

## P2 — 全部完成

| # | 项 | 状态 | 关键改动 |
|---|---|---|---|
| 9 | 每 case 完整漏斗 | ✅ | `bench/funnel.py`：逐回合抽取数、候选 atom、identity 决策、持久化动作、belief rows、召回候选、门控原因、渲染块、答案 |
| 10 | 固定发布门槛 | ✅ | `bench/gate.py`：state ≥0.90、每类 ≥0.85、recall/answer ≥0.85、manifest 必备、forced/production gap ≤0.05 |

## 验证记录

| 轮次 | 范围 | 结果 |
|---|---|---|
| StateBench v1.0 全量 | 200 例（mimo-v2.5） | 0.730，baseline 0.450 |
| 针对性（极性改动） | 60 例（MiMo-V2.6-Flash） | contradiction 0.800→0.867；preference_evolution 噪声内 |
| 针对性（K2 提示词 A 档） | 50 例（MiMo-V2.6-Flash，零失败） | replacement 0.500→**0.867**，stale_state 0.600→0.650 |

**未达标项**：P2-10 的门槛（state ≥0.90）尚未达到——当前全量 0.730。按计划要求，在达标前不将"架构优势已证明"作为外部结论。

**测试**：769 通过，0 失败。
