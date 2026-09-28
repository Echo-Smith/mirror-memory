# Mirror 当前代码分析（2026-09-24）

## 分析范围与实测基线

- 代码：`main`，`5393053`；分析前工作区干净。
- 全量测试：`.venv/bin/pytest -q` → **769 passed，1 skipped，1 warning**，9.56 秒。
- 本轮 gate/CLI/测试修改后复验：**776 passed，1 skipped，1 warning**；修改过的 Python 文件 Ruff 全通过，`git diff --check` 通过。
- 静态检查：`.venv/bin/ruff check src tests --output-format concise` → **63 errors**，其中 49 项可自动修复。
- StateBench v1.1：deterministic、200 例 → state **25/200 = 0.125**，recall **2/200 = 0.010**，answer 未运行。报告和 manifest 存于 `/tmp/mirror-statebench-current.*`。
- deterministic 模式在 `bench/statebench_v1_1_runner.py:277-287` 关闭 LLM；上述分数**不能评价**最近的 K2 提示词修复或生产抽取能力。仓库记录的 0.730 是 StateBench v1.0 的模型运行结果，不能直接与 v1.1 deterministic 分数比较。

## 结论

Mirror 的 **StateBench 缺陷修复阶段**已取得实质进展：节流、新状态表达、极性、事件身份、查询时间模式、漏斗和 manifest 均有代码与测试。`docs/mirror-statebench-v1.1-fix-list.md` 的“P0/P1/P2 完成”描述的是这批局部修复，**不等于** `docs/plans/2026-09-22-runtime-closure.md` 中的 Runtime Closure P0 已完成。

运行时仍由多个入口直接写 `Belief`。`Publisher` 目前只是 K3 snapshot 的实际写入路径；它缺少原子 revision claim、幂等提交和失败回滚。时间字段和极性虽然进入 `Belief`，首次 CREATE 仍不写独立时间列，A → B → A 仍通过复活旧行覆盖原阶段。检索也仍依赖对当前 `Belief` 行的筛选与修补，而非消费可回放的版本账本。

因此，当前阶段可以描述为：**状态语义的局部正确性与评测可观测性明显增强；统一、可审计、可并发的 Epistemic Runtime 尚未闭环。**

这两个目标需要分开排期。发布分数的主要杠杆是状态区间、极性和查询时间选择；Publisher
原子性是并发与重试时必须满足的安全属性，单线程 StateBench 不会测到它。完整
Observation/Assertion 拆分、RecallQuery 编译器和 Maintenance Planner 是演进方向，
不应成为本轮发布门的前置条件。

### 失败集中度比总分更有诊断力

本次 deterministic 报告共有 175 个 state 失败。五个整桶为零的 transition 合计
**108/175（61.7%）**：

| transition | 通过/总数 | 直接相关的状态契约 |
|---|---:|---|
| `current_value_replacement` | 0/30 | 关闭旧 interval，打开新 interval |
| `current_value_revival` | 0/30 | 同一值回归时新建 interval，不覆盖最早阶段 |
| `preference_reversal` | 0/20 | 同一对象的 current polarity 切换 |
| `query_time_selection` | 0/20 | 查询时选取对应有效期；它是读取操作，不是写入 intent |
| `goal_cancellation` | 0/8 | `END_CURRENT` 关闭目标状态 |

对照组是 `goal_reactivation` 10/10、`explicit_correction` 3/4。这支持优先检查状态与
查询语义，而不是把 deterministic 的每项失败都归因于抽取。它仍不能预测 K2 的 forced
成绩；必须先跑同一 v1.1 集上的 forced/production 成对评测。forced 总分即使偏低，
也不能单独证明抽取是瓶颈，因为写入和查询错误也会压低它；需要逐例漏斗拆因。

## 与 9 月 22 日分析相比，已经变化的部分

| 能力 | 当前进展 | 边界 |
|---|---|---|
| 抽取节流 | 加入冷启动与内容新颖性逃逸；不再仅按维度覆盖跳过 | 生产表现仍需同一 v1.1 集上的 forced/production 成对运行 |
| 谓词与身份 | 扩展规范化，保留 `raw_predicate`；修复更新时的 key 碰撞 | `relation.py:118` 的 subject 仍固定为相同 |
| 极性与目标 | `Belief.polarity`、`lifecycle_state`、self correction/source conflict 分流 | 撤回仍有文本标记过滤，状态轴仍混在单行 Belief 上 |
| 事件与时间查询 | occurrence key 纳入时间信息；支持 `all_occurrences` 和列举式渲染扩容 | 首次 CREATE 没有写 `observed_at/valid_from/valid_to` 列；all-occurrences 的候选 SQL 仍只取 active |
| 诊断 | per-case funnel、三轨报告、manifest、发布阈值函数已加入 | 原发布阈值函数与 v1.1 报告不兼容，本轮已修复；完整漏斗须单独调用 `capture_funnel()`，v1.1 runner 不自动保存 |

这些进展是局部能力，不应被旧分析中的“完全缺失”措辞掩盖。例如 polarity 已经结构化，correction 也已有关闭旧行并创建新行的路径。现存问题是这些能力没有在统一事务和版本身份下串联。

## 当前最重要的代码事实

### 1. 写入边界仍有多条旁路（P0）

`extraction/pipeline.py:494-654` 对 SUPPORT、UPDATE、CREATE、CONTRADICT 直接调用 repository，随后另行记录 evidence edge；round trip 还直接改写中间 `Belief` 行。`api.py:508-605` 的 targeted forget/correct 直接调用 repository。`extraction/verification.py:344-350` 的 confirm/reject 也是直接 mutation。只有 `worker/evolution.py:232-248` 将 snapshot proposal 交给 `Publisher`。

因此 README 中“every state change is a proposal; only Publisher commits”的描述与生产路径不一致。当前测试全部通过，只能说明这些路径在已覆盖情景下可运行，不能证明 single-writer 不变量。

### 2. Publisher 仍不具备原子发布协议（P0）

`core/proposal.py:58-86` 的 proposal 没有持久 `proposal_id`、幂等键、actor 或必填 expected revision；`claimed_revision=0` 是默认值。`core/publisher.py:107-111` 在 claimed revision 为 0 时跳过检查；非零时也是读 revision 再写，并非带 expected revision 的数据库 CAS。`publish()` 在 handler 异常后返回 `commit_error`，没有 savepoint 或 rollback（`publisher.py:73-93`）。

`evidence_ids` 只检查已找到的跨用户行（`publisher.py:114-126`）；不存在的 ID 不会被拒绝，handler 也没有用它自动建立 evidence edge。repository helper 自行 bump revision；Publisher 没有一次 transition 一次 revision 的事务边界。`repository.py:121-145` 的 revision increment 是原子加法，但它不使前面的 gate 成为原子 CAS。

这意味着 concurrent/stale、重复投递、handler 中途失败、证据缺失这些情形尚无可靠 commit receipt。需要把写入协议和入口迁移作为一个连续阶段完成。

### 3. 首次状态和历史阶段仍不可完整表达（P0）

Pipeline 已经把 `CandidateAtom.observed_at/valid_from/valid_to` 传给 UPDATE（`pipeline.py:445-455, 522-550`），但 CREATE 调用 `record_claim()` 时没有这些参数（`pipeline.py:607-645`）；`record_claim()` 签名和新 `Belief` 构造也不接收时间列（`repository.py:831-1000`）。时间字符串可能留在 `value_json`，但 interval 查询依赖独立列。

`Belief` 仍受 `(user_id, key)` 唯一约束（`models.py:100`）。`update_belief_by_id()` 遇到 A → B → A 会重新激活旧 A 并覆盖 `valid_from/valid_to`（`repository.py:1281-1325`）。`tests/test_longmemeval_regressions.py:133-152` 仍明确要求复用原 A 的 row id。这个契约能让“现在是什么”恢复正确，但不能同时保留 A 的第一段和第二段有效期。

`relation.py:118` 把 `same_subject` 固定为 `True`。读取端的 `_keep_newest_per_single_slot()`（`repository.py:599-634`）会隐藏同谓词的较旧 active 行；它改善当前回答，但也说明写入端尚未保证单值槽只有一个 current 版本。

### 4. 遗忘与证据治理仍存在回流路径（P0）

`forget_belief()` 清理 belief、事件、关联边、孤立 evidence，并使 snapshot/job 失效（`repository.py:1748-1789`）；这是比旧版更完整的治理。但它不处理 `SessionSummary`。启用 summary 功能时，pipeline 将原始对话片段写入 summary（`pipeline.py:201-207, 326-342`），renderer 在 belief 匹配不足时直接读 summary 文本（`renderer.py:404-478`）。`SessionSummary` 没有 belief lineage、validity 或 forget 标记（`models.py:326-341`）。因此 targeted forget 后，包含相同内容的 summary 仍可能被 fallback 召回。该功能默认关闭（`config/schema.py:235-239`），风险发生在显式启用时。

`record_evidence()` 对同一 `(user_id, ref)` 刷新 content、session、method、authority 和 observed_at（`repository.py:195-242`）。它适合去重，但原始来源事实不是不可变的。当前没有 Observation/Assertion/Transition 三级身份，无法明确区分“来源变了”和“抽取解释变了”。

### 5. 检索已更灵活，但治理规则分散（P1）

`renderer.py:171-199` 推断 current/history/all-occurrences；`repository.py:483-570` 做 lexical admission、时间和 read-side 单值过滤；`retrieval.py` 打分；renderer 再执行 L4/confirmation gate、diversity 和预算。没有贯穿这些阶段的 `RecallQuery/RecallResult` 契约，也没有统一的 belief 与 summary governance gate。

值得注意的是 `recall_candidates()` 对 `historical` 查询取 active+superseded，但 `all_occurrences` 只取 active（`repository.py:513-517`），所以该模式并不总能列出一个槽的所有历史阶段。`_filter_withdrawn()` 仍按 claim 文本中的词过滤 current（`repository.py:647-675`）；结构化 negative belief 可能因为包含 “don't like” 等标记而被整行隐藏。后续需要以 transition intent、polarity 和时间轴决定可见性。

### 6. 评测工具原有契约断裂；本轮已修复接线

StateBench v1.1 报告只有 `tracks["state"/"recall"/"answer"]`，每条 track 是包含 `score`、`by_category` 的字典（`statebench_v1_1_runner.py:80-102, 354-380`）。`check_release_gate()` 却读取旧报告的顶层 `.score/.by_category`，并尝试对 `tracks["recall"]` 这个字典调用 `float()`（`bench/gate.py:94-119`）。用本次真实 v1.1 报告调用该函数，得到：

```text
TypeError: float() argument must be a string or a real number, not 'dict'
```

原有测试用旧形状的 fake report（`tests/test_polarity_lifecycle.py:518-590`），所以 769 个测试没有发现这一断裂。`compare_forced_production()` 也读取旧报告 `.score`：传入两份 v1.1 形状的报告，即使 state 分别为 0.95 和 0.20，它也返回 `gap=0.0, gate=True`。

本轮工作区已使 gate 读取 v1.1 三轨结构、区分 answer 未运行、核对 forced/production 的
数据集、case 顺序和模型，并在 CLI 中提供 `--gate` 与 `--production-report`。已补充针对
真实 v1.1 report 形状的测试。发布 gate 现在还要求完整的 200 个 case，避免小样本
`--case-ids` 评测因分数较高而被误判为正式发布通过。

### 8. OpenRouter 小样本实测：发现 K1 与 K2 状态合流缺口

使用 `nex-agi/nex-n2.5-pro:free`、相同代码/配置/数据集，对两个 development case
完成了 forced 与 production 配对。answerer 未运行。两种模式的逐回合 claim 数完全相同：

| case | transition | forced state/recall | production state/recall | 主要违规 |
|---|---|---|---|---|
| `sb-replacement-001` | `current_value_replacement` | fail/pass | fail/pass | 旧上海值的 K1 pattern、keyword 行仍为 active |
| `sb-round-trip-001` | `current_value_revival` | fail/pass | fail/pass | 旧柏林值的 K1 keyword 行仍为 active |

配对小样本的 state 为 **0/2 对 0/2**，recall 为 **2/2 对 2/2**。这里的 gap=0
只表示这两例没有测出节流差异；两种模式都失败，不能解释为发布门的
`production gap <= 0.05` 已达标。结构化 K2 旧值在 replacement case 中已经被
supersede，但并行写入的 K1 非结构化事实/主题仍被 state contract 视为 current。
因此第 1 阶段除 interval 与 polarity 外，还必须界定 K1/K2 同槽归并、降级或 current
读取过滤的规则，并用这些 case 回归。

第三个 case 的 production 请求先收到模型提供方 HTTP 520，随后超时；间隔后的重试
首个请求再次超时，故不把它计入有效配对。完整 v1.1 forced/production 成绩仍**未知**。
全量数据集有 200 例、每模式 486 个 turn；该免费模型的当前服务稳定性不足以完成本轮
全量成对评测，且 [OpenRouter 的免费额度](https://openrouter.ai/pricing)不支持一次完成数百次抽取请求。临时报告在 `/tmp/mirror-openrouter-pilot-{forced,production}.json`，
其中 production 的第三例包含失败后的 fallback，分析时必须排除。

### 9. K1/K2 同槽归并与当前值资格：已实施（2026-09-25）

针对第 8 节发现的两个失败 case，完成了“状态正确性优先”中的同槽切片。改动：

1. `PatternRule` 增加 `predicate` 与 `object_group`（`config/schema.py`），
   `extraction.yaml` 为 age、name、i_live、moved_to（新增搬迁句式）、i_want、
   like、dislike（否定式与负向动词各一条）声明谓词；`deterministic.py` 对声明
  谓词的 pattern 产出认知三元组，key 为 `predicate:object`。K1 与 K2 对同一槽
  位现在是同一类 claim，走同一条 identity 解析。
2. 无三元组的 claim（K1 keyword 命中等）不再落为 Belief 行，但该轮观测仍写入
   Evidence（`pipeline.py`）。这消除了“旧值的 snippet 行永远 active”的污染：
   它以前既拖住 replacement/revival 的 state contract，又意外满足部分断言。
3. `PATTERN_CONFIDENCE` 0.5 → 0.6：声明谓词的 pattern 是结构化断言，此前低于
   L4 渲染水印 0.55，导致所有 K1 行对召回不可见。
4. `resolve_identity` 增加极性替换：相反极性关于同一 object 的断言关闭当前值
   （“I do not like coffee” 关闭 “likes coffee”），实现同一 object 同一时间
   最多一个当前极性。

实测（同 commit、deterministic、200 例）：state **25/200 = 0.125 → 38/200 = 0.190**，
recall **2/200 = 0.010 → 46/200 = 0.230**；完整测试 **788 passed, 1 skipped**，
零回归；触及文件 Ruff 无新增。第 8 节的两个 pilot case（`sb-replacement-001`、
`sb-round-trip-001`）state 与 recall 均转为通过。分类变化：multi_value 4→10、
preference_evolution 12→18、replacement 0→2、round_trip 0→4；transition 级
coexisting_values 4→10、current_value_replacement 0→2、preference_revival 2→10/10。
返回句式（"I returned to Shanghai"）以上下文受限的 pattern 声明 lives_in，不经过
有域歧义的 synonym map。

同时有 8 例 state 回退，全部经逐例核对为**此前靠 keyword 行文本蒙混通过**：
`fact:I_am` 之类行的 claim_text 恰好含有合约要求的字符串（如
"applying for manager"），既无谓词也不会被 supersede。真实缺口是 K1 缺少
termination（"I sold X"）、correction、relationship（"friends with X"）、
goal resume（"applying for X again"）句式，以及 event occurrence 身份；这些
属于本计划后续的 END_CURRENT 与 K1 覆盖单元，不应靠恢复污染行来“提分”。

### 10. v1.1 forced/production 全量成对实测（2026-09-25，MiMo-V2.6-Flash）

在同一 commit（`5393053` + 同槽切片）、数据集、配置与模型下完成 200 例全量成对：

| 模式 | state | recall |
|---|---|---|
| forced | 148/200 = 0.740 | 168/200 = 0.840 |
| production | 149/200 = 0.745 | 168/200 = 0.840 |
| gap | -0.005（门禁 <= 0.05 通过） | 0.000 |

发布门判定：**FAIL**。state overall 0.740 < 0.90；分类中 contradiction 0.400、
preference_evolution 0.500、round_trip 0.767、stale_state 0.750 低于 0.85
（multi_value 1.0、replacement 0.867、temporal_event 0.900 通过）；recall 0.840 < 0.85
（差 2 例）；answer 未评；200 例覆盖完整；production gap 通过。15 例在两种模式间
翻转（7 例仅 forced 通过、8 例仅 production 通过），节流敏感但总体代价接近零。

**责任归属（逐例核对 52 个 state 失败）：49 例是状态模型失败，不是抽取失败**——
期望值已经出现在某个 belief 的 surface 上，但旧值行仍为 active（`not-current`
违规）。recall 失败 32 例中 29 例伴随 state 失败，主因同样是旧值泄漏
（`leaked_forbidden`）。剩余零分/低分 transition：explicit_correction 0/4、
relationship_termination 0/3、goal_cancellation 1/8、goal_reactivation 2/10、
preference_reversal 14/20、current_value_revival 23/30。仅 3 例真抽取缺口
（event key 哈希后缀不一致 2 例、漏抽 1 例）。

结论：forced 上限 0.740 的瓶颈是 **END_CURRENT/CORRECT 转移语义缺失**（替换、
往返复活、偏好反转、目标取消/重启、显式纠正、关系终止时旧当前值不关闭），
不是抽取质量，也不是检索通道。节流已不是问题（gap≈0）。因此 Hybrid Retrieval
（P1）应继续推迟：recall 的差距主要是状态失败的下游，加通道不会修复它。

### 11. 撤回语义切片：自纠与终止关闭旧当前值（2026-09-25，待 forced 复测）

针对第 10 节定位的 49/52“值已抽出但旧当前值未关闭”，实现撤回（retraction）分支：

- `memory/polarity.py` 新增 `is_self_correction`（"Correction:"、"I was wrong"、
  "That was mistaken" 等元语言标记）与 `is_termination`（"gave up"、"cancelled"、
  "sold the"、"no longer" 等结束当前状态的短语），以及 `object_tokens`（对象标识
  token，滤掉通用词）。这些标记是关于对话的元语言，不是承担事实模型职责的否定词。
- `resolve_identity` 增加撤回分支：候选带撤回标记且与某个 active belief 的 object
  token 重叠时，以该行为目标返回 UPDATE（lifecycle 分别记 `SELF_CORRECTION` /
  `END_CURRENT`），旧行被关闭、新行成为当前值。指代按 token 而非精确谓词/对象匹配，
  因为抽取器每轮拼写不同（`rust_coding`→`rust`、`run_a_marathon`→`marathon_goal`）
  且常换谓词（`wants_to`→`gave_up`）。
- `repository.py` 新增 `beliefs_referenced_by_tokens`；pipeline 对撤回类 claim 使用
  这个更宽的候选集——按谓词/精确对象收窄的查询看不到撤回要关闭的那一行。

验收：单元测试覆盖五种修正句式、四种终止句式与两个对照例（无标记不关闭、无指代
回落普通 CREATE）；集成测试覆盖 correction 与 goal cancellation 端到端。
完整测试 **794 passed, 1 skipped**；deterministic 38/46 无变化（预期：该切片是
K2 可见的）。forced 复测进行中。

### 12. 撤回切片复测与目标生命周期切片（2026-09-25）

第 11 节的撤回分支在 MiMo-V2.6-Flash 全量 forced 上复测：state **148 → 159/200**
（0.740 → 0.795），recall 168 → 167。goal_cancellation **1/8 → 8/8**，
relationship_termination 0/3 → 2/3，preference_reversal 14/20 → 16/20。
+20 改进 / -9 回退，净 +11；回退经逐例核对均为同一机制的不同拼写
（终止行的 interval 由抽取器记为过去事件，`negative-current` 断言要求
`temporal_mode=current`，见下）。

剩余 41 个 state 失败中 37 个是“值已抽出但不匹配状态合约”。按 transition：
goal_reactivation 8、current_value_revival 6、query_time_selection 5、
explicit_correction 4、preference_revival 4、preference_reversal 4。

据此实现目标生命周期切片：

- `is_resumption`（"again"、"restarted"、"signed up"、"booked" 等）与
  `refers_to_a_goal_generically`（"the goal"/"that plan"/"the project"）。
- 撤回分支增加通用目标指代回落：终止标记命中但与任何 active belief 无 token
  重叠时，若存在 active 目标族 belief（`is_goal_predicate`）且候选泛泛地指代
  目标，则关闭该目标。"I gave up on the goal" 的对象 `unnamed_goal` 与
  `run_marathon` 无共享 token，只能这样找到指代。
- 新增重启分支：候选带重启标记时，关闭被本引擎标记为 `transition=END_CURRENT`
  的 active 行——取消行自己是当前行，重启必须把它退场，否则目标状态永远停在
  “已取消”。
- pipeline 对 END_CURRENT 解决路径产生的新行打 `value.transition=END_CURRENT`
  标签；重启类 claim 用近期 active 候选集（标签行不被谓词/对象查询覆盖）。

验收：798 passed, 1 skipped；deterministic 38/46 无变化；触及文件 ruff 干净。
forced 复测（v3）进行中。

**已定位待办**：终止行的 `valid_to` 被抽取器记为过去事件日期（"sold last week" →
`valid_to=2026-09-20`），而 `negative-current` 合约要求当前状态行——需要区分
“终止事件发生在过去”与“终止产生的状态是当前的”，即 END_CURRENT 应产出当前
负向状态行，而不是一个已结束的事件行。

### 13. 测量完整性缺口：forced 在 K2 失败时静默降级（2026-09-25，已修复）

第二刀复测一度得到与 deterministic 完全一致的 38/200、46/200，且 `k2 calls` 行
显示 486 次调用 0 失败。逐层排查发现**两个静默吞错点**：

1. `SemanticExtractor.extract` 自己捕获 LLM 异常返回空 claims（异常不往上抛）；
2. `OpenAILLM.generate` 也捕获异常并返回调用方的 fallback 字符串
   （`'{“claims”: []}'`）——extractor 据此认为“模型正常回答但没有 claim”。
   端点实际返回 429（每日 $1 额度用尽），于是 486 次“成功”调用全部产出 0 条
   claim，forced 跑实际是 K1 数字。

修复（不需要 LLM 额度即可验证）：

- `OpenAILLM` 增加 `calls`/`call_failures` 计数器，在真正抛出异常的位置计数；
- `SemanticExtractor` 增加 `llm_calls`/`llm_failures`，并通过客户端计数增量识别
  “吞错后返回 fallback”这条路径；
- `ExtractionPipeline` 以属性暴露 `k2_attempts`/`k2_failures`；
- runner 每例记录 K2 健康，报告新增 `extraction_health` 字段与 `k2 calls` 渲染行；
- **守门**：forced/production 模式下若 K2 尝试数为 0，或失败数等于尝试数，
  `run_statebench_v1_1` 直接 `RuntimeError`，拒绝把 K1 数字当成模型结果。
- 单测覆盖抛异常与吞错返回 fallback 两种客户端形态（800 passed, 1 skipped）。

**当前测量状态**：最后一份有效 forced 测量仍是 v2（0.795，覆盖同槽+撤回两个
切片）。目标生命周期切片已实现并通过单测，但**尚未测量**——v3/v4 两次复测都落在
端点 429 窗口内，属于被吞掉的降级运行（v4 若在守门修复后运行会被直接拒绝）。
额度恢复后第一件事是重跑 forced 测量该切片。

### 14. Revival 失败根因：两类 bucket 分开归因（2026-09-25）

对 v2（0.795）的 revival 失败逐例归因，两类 bucket 的机制**不同**：

**`current_value_revival` 6 例**：4 例是 K1 `moved_to` pattern 在雇佣/就读语境
误报——"My employer is Northstar." → "I moved to Redwood Health."（实为换雇主）。
且实际顺序并非 K1 先写：`assemble_claims()` 已按 `(-has_triple, -confidence)`
把 K2 排在 K1 前，但 K1 的 `lives_in:redwood_health` 行随后仍被创建，并通过
`same_attribute_any`（同 object + 同 polarity 类）把**正确的 K2 works_at 行
supersede 掉**——比原判断更糟。剩余 2 例（sb-round-trip-020/024）是真正的
revival 语义问题，与 pattern 无关。

**`preference_revival` 4 例**：与 `moved_to` 无关（复核实测：4 例均无 K1 位置行）。
机制与 goal_reactivation 相同——**中间撤回行在重启后仍为 active**
（`gave_up_running`、`abandoned_gardening` 等），且撤回行的对象常常无法与重启
claim 匹配（sb-preference-evolution-009 的重启对象被抽成 `unspecified_activity`）。
另 2 例的撤回句式（"stopped enjoying"、"avoided hiking"）不在终止标记表内。

### 15. 同轮仲裁（c′）与守门加固（2026-09-25，已实施）

针对第 14 节第一类根因实施仲裁，替代原方案 (c)（调整持久化顺序——已被
`assemble_claims` 的排序证伪）：

1. claim 增加显式 `extractor_stage=k1|k2`（此前两者 `source` 同为 `extracted`）。
2. `assembler.arbitrate_claims()` 在持久化前按对象身份（对象 token 相交，
   "coffee" 与 "coffee_again" 同组）仲裁：
   - 同谓词族 → 合并，保留 K2 的结构与时间字段；
   - 谓词族冲突且 K2 命名的是已配置谓词 → K2 进入 belief，K1 降级为
     Evidence/trace；
   - K2 失败/未覆盖该对象 → K1 作为降级路径保留。
3. `same_attribute_any` 收窄：仅当候选谓词不在策略表中（未映射动词，如
   `returned_to`）且匹配行同属一个谓词时允许跨谓词 revival。
4. 目标生命周期两洞已修：pipeline 在 token 查询为空时补 `active_goal_beliefs`
   候选（"I gave up on the goal" 的对象 `unnamed_goal` 与目标无共享 token）；
   多个 active 目标时通用指代不关闭任何目标；END_CURRENT 行打
   `ended_object`/`ended_belief_id` 标，重启按该身份匹配（"I like coffee again"
   不会关掉卖车行）。
5. 守门加固：forced/production 要求 `k2_attempts == dataset_turns` 且
   `k2_failures == 0`（runner 层 RuntimeError）；`extraction_health` 新增
   `k2_empty_responses`/`k2_parse_failures`/`k2_schema_rejections` 分类计数
   （“模型没话说”与“模型没被听到”可区分）；release gate 检查上述全部健康项，
   部分失败也拒绝。`SemanticExtractor` 各早退路径统一返回 dict 形状。

验证：804 passed, 1 skipped；deterministic 38/46 无回归；**全部触及文件
Ruff 干净**（含清理 tests/test_integration.py 既有 12 个 E501）。Redwood 端到端
复现已转为仅保留正确雇主行；目标取消/重启/多目标安全/无关重启安全均端到端验证。
**尚未 forced 复测**（端点 429 窗口）。预期覆盖：current_value_revival 4/6、
preference_revival 中 2/4（有 ended_object 可匹配者）；"stopped enjoying" 句式与
unspecified 对象重启需下一轮标记或单活跃 ended 回落，已记录待办。

### 7. 数据库升级路径仍缺失（P0）

`MemoryEngine._ensure_db()` 只调用 `Base.metadata.create_all()`（`api.py:91-101`），仓库没有 schema migration runner。新增 polarity、raw predicate、时间和 evidence 字段后，全新库可建表，但已有库不会由 `create_all()` 自动补列。公开升级前需要迁移、版本检测和旧库回归测试。

## 证据边界

1. **788 passed** 说明当前测试定义全部满足；其中有一例跳过。它不能证明并发发布、旧库升级和所有来源的 targeted forget。
2. **v1.1 deterministic 0.190/0.230** 是 K1/无 LLM 基线；对 K2 修复没有解释力。同槽切片把该基线的 recall 从 0.010 提到 0.230、state 从 0.125 提到 0.190，两个 pilot case 已转通过；剩余失败集中在 K1 未覆盖句式、CREATE 时间列缺失与版本化 interval。要评价近期改动，应在同一 commit、数据集、模型、prompt 和回答器下分别运行 forced 与 production，并保留 manifest 和逐例漏斗。
3. **v1.0 0.730、targeted 0.867** 来自仓库文档中的不同轮次和范围。它们反映进展；本轮只有两个有效 v1.1 配对 case，不能宣称达到了 `state >= 0.90` 与 `recall >= 0.85` 的发布门槛。
4. Ruff 的 63 项主要分布在 import、未使用变量及旧测试长行；它是工程卫生债，不应替代上述运行时正确性优先级。

## 修订后的顺序与验收

| 顺序 | 工作 | 可验收结果 |
|---|---|---|
| 0a | 修复 v1.1 gate、成对比较与 taxonomy 映射 | 当前工作区已实现；真实 v1.1 报告不再抛异常，差距不再误报为零 |
| 0b | 在稳定且额度足够的模型端点上完成 forced + production 成对评测 | 本轮仅完成 2 个有效配对；全量须同一 commit、case、数据集、模型与参数，保存 manifest 和按 transition 的失败漏斗 |
| 1 | 完成 CREATE 时间列、polarity/END_CURRENT、interval 版本化、查询时间选择和 K1/K2 同槽处理 | 五个零分桶不再系统性为零；A → B → A 保留三个有效期；历史查询选中对应版本；K1 旧值不再伪装为 current。K1/K2 同槽与极性替换已落地（见第 9 节），CREATE 时间列、END_CURRENT、版本化 interval 与 `at_time` 仍未完成 || 1a | 与版本表一起确定容量界、压缩语义和最小 schema migration | 每槽最大精确版本数与归档规则可配置；超过界后存储有上限；旧库可升级；压缩后的历史查询精度明确 |
| 2 | 原子化 Publisher：CAS、幂等、失败回滚与证据范围检查 | 独立并发、故障注入、重复投递测试通过；StateBench 分数不作为此项验收 |
| 3 | 迁移 observe → API → verification 写入口 | 生产状态变化都有 proposal/receipt；旧直写路径被移除 |
| 4 | 用 forced 再测剩余失败，按漏斗决定下一批抽取或召回修复 | 发布门可按实际 v1.1 结果判断；没有达到门槛时给出剩余瓶颈归因 |

版本化与容量界必须同设计，但压缩旧 interval 会损失精确的远期 `at_time` 查询能力。
实施前要明确精确历史的服务范围；不能一边承诺所有历史时间点可查询，一边只保留最近
N 个 interval。schema migration 是新表上线的前提，应与第 1/1a 项一起落地，不能等到
Publisher 迁移后才做。summary/Observation/Assertion、完整 RecallQuery 和 Maintenance
Planner 保留在后续演进计划中；定向遗忘的回流问题仍是独立的治理验收项。
