# Runtime Closure — PR4 后续：State 失败的第一轮修复

Date: 2026-09-28
Status: **partially verified**（984 tests 全绿；单项案例有前后对比，
全量指标受提取器方差影响尚未复测）

## 背景

PR4 的成对测量（forced state 0.830 / production 0.840，gap +0.010）
把损失定位于 runtime 的 state transition。本轮针对失败集中的机理
（旧 current value 未关闭）做了五个修复，其中四个有个案验证。

## 修复

### 1. 答题器提示（测量有效性，非引擎）

answer 赛道 0.610 的成因是答题器风格：块里写着 "User does not like
coffee (active)"，答题器答 "No"——语义正确但不含契约要求的表层短语。
契约一直要求表层短语，而提示从未要求引用。`_build_answerer` 改为
"先逐字引用相关记忆行，再作答"。

**效果（全量）：answer 0.610 → 0.815（forced）/ 0.765（production）。**

### 2. profession 槽折叠（config/identity_policy.yaml）

提取器把 "I was a teacher" 标为 `profession`、把 "I am now a product
manager" 标为 `works_as`——同一槽两个谓词，resolver 永不相遇，旧值
保持 active。synonym_map 现把 `profession` / `retrained` / `became` /
`career_is` 全部折到 `works_as`。

**效果（个案）：sb-stale-state-015、sb-stale-state-019、sb-replacement-029
由失败转为通过**（旧值正确 superseded）。stale_state 类 17/20 → 19/20。

### 3. 返回措辞 → revival（extraction/pipeline.py + memory/identity.py）

"I went back to Falcon Works" 被提取器读成 `went_to`（策略内的 episodic
谓词，自占一个槽），绕开了 resolver 记载的 different-verb revival 路径。
pipeline 现在把返回措辞（went/going/go back to、returned/returning to、
moved/switched/transferred back to、back at、rejoined 等）改写为刻意
未映射的 `returned_to`，revival 门禁放行；候选集也改为包含 superseded
行的对象感知查询（原先 `resuming` 分支只取 active，revival 目标根本不在
候选集里）。

### 4. Publisher 的 revival 门禁（core/publisher.py）——PR2 回归

revival 的 UPDATE 目标正是 superseded 行，而 PR2 后 Publisher 的通用
不变量拒绝 "target_belief_superseded"。这是迁移引入的回归：直调
repository 时代没有这道门。现在 payload 带 `revival` 标记时放行。

### 5. revival 行的谓词/键继承（extraction/pipeline.py）

revival 产生的新行原先继承改写后的谓词（`returned_to`），导致它归属的
槽反而空着。现在继承目标行的 predicate 与 key。

**效果（个案）：sb-round-trip-015、sb-round-trip-019 由失败转为通过**
（三段区间：初值 superseded、中间值 superseded、返回值为 active）。

### 试过并回退：token 加宽的对象匹配

为 "Kingsley" vs "kingsley_college" 这类缩写，曾把 revival 的对象匹配
放宽到共享身份 token。它没有救下目标案例（020 仍失败），却引入了两个
新失败（017、024）——净收益为负，已回退。

## 未解决（按失败集中度）

- **contradiction 0.700**：explicit_correction / state_termination /
  relationship_termination / preference_reversal 的旧值关闭。
- **preference_evolution 0.467-0.533**：revival / reactivation /
  reversal。其中一部分是**契约表层匹配的脆弱性**：引擎行为正确
  （三行、最新值 active），但提取器把 "These days I enjoy jazz again"
  改写为 "User enjoys jazz these days"，契约要找的 "enjoy jazz again"
  不在表层上。
- **round_trip ~0.87**：020 的对象缩写（需要比 token 加宽更保守的
  方案）；其余失败在运行间漂移。
- **answer 0.815**：仍低于 0.85，剩余部分仍是答题器风格（如只引用
  部分相关事实）。

---

# 第二轮：contradiction 专项（同日追加）

## 诊断

contradiction 的失败机理是**撤回/终止没有关闭原值，而是并存一条反极性
信念**：

```
skills_rust          active   ← 正值
never_learned_rust   active   ← 撤回，本应关闭上面那条
```

三个断点：

1. **标记表只覆盖元语言纠正**（"I was wrong" / "correction"），漏掉
   实质性否定（"never learned" / "not certified" / "does not speak" /
   "only knows a few"）与处置终止（"gave the camera away"）。
2. **处置短语的宾语在动词和粒子中间**——连续子串 "gave away" 匹配不到
   "gave the film camera away"。
3. **撤回行的极性**：提取器给这些谓词（`gave_away` / `not_speak` /
   `never_learned`）不带极性，`infer_polarity` 的封闭集合里没有，
   结果撤回行是 neutral，契约的 `negative-current`（要求
   polarity=negative）匹配不上。

## 修复

- `_CORRECTION_MARKERS` 增加实质性否定（never learned / have never /
  not certified / do not speak / not fluent / only know a few / …）。
- `_TERMINATION_MARKERS` 增加处置动词（gave away / threw away /
  recycled / donated / lost my / …），并新增 `_is_disposal_phrasing`：
  动词与 away/out 成对出现即算处置（"gave the film camera away" 命中，
  "gave a presentation" 不命中）。
- 管线：撤回/终止类 claim 若推断极性非负，则置为 negative——撤回断言
  的就是被关闭值的否定。

## 效果

- contradiction state：**0.700 → 0.833**（标记扩展后实测；极性规则后
  一次运行为 0.800，见下方方差说明）。
- 984 tests 全绿。

## 全量结果（所有修复后，forced + production 成对）

```
                  forced                production
state   0.860 (172/200)           0.855 (171/200)      gap -0.005
recall  0.850 (170/200)           0.855 (171/200)      gap +0.005
answer  0.805 (161/200)           0.795 (159/200)      gap -0.010
```

**Gate 仍 FAIL**：state overall 0.860 < 0.90；contradiction 0.833 < 0.85；
preference_evolution 0.500 < 0.85；answer 0.805 < 0.85。

**但相比本轮开始前的首次测量（forced state 0.825 / answer 0.610）**：

| 指标 | 首测 | 现在 | 变化 |
|---|---|---|---|
| state (forced) | 0.825 | 0.860 | **+0.035** |
| answer (forced) | 0.610 | 0.805 | **+0.195** |
| contradiction | 0.700 | 0.833 | +0.133 |
| replacement | 0.933 | 0.967 | +0.034 |
| stale_state | 0.850 | 0.950 | +0.100 |
| round_trip | 0.867 | 0.900 | +0.033 |

forced/production gap 保持 -0.005（±0.05 以内）——提取节流仍不是瓶颈。

## 距 gate 还差什么

- state overall 需 +0.040（8 个案例）。
- **contradiction 0.833 → 需 1 个案例**过 0.85 线（025/026/027 一类的
  剩余失败是撤回行极性与表层匹配，已有方向）。
- **preference_evolution 0.500 是最大单点损失**（15/30）：其中一部分是
  契约表层脆弱性（提取器把 "enjoy jazz again" 改写为 "enjoys jazz these
  days"），需要区分“引擎错”与“契约错”。
- answer 0.805 → 0.85 需 9 个案例，剩余仍是答题器引用不全。

## 方差警告（本轮的关键教训）

contradiction 类在两次仅差“极性规则”的运行间从 0.833 变成 0.800
（失败集从 {004,021,024,026,027} 变成 {004,021,022,024,026,027}，
022 由通过变失败）。combined with the earlier full-run variance
（state 0.830 vs 0.825），结论强化：

**分类别 ±0.03 是噪声，逐例“修复”不可信，只有带信念转储的单例前后
对比可靠。** 这也是 token 加宽被回退的原因——它在一个运行里“救”了
目标案例，却在另一个运行里引入两个新失败。

因此本轮停止逐例优化，转为一次全量测量记录当前状态。

---

# 第三轮：preference_evolution 逐例分类（同日追加）

## 分类方法

对每个失败案例，把契约断言的匹配拆成两半：**语义**（status /
predicate / polarity / temporal_mode）与**表层**（surface_all /
surface_any 的短语）。语义满足而表层不满足 → 契约表层错；语义本身
不满足 → 引擎状态错。另外单列极性错（语义对但 bench 的
`_polarity()` 关键词启发式把负值行判成正）。

## 分类结果（首轮）

```
ENGINE   13 例   ← 撤回行永不被关闭
POLARITY  0 例
SURFACE   1 例   ← 引擎状态正确，提取器改写打败了契约短语
```

13 个引擎错是同一个机理：**revival / reactivation 不关闭中间的撤回行**，
三条 active 并存（正值、撤回、revival）。逐层定位到四个断点：

1. 候选集按 `predicate + 精确对象` 取行，"dislikes museums"（谓词
   dislikes、对象 museums）对 "enjoys museum trips"（谓词 likes、对象
   museum_trips）三条 clause 全不中——撤回行根本不在候选集里。
2. 对立极性闭合（"opposing row becomes the update target"）同样要求
   精确对象相等。
3. resumption 规则只认 `END_CURRENT` 标签的行，而撤回行多是普通
   CREATE 的负极性信念。
4. 撤回行与 revival 的对象常常零 token 重叠（"postponed buying" vs
   "viewing houses"），但与**原目标行**有重叠（vs "plans to buy a
   house"）。

## 修复

- 候选集与 `beliefs_referenced_by_tokens(object_tokens)` 求并集
  （LIKE 匹配 object/key/claim_text）。
- 对立极性闭合与 `polarity_conflicts` 改用 `_tokens_overlap`（精确
  token 或 ≥5 字符前缀，traveling/travel、museums/museum）。
- resumption 的 `ended` 识别扩展到：负极性 / 终止措辞 / paused·
  cancelled 生命周期，匹配改为**两跳**——ended 行与 revival 的 token
  重叠，或与任何跟 revival 重叠的行的 token 重叠。
- 终止标记补 "stopped enjoying/liking/doing…"、"rejected"、
  "postponed"、"removed" 等。

## 效果

| 指标 | 本轮开始 | 现在 |
|---|---|---|
| preference_evolution state | 0.500 | **0.733**（22/30） |
| 984 tests | 绿 | 绿 |

preference_reversal 子类（012/013/015/016/020）全部修复；
preference_revival 的 007/010 修复。

## 剩余 8 例的归因

- **1 例契约表层错**（004 jazz）：引擎状态正确（三行、最新值 active、
  两段历史关闭），但提取器把 "These days I enjoy jazz again" 改写为
  "User enjoys jazz these days"，契约的 `restored-current:jazz` 要求
  表层含 "enjoy jazz again"/"love jazz"。
- **7 例提取器对象匿名化/换词**，超出 token 匹配能力：
  `became_afraid_of_it`（代词 it）、`abandoned_gardening` vs revival
  的 `the activity`、`abandoned_manuscript` vs `writing_novel`、
  `decided_to_stay_put` vs `restarted_canada_move_process`、
  `rejects_management_roles` vs `manager_positions`（前缀本可中，
  待查）、`postponed_buying` vs `house_viewing`、`cancelled_project`
  vs `new_pilot_episode`。

这 7 例需要的是提取器把撤回句**链到已有目标**（K2 提示层面），属于
extraction 侧工作；按 PR4 的决策树（forced < 0.90 → 修 runtime），
runtime 侧已做的都做完了。

## 测量协议的重要发现：方差

同一 commit、同模型、temperature 0 下，两次全量 run 的 state 分别位
0.830 与 0.825；分类别摆动更大（preference_evolution 0.500 vs 0.467，
round_trip 0.900 vs 0.867，multi_value 0.967 vs 1.000）。单请求级别
是确定的（4 案例双跑完全一致），但全量级别不是——relay 疑似按负载
路由到不同后端。

**后果**：±0.01 的整体差与 ±0.03-0.05 的类别差是噪声级。单项案例的
前后对比（有信念转储为证）是可靠的；全量指标的提升需要多次运行取中位，
或换用确定性的 provider。PR4 的 gate 判读应记录运行次数与方差。

## 复跑方式

`MM_KEY=<key>` 后运行 PR4 脚本（extractor 与 answerer 同模型、同
extra_body、temperature 0）。报告落盘 `/tmp/statebench_v1_1_pr4/`。

---

# 第四轮：累积全量测量（relay 恢复后）

## 结果

```
                  forced                production
state   0.895 (179/200)           0.910 (182/200)      gap +0.015
recall  0.900 (180/200)           0.890 (178/200)      gap -0.010
answer  0.780 (156/200)           0.820 (164/200)      gap +0.040
```

**Gate 仍 FAIL，但只差一点**：gate 评的是 forced——state overall 0.895
（差 1 个案例到 0.90）、contradiction 0.767、preference_evolution 0.767、
answer 0.780。配对的 production 跑过 0.910，gap +0.015 在 ±0.05 内。

## 相对本轮起点（forced state 0.825 / answer 0.610）

| 指标 | 首测 | 现在 | 变化 |
|---|---|---|---|
| state (forced) | 0.825 | 0.895 | +0.070 |
| recall (forced) | 0.850 | 0.900 | +0.050 |
| answer (forced) | 0.610 | 0.780 | +0.170 |
| preference_evolution | 0.467 | 0.767 | +0.300 |
| replacement | 0.933 | 0.967 | +0.034 |
| stale_state | 0.850 | 0.950 | +0.100 |

## 最后一处引擎修复

累积测量暴露出 contradiction 的 026（"I am not certified to dive" vs
`i_am_certified_diver`）：retraction 路径的 `sharing` 用**精确 token 交集**，
`diving_certification` 与 `certified_diver` 前缀本可匹配。统一到
`_tokens_overlap` 后 984 tests 全绿。

## contradiction 剩余 7 例的归因（分类纪律）

- **bench 极性启发式漏判**（3 例）：`_polarity()` 的负标记表里没有
  "gave the camera away"（宾语在中间）、"only knows a few"；027 是契约
  短语写 "do not speak" 而提取器写 "does not speak" 的子串不匹配。
- **引擎缺口**（1 例，已修）：026 的 retraction token 匹配。
- **其余**：012 的 goal cancellation 后 re-plan 未关闭 cancelled 行
  （revival 句缺 "again" 标记，resumption 路径不触发）。

## 结论

runtime 侧按失败集中度可做的都已完成；剩余缺口一半在 measurement
（bench 的极性启发式与契约短语），一半在 extraction（撤回句链到目标、
对象匿名化）。若要继续推 gate，下一步应是修 bench 的 `_polarity()`
（补 "gave … away"/"only … a few" 类标记）与契约短语，而不是继续改引擎。

---

# 第五轮：LongRunBench（多年运行能力）

## 目标

MetabolismBench 的 29 个案例证明规则；LongRunBench 证明**运行时**：
1 用户 / 1 年模拟时间 / 5000+ 观察 / 300+ 信念 / 3000+ evidence /
100 纠正 / 50 冲突 / 20 遗忘，代谢周期与召回穿插其中。

七项目标（全部达成）：

| 指标 | 目标 | 实测 |
|---|---|---|
| semantic preservation | ≥ 0.99 | **1.000** |
| correction loss | 0 | **0** |
| conflict loss | 0 | **0** |
| forgotten resurrection | 0 | **0** |
| wrong archive | 0 | **0** |
| hot evidence reduction | ≥ 0.70 | **0.820** |
| recall p95 | < 100 ms | **4.1 ms** |

运行规模：5,045 观察、364 信念、4,989 evidence、53 周期、122 召回、
100 纠正、50 冲突、20 遗忘；摄入 167,960 字符 → 热库 30,193 字符
（923 / 4,989 行存活）；墙钟 41s。

## 实现

``bench/longrun.py``：确定性场景生成器（K1 提取，无 LLM）+ runner
（模拟时钟盖章在每个观察上，让保留窗口看到一年老化）+ 验证器
（按 runner 记录的 id 精确检查）+ 报告。``tests/test_longrun_bench.py``
用小场景覆盖 harness（11 个用例，CI 可跑）。

## Bench 驱动的引擎修复

**revival 行的 cardinality 取错了谓词**：UPDATE payload 的
``cardinality`` 取自改写后的 ``returned_to``（→ multi），而不是 revival
继承的目标谓词（``lives_in`` → single）。结果返回住处的行被当成多值
偏好，180 天就开始降温——LongRunBench 的 semantic preservation 因此
只有 0.667。修为按 ``revival_predicate`` 取基数后 1.000。

## Bench 自身的五个修正（测量有效性）

1. **观察未产生独立 evidence 行**：每个 observe 都用同一个
   ``turn_count``，管线按 ``session:turn`` 去重，一天的全部观察塌缩成
   一行——压缩阈值永远达不到。改为递增序号。
2. **conflict 在压缩之后注入不算丢失**：验证器原先查 digest 存在性，
   把“先压缩后争议”误判为丢失。改为检查**争议证据本身**是否被折叠。
3. **forget 后会重新观察**：脚本在年末前删除，但观察持续到年末，
   重新陈述是新信念而非复活。改为最后一天删除。
4. **semantic 验证器漏了 name/age 行**：这些行的 key 是 ``name:lena``
   形式，不在 ``("my_name", "age")`` 白名单里。改为按谓词匹配。
5. **wrong_archive 循环嵌套错**：嵌在遗忘循环里，每遗忘一条就全表
   扫一遍，同一 key 追加 20 次。

另有两个 bench 设计修正：纠正/冲突只作用于**仍在扫描范围内**的信念
（对已归档信念“纠正”实为 restore），遗忘排除 canonical 事实
（名字/年龄/城市不该被这个场景遗忘）。

---

# 第六轮：发布验收收尾（ledger 删除语义 / 迁移边界 / 基准强度）

针对外部审计的五项阻断与两项高估，逐条处置。

## 1. P0：forget 没有删除 versioned ledger（已修）

审计复现（创建 → targeted forget → full delete）原先残留
`BeliefVersion.object == "shanghai"` 与孤儿 `BeliefIdentity` /
`ProposalLog`。现已：

- `forget_belief` 通过 `_purge_belief_ledger` 删除该 slot 的**整条版本链**
  （ BeliefVersion.object 是明文，只删当前行等于没删）+ identity；
- `ProposalLog` 按 `target_belief_id` **或** `created_belief_id` 匹配清除
  （CREATE 行的 target 为空，故新增 created_belief_id 列记录它创建的行）；
- `delete_user_memories` 同样级联 versions / identities / proposals，
  计数回填 `DeleteResult`；
- tombstone 照旧幸存（它是“删除已发生”的凭证，不含内容）。

审计复现现已全零；5 个隐私回归测试钉住（tests/test_deletion_privacy.py）。

## 2. 存量 backfill 与版本容量（已实现）

- `backfill_belief_ledger(user_id)`：为没有版本的存量 Belief 追加开版本
  （proposal_id="backfill"），幂等；否则旧信念首次更新时历史链从半路开始。
- `LedgerConfig`（config/metabolism.yaml 的 `ledger` 段）：
  `max_versions_per_identity`（默认 50）与 `version_ttl_days`（默认 730），
  Publisher 按 config 执行修剪；TTL 以 `epistemic_to`（被取代的时间）
  为准，最新版本永不修剪。
- 修：修剪时 naive/aware 混比导致 UPDATE 全线 TypeError（SQLite 返回
  naive 而 utcnow() 是 aware）。

因此 PR3 现可称“dual-write + interval 语义 + 迁移边界”完成。

## 3. LongRunBench 三个指标被高估（已修）

- **semantic_preservation**：原先只检查三个谓词各有一条 active 行。现改为
  逐值比对（Scenario 记录 expected_name/age/city，验证器比较 object），
  错城仍会得 1.000 的可能已消除。
- **conflict_loss**：`conflicted_ids` 记录了 50 个 id 但最终未使用，扫描
  只覆盖仍带标记的行。现改为**逐 id 检查**：行缺失、flag 丢失、被归档都
  计损失；被后续用户陈述取代或遗忘的合法豁免。
- **correction_loss**：补充 lineage——只有“仍是 active 却被归档”算静默
  丢失；被更新陈述取代或被遗忘不算。
- 新增 7 个反向测试（TestOracleDetectsDamage）：篡改每个指标后必须被
  检出，证明 oracle 不是恒零。

## 4. Gate 的准确状态与 provenance（已修正）

- StateBench 数据集升版 **1.1.0 → 1.2.0**（契约短语 "do not speak Italian"
  改为 "not speak Italian"，兼容提取器的 "does not"）；runner 常量同步。
- gate 脚本现写 `RunManifest`（dataset sha256、commit、dirty、模型、
  config/prompt 哈希）并传给 `check_release_gate`，**manifest not evaluated
  已消除**。
- README 的绝对延迟标注为“参考测量”（本机 recall p95 24ms / 周期 p95 2s，
  审计机 30.59ms / 135s），并给出复跑命令。

## 5. 工具收尾（已完成）

- `python -m mirror_memory.bench.longrun_cli` CLI 入口（--json/--seed，
  七目标未达即非零退出）。
- Ruff：本次新增/触及文件全部清零（含历史遗留的 8 处）。
- 测试：**1014 passed, 1 skipped**。

## 最终 Gate（数据集 v1.2.0，manifest 已写）

```
                  forced                production
state   0.925 (185/200)           0.910 (182/200)      gap -0.015
recall  0.900 (180/200)           0.895 (179/200)      gap -0.005
answer  0.830 (166/200)           0.815 (163/200)      gap -0.015
```

**Gate FAIL，仅剩两项**：preference_evolution 0.767 < 0.85 与 answer
0.830 < 0.85。相比本轮开始前的 4 项失败（state overall、contradiction、
preference_evolution、answer、manifest），forced state 从 0.825 升至
**0.925**，六个类别过线。剩余两项的归因不变：preference_evolution 缺在
extraction 侧（撤回句的对象匿名化/换词，runtime 规则已就位），answer 缺在
答题器引用习惯。这两项都不再是 runtime 缺口。
