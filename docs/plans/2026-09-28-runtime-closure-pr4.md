# Runtime Closure v1 — PR4: StateBench v1.1 Gate Closure

Date: 2026-09-28
Status: **measured, gate NOT met**（forced state 0.830 / production 0.840；
按既定决策树：forced < 0.90 → 继续修 runtime，不碰 extraction）

## 运行条件（全部固定）

| 项 | 值 |
|---|---|
| commit | `7b81e3a` |
| extractor | DeepSeek-V4-Flash（temperature 0，thinking disabled，max_tokens 2000） |
| answerer | DeepSeek-V4-Flash（同一模型、同一 provider 设置） |
| dataset | statebench_v1_1（packaged，200 用例） |
| config | `config/`（仓库当前配置） |
| 模式 | forced + production，背靠背同条件运行 |

## 结果

```
                  forced                production
state   0.830 (166/200)           0.840 (168/200)      gap +0.010
recall  0.850 (170/200)           0.845 (169/200)      gap -0.005
answer  0.610 (122/200)           0.610 (122/200)      gap  0.000
```

**Gate 判定：FAIL**

- state overall 0.830 < 0.90
- category contradiction 0.700 < 0.85
- category preference_evolution 0.500 < 0.85
- answer 0.610 < 0.85
- manifest not evaluated（见“遗留项”）

**通过的检查**：production-forced gap = +0.010（state）/ -0.005（recall），
在 ±0.05 以内——说明 forced 与 production 的提取差异不是当前的瓶颈，
问题在 runtime 本身。

## 与历史数字对比

| 指标 | 此前记录（v1.1 同模型） | 本次 |
|---|---|---|
| state (forced) | 0.740 | **0.830** |
| state (production) | 0.745 | **0.840** |
| recall | 0.840 | 0.850 |

提升来自：谓词声明式 pattern（提取层）+ PR1-PR3 的发布协议与 ledger。

## 失败解剖（forced，200 用例）

三条赛道各自失败：**state 34、recall 30、answer 78**。answer 的大头
由 state/recall 失败传导（块错了答案就错），加答题模型自身误读。

### state 失败按 (类别, transition)

```
preference_evolution/preference_revival      5
preference_evolution/preference_reversal     5
preference_evolution/goal_reactivation       5
contradiction/explicit_correction            4
round_trip/current_value_revival             3
stale_state/query_time_selection             3
replacement/current_value_replacement        2
contradiction/state_termination              2
contradiction/relationship_termination       2
multi_value/coexisting_values                1
contradiction/preference_reversal            1
temporal_event/distinct_occurrences          1
```

### 机理：旧 current value 没有被正确关闭

抽样断言细节：

```
sb-replacement-024: violated=['not-current:26']
  match_counts={current:27: 1, history:26: 1, not-current:26: 1}
  → 26 已在 history，但仍出现在 current 面（读侧单值槽规则漏掉它）

sb-replacement-029: missing=['history:photographer'] violated=['not-current:photographer']
  match_counts={current:operations manager: 1, history:photographer: 0, not-current:photographer: 1}
  → 旧值既没进 history，也没被标为 not-current（写侧没关区间）

sb-round-trip-015: missing=['history:Riverstone'] violated=['not-current:Riverstone']
  → revival 路径的中间值未关闭
```

这正是仓库 Runtime Closure 文档此前的判断（52 个 state failure 里
49 个是“新值抽出来了，但旧 current value 没正确关闭”），本轮数据
完全复现且集中在：revival / reactivation / reversal / correction /
termination 六类 transition。

## 决策（按既定决策树）

> 如果 forced >= .90 而 production 差很多 → 回 extraction。
> 如果 forced 自己还 < .90 → 继续修 runtime。

forced state 0.830 < 0.90 → **下一轮修 runtime**，目标按失败集中度排序：

1. **preference_evolution（0.500）**——revival / reactivation /
   reversal：偏好与目标的生命周期 transition 是最大单点损失
   （15/30）。注意 PR3 之后 A→B→A 走“新 version”路径，读模型出现
   `enjoys_running~2` 这类消歧键，需确认读侧单值槽规则按谓词
   （而非键前缀）聚合。
2. **contradiction（0.700）**——explicit_correction / termination /
   preference_reversal：自我纠正与终止类的旧值关闭。
3. **round_trip revival（0.900）**与 **stale_state query_time_selection
   （0.850）**：区间关闭与查询时间选择。
4. **answer（0.610）**：state/recall 修好后重测；若仍低，再查渲染
   选择与答题提示。

## 遗留项

- **manifest 未接线**：gate 的 manifest 检查（数据集/配置指纹）存在，
  但 runner 不产出 manifest，所以该项显示 not evaluated。发布协议
  应补上：runner 写 manifest（dataset hash + config hash + model +
   commit），gate 校验之。
- 完整报告（forced/production 各 ~630KB JSON）未入库；摘要即本文。
  复跑脚本：设置 `MM_KEY` 后运行 PR4 的运行脚本（extractor 与
  answerer 同模型、同 extra_body、temperature 0）。
