# 预算护栏设计文档

## 概述

预算护栏（R11）控制 Mirror Memory 的资源消耗，防止一个用户耗尽共享资源。

## 架构

```
BudgetProfile (产品级配置)
├── 每用户每日 token 上限
├── 每用户每日费用上限
├── 每用户并发任务上限
├── 软阈值比例 (warn)
├── 硬阈值比例 (reject)
├── 冷却时间
└── 恢复阈值比例

BudgetReservation (事务级预留)
├── 操作前预留
├── 操作后结算
└── 过期自动释放

UsageLedger (不可变记账)
└── 所有消耗的永久记录
```

## 生命周期

```
1. observe 请求到达
2. check_and_reserve():
   ├── 检查是否控制操作 (forget/correct/sync_auth → 豁免)
   ├── 获取 BudgetProfile
   ├── 检查并发限制
   ├── 计算当日已用量 + 预留量
   ├── 硬阈值检查 → 拒绝或继续
   ├── 软阈值检查 → 警告标记
   └── 创建 BudgetReservation
3. 处理 (Compute + Publish)
4. settle(): 用实际消耗更新预留，写入 UsageLedger
```

## 产品适配

不同产品的差异全部在 BudgetProfile 中：

| 参数 | 产品 A | 产品 B |
|---|---|---|
| scope_daily_token_limit | 10000 | 20000 |
| scope_daily_cost_limit | 1.00 | 2.00 |
| scope_concurrent_jobs | 2 | 3 |
| soft_threshold_ratio | 0.8 | 0.85 |
| hard_threshold_ratio | 1.0 | 1.0 |
| cooldown_seconds | 300 | 60 |
| recovery_threshold_ratio | 0.6 | 0.5 |

## 控制操作豁免

以下操作不受预算限制：
- `forget` (删除)
- `correct` (纠正)
- `sync_authorization` (授权)

这确保用户始终能行使隐私控制权。

## 待校准项 (需要真实数据)

以下参数需要基于真实用户数据校准，当前使用保守默认值：

- **误限率**: 需要测量正常用户被误拒的比例
- **合理默认值**: 每产品的 token/cost 上限需要基于业务基线
- **成本/质量帕累托曲线**: 需要接入实际提取模型（如 OpenAI、DeepSeek、本地模型等）后，通过不同配置下的质量与成本对比数据得出

## API 端点

预算状态可通过以下方式查询：

```python
svc = BudgetService(session)
summary = svc.get_usage_summary(scope)
# {
#   "scope": "tenant:app:subject",
#   "period_cost": 0.45,
#   "period_tokens": 3200,
#   "active_calls": 1,
#   "profile": {
#     "app_id": "app_alpha",
#     "daily_cost_limit": 1.0,
#     "daily_token_limit": 10000,
#     "concurrent_jobs_limit": 2
#   }
# }
```

## HTTP API

预算通过 `/v1/observe` 端点自动检查。超限时返回：

```json
{
  "success": false,
  "reason_code": "CAPACITY_LIMIT",
  "reason": "daily_cost_limit_reached (1.05/1.00)"
}
```