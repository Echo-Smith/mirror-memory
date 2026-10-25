# Decision Record 003: Budget Guardrail Design

Date: 2026-09-17
Status: Accepted

## Context

R11 requires budget controls: "按 profile 限制、延后和恢复；并发预留、重试和在途调用计入预算，撤销/删除不被费用闸门阻断."

No real user data is available yet. Budget parameters (limits, thresholds) are product-specific.

## Decision

Separate **mechanism** from **parameters**:

1. **BudgetProfile**: Per-app configuration (daily token/cost limits, concurrent job limits, soft/hard/recovery thresholds, cooldown). Each product defines its own profile.

2. **Reserve/Settle/Release lifecycle**: Pre-operation reservation with threshold checks, post-operation settlement with actual usage, automatic release on expiry.

3. **Control operation exemption**: `forget`, `correct`, `sync_authorization` bypass all budget checks. Users can always exercise privacy controls.

4. **Three-tier threshold**: soft (warn), hard (reject), recovery (resume after cooldown).

## What's Not Decided (deferred to real data)

- Per-product default values (need business baseline)
- Mis-limit rate measurement (need real traffic)
- Cost/quality Pareto curve (need actual model calls)

## Consequences

- No budget profile = no limits (allows development without configuration)
- Budget check is integrated into `observe()` — rejected before evidence is created
- `UsageLedger` is immutable — all consumption is permanently recorded
- Profile changes are versioned — rollback is possible