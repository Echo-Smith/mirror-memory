# ADR-004: Host vs Mirror Responsibility Boundary

## Status

Accepted (2026-09-20)

## Context

Mirror Memory is a library, not a platform. Where does Mirror's responsibility end and the host application's begin?

## Decision

### Mirror owns (memory semantics):
- **Identity**: Triple model, cardinality, IdentityResolver
- **State**: Belief lifecycle (CREATE/SUPPORT/UPDATE/CONTRADICT/SUPERSEDE)
- **Confidence**: 7-factor weighted scoring
- **Extraction**: K1 deterministic + K2 LLM semantic + K3 synthesis
- **Retrieval**: Query-aware scoring, dual-channel fallback
- **Forget**: Cascade + targeted deletion
- **Correction**: User-initiated in-place correction
- **Provenance**: BeliefEvent audit trail + explain()

### Host owns (platform governance):
- **Authentication**: Who can call Mirror APIs
- **Authorization**: Per-user/per-feature access control
- **Tenant isolation**: Multi-tenant data separation
- **Purpose binding**: What the memory is used for
- **Rate limiting**: API call throttling
- **Generation barriers**: Preventing stale writes after deletion
- **Backup/replay safety**: Ensuring deleted data stays deleted
- **Compliance receipts**: Audit trails for regulatory requirements
- **Deployment**: Database choice, scaling, monitoring

### Interface contract:
- Mirror accepts `user_id: str` as the identity namespace. The host decides how to construct it (e.g., `tenant:app:user`).
- Mirror exposes `before_observe` / `before_recall` hooks (future) for host-side auth checks.
- Mirror's error types (`MirrorMemoryError` hierarchy) allow the host to handle failures appropriately.

## Consequences

- Mirror stays lightweight — no auth/tenant/compliance code.
- Host applications can layer governance without modifying Mirror.
- The `user_id` is the single boundary contract between Mirror and host.
