# ADR-005: Async Worker Design — Claim/Compute/Publish Pattern

## Status

Accepted (2026-09-20), partial implementation

## Context

The evolution worker processes beliefs asynchronously (K3 synthesis, snapshot generation). How should it handle concurrency, failures, and stale writes?

## Decision

### Current implementation (v1):
- **Claim**: `enqueue_job()` creates a pending job with idempotency key
- **Compute**: `_process_single_job()` runs 6-node pipeline (load → formulate → validate → compile → synthesize → persist)
- **Publish**: `persist_snapshot()` writes versioned snapshot

### Invariants:
- One pending/running job per user at a time
- Idempotency key = `user_id:watermark:prompt_version`
- Crash recovery: stale running jobs (>5 min lease) reset to pending
- Atomic claim: `UPDATE WHERE status='pending'` prevents double-processing

### What's NOT implemented (future):
- **Generation barrier**: publish-time check that memory state hasn't changed since claim
- **Auth version recheck**: verify authorization hasn't been revoked during compute
- **Deletion generation**: verify user hasn't been deleted during compute
- **CAS on View**: compare-and-swap for concurrent snapshot updates

### Design principle:
> The async worker should be designed so that future generation barriers can be inserted between Compute and Publish without restructuring the pipeline.

## Consequences

- The current worker is functional but lacks stale-write protection.
- Generation barriers are a host-level concern that can be added later.
- The 6-node pipeline is structured to allow inserting pre-publish checks.
