# Decision Record 002: Three-Phase Worker Architecture

Date: 2026-09-17
Status: Accepted

## Context

The original Worker used a single Session for the entire lifecycle (claim → process → commit). This created concurrency hazards:
- Stale workers could overwrite results after lease expiry
- Revocation during processing couldn't prevent atom creation
- No atomic barrier between authorization check and write

## Decision

Implement Claim → Compute → Publish as three separate transaction boundaries:

1. **Claim (T1)**: Short transaction. CAS lease acquisition, authorization check, load evidence snapshot. Commit returns immutable `ClaimTicket`.

2. **Compute**: No database session. Pure extraction (rule-based or model). Safe to fail without side effects.

3. **Publish (T2)**: New transaction. `SELECT FOR UPDATE` on ScopeControl, Evidence, and Job. Re-verify authorization, deletion generation, evidence existence, and lease validity. Write atoms, CAS ViewHead, fencing complete. All-or-nothing rollback.

## Consequences

- ClaimTicket is constructed before T1 commit (snapshot from verified state)
- `expire_on_commit=False` on all session factories prevents post-commit re-queries
- Purpose is frozen on the Job at Observe time, not passed as optional Worker parameter
- Control operations (delete/revoke) serialize with Publish via row locks

## Verification

15 concurrent/phase-interleaving tests on PostgreSQL, including:
- Dual-thread concurrent Publish (Barrier + CAS)
- Revoke/delete between Claim and Publish
- Lease expiry without reclaim
- Evidence-level deletion blocking
- no_memory + revocation not marking ready