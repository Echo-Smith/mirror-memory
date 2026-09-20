# ADR-002: Forget Semantics — Cascade + Targeted

## Status

Accepted (2026-09-20)

## Context

Users must be able to delete their memory data. Two scales:
1. **Full forget**: delete everything for a user (GDPR-style)
2. **Targeted forget**: delete a specific belief or session summary

The system must ensure deleted data does not reappear through stale workers or cached state.

## Decision

### Full forget (`delete_user_memories`)
Deletes ALL tables: beliefs, events, stats, snapshots, jobs, consent grants, session summaries. Returns per-table counts. No generation barrier in v1 (host-level concern).

### Targeted forget
- `forget_belief(user_id, belief_id)` — deletes belief + its event history
- `forget_session(user_id, session_id)` — deletes a session summary

### What Mirror owns vs what Host owns
Mirror handles the **data deletion**. The host is responsible for:
- Preventing stale workers from re-creating deleted data (generation barrier)
- Backup/replay deletion safety
- Compliance receipts and audit trails

## Consequences

- `delete_user_memories` is complete — includes SessionSummary (the "second channel").
- Targeted forget allows surgical removal without wiping everything.
- Generation barriers and compliance receipts are host responsibilities, not Mirror's.
