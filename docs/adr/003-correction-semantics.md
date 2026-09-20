# ADR-003: Correction Semantics — In-Place Update + Event Provenance

## Status

Accepted (2026-09-20)

## Context

Users need to correct memories: "this is wrong, it should be X." This is different from:
- **UPDATE** (IdentityResolver-driven): system detects a new value for a SINGLE predicate
- **CONTRADICT**: system detects conflicting evidence

Correction is a **user-initiated** action with explicit intent.

## Decision

`correct_belief()` performs an **in-place update** of the existing belief's `claim_text`:
- Sets `source = "user_corrected"`
- Logs a `corrected` BeliefEvent with the correction note
- Does NOT create a new row (UNIQUE constraint on `(user_id, key)`)
- Confidence is preserved (or bumped to min 0.5)

The correction is tracked in BeliefEvent for full provenance.

### Why in-place, not supersede?

The `(user_id, key)` UNIQUE constraint prevents two rows with the same key. Creating a new row would require changing the key, which breaks identity semantics. In-place update preserves the key and logs provenance via events.

## Consequences

- `explain_belief()` can show the full correction history via events.
- Correction is distinguishable from normal SUPPORT/UPDATE by `source="user_corrected"`.
- The UNIQUE constraint remains the identity anchor.
