# ADR-001: Memory Identity — Triple + Cardinality + Resolver

## Status

Accepted (2026-09-20)

## Context

How should Mirror Memory determine whether two pieces of information about a user are "the same thing" or "different things"?

Traditional approaches use a key-based identity (`user_id + key`), but this causes collisions when multiple facts share the same key type (e.g., "likes coffee" and "likes painting" both map to key="like").

## Decision

We adopt a **cognitive triple** model: `(subject → predicate → object)` with **cardinality semantics**:

- **SINGLE**: only one active value at a time (e.g., `lives_in`, `age`, `works_at`)
- **MULTI**: multiple values coexist (e.g., `likes`, `has`, `skills`)
- **EVENT**: each occurrence is distinct (e.g., `went_to`, `attended`)

An **IdentityResolver** (pure function) decides the action for each new CandidateAtom:
- `CREATE` — no existing match
- `SUPPORT` — same predicate+object, strengthen evidence
- `UPDATE` — SINGLE cardinality, new object supersedes old
- `CONTRADICT` — explicit conflict on existing belief
- `NOOP` — rejected / resurrection guard

The resolver is **configuration-driven**: `identity_policy.yaml` maps predicates to cardinality types. Domain adapters can override.

## Consequences

- Each fact gets a unique identity based on content, not just category name.
- "likes coffee" and "likes painting" coexist as separate beliefs.
- "lives in Shanghai" → "lives in Beijing" correctly supersedes the old value.
- The `(user_id, key)` UNIQUE constraint remains but keys are now content-specific (e.g., `like_coffee` not `like`).
- The resolver is a pure function — easily testable, no side effects.
