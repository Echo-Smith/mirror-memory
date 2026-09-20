# Architecture

## Overview

mirror-memory is a structured memory engine for AI agents. It extracts, stores, retrieves, and renders user memories across conversations using a **cognitive triple** model (subject → predicate → object) with identity resolution.

### Cognitive Triple Model

Every piece of user knowledge is represented as a cognitive triple: **subject → predicate → object** (e.g., `user → likes → painting`). This model provides a uniform structure for extraction, storage, and retrieval across all dimensions (topic, preference, fact, event, goal, pattern, boundary).

### Identity Resolution and Cardinality

The **IdentityResolver** determines how new observations relate to existing beliefs. Each predicate is assigned a **cardinality** type:

| Cardinality | Behavior | Example |
|-------------|----------|---------|
| SINGLE | Only one active value; new value supersedes old | `lives_in`, `age` |
| MULTI | Multiple values coexist | `likes`, `has`, `skills` |
| EVENT | Each occurrence is distinct (temporal fingerprint) | `went_to`, `attended` |

### State Transitions

When a new CandidateAtom arrives, the IdentityResolver produces one of these transitions:

| Transition | Effect |
|------------|--------|
| CREATE | New belief created (active) |
| SUPPORT | Existing belief confidence increased (weighted gain) |
| UPDATE | Old belief superseded; new belief becomes active |
| CONTRADICT | Existing belief confidence reduced; marked for clarification |

## Core Abstractions

### CandidateAtom (extraction output)

A structured observation awaiting identity resolution:

| Field | Description |
|-------|-------------|
| `subject` | Who the claim is about (default: "user") |
| `predicate` | Relationship verb (likes, lives_in, went_to) |
| `object` | What the predicate applies to (coffee, Shanghai, art_show) |
| `dimension` | Classification (topic, preference, fact, event, goal, pattern) |
| `claim_text` | The specific information stated |
| `confidence` | Extraction confidence [0, 1] |
| `temporal` | When it happened/occurs (for events) |
| `relation` | supports / contradicts |

### Belief (persisted state)

A belief is a persisted observation with lifecycle management:

| Field | Description |
|-------|-------------|
| `dimension` | Category |
| `key` | Content-specific identifier (e.g., `like_coffee`) |
| `predicate` | Relationship verb |
| `object` | What the predicate applies to |
| `cardinality` | single / multi / event |
| `confidence` | 7-factor weighted score |
| `layer` | L1 (user stated) → L2 (confirmed) → L4 (extracted) → rejected |
| `status` | active / superseded / corrected / rejected |
| `superseded_by` | ID of the belief that replaced this one |

## Identity Resolution

The **IdentityResolver** decides what happens when a new CandidateAtom arrives:

```
CandidateAtom
    ↓
Canonicalization (predicate synonyms, object normalization)
    ↓
IdentityResolver(existing_beliefs, policy)
    ↓
CREATE / SUPPORT / UPDATE / CONTRADICT / NOOP
```

### Cardinality

| Type | Behavior | Example |
|------|----------|---------|
| SINGLE | Only one active value; new value supersedes old | lives_in, age |
| MULTI | Multiple values coexist | likes, has, skills |
| EVENT | Each occurrence is distinct (temporal fingerprint) | went_to, attended |

### State Transitions

```
CREATE    → new belief (active)
SUPPORT   → existing belief: confidence += weighted gain
UPDATE    → old belief → superseded, new belief → active
CONTRADICT → existing belief: confidence *= 0.8, mark needs_clarification
NOOP      → resurrection guard / rejected / empty predicate
```

## Three-Layer Extraction

```
User message
    │
    ├─► K1 Deterministic (every turn)
    │   └─ Keyword + regex matching → instant claims
    │
    ├─► K2 LLM Semantic (throttled)
    │   └─ LLM extracts cognitive triples → validated + canonicalized
    │
    └─► K3 Synthesis (async worker)
        └─ Aggregates beliefs → structural understanding
```

## Dual-Channel Memory

```
observe() ─► K1+K2 ─► Belief Store ("understand the person")
          ─► Snippets ─► Summary Store ("remember the facts") [optional]

recall()  ─► Render beliefs (scored + budget-packed)
          ─► If beliefs don't match query → search session summaries
          ─► Merge output
```

## Scoring

Beliefs are scored using weighted signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| Topic hit | +5 | Key matches query topics |
| Predicate hit | +4 | Belief predicate matches query predicates |
| Object hit | +3 | Belief object matches query objects |
| Layer confirmed | +3 | L1/L2 user-confirmed |
| Activity | +2 | Confidence × time decay |
| Dimension boost | +4 | Dimension-specific bonus |
| Evidence | +1 | Multiple evidence items |
| Freshness | +1 | Recent evidence |

## Memory Lifecycle

### Correct (user-initiated)
```
User: "This is wrong, it should be X"
    ↓
correct_belief() → in-place update + correction event
```

### Forget (targeted)
```
forget_belief(belief_id)      → delete belief + events
forget_session(session_id)    → delete session summary
delete_user_memories(user_id) → cascade delete all
```

### Provenance
```
explain(belief_id) → full history: events, evidence, supersession chain
```

## Shadow Lifecycle

```
Single session → shadow snapshot (pending confirmation)
    ↓
Multi-session evidence → auto-promote to active
```

## Worker Pipeline

```
1. load_evidence    → Read active beliefs
2. formulate        → LLM candidate understanding
3. validate         → Deterministic checks (forbidden labels, third-party, contradiction)
4. compile_policy   → Controlled policy enums
5. synthesize       → K3 understanding
6. persist_snapshot → Versioned snapshot (shadow/active)
```

## Configuration

All domain-specific content is loaded from YAML files:

- `dimensions.yaml` — dimension definitions (7 dimensions)
- `anchors.yaml` — keyword anchors (41 anchors)
- `display.yaml` — user-facing labels
- `extraction.yaml` — keywords + regex patterns
- `identity_policy.yaml` — predicate → cardinality mapping (24 predicates)
- `prompts/` — K2/K3/verification/worker prompt templates
