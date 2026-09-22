# Architecture

## Overview

mirror-memory is a structured memory engine for AI agents. It extracts, stores, retrieves, and renders user memories across conversations using a **cognitive triple** model (subject → predicate → object) with identity resolution.

### Cognitive Triple Model

Every piece of user knowledge is represented as a cognitive triple: **subject → predicate → object** (e.g., `user → likes → painting`). This model provides a uniform structure for extraction, storage, and retrieval across all dimensions (topic, preference, fact, event, goal, pattern, boundary).

### Identity, Relation, and Lifecycle

Three concerns, deliberately separate:

| Module | Question it answers |
|--------|---------------------|
| `memory/identity.py` | Which existing belief should this be compared against? |
| `memory/relation.py` | What *is* the relationship between the two facts? |
| `memory/lifecycle.py` | Given that relationship, what should happen? |

The old `IdentityResolver` answered all three in one function, which is what
produced the `SINGLE = UPDATE` rule. Each predicate is assigned a **cardinality**
and a **temporal scope**:

| Cardinality | Behavior | Example |
|-------------|----------|---------|
| SINGLE | Only one value is true at a time | `lives_in`, `age` |
| MULTI | Multiple values coexist | `likes`, `has`, `skills` |
| EVENT | Each occurrence is distinct (temporal fingerprint) | `went_to`, `attended` |

| Scope | Behavior | Example |
|-------|----------|---------|
| `current_state` | A later observation closes the earlier interval | `lives_in`, `works_at` |
| `persistent` | True until something explicitly ends it | `has`, `is_married` |
| `episodic` | Each occurrence is its own fact | `went_to`, `attended` |

### State Transitions

The lifecycle decision comes from identity **plus** the temporal relation
between the two validity intervals, not from cardinality alone:

| Lifecycle | Effect |
|-----------|--------|
| CREATE | New belief created (active) |
| SUPPORT | Existing belief confidence increased (weighted gain) |
| TEMPORAL_UPDATE | Old interval **closed** at the new fact's start; new belief becomes current |
| COEXIST | Overlapping or different period: both stay active |
| CONTRADICT | Existing belief confidence reduced; marked for clarification |

`TEMPORAL_UPDATE` is the important difference from the old `UPDATE`. The old
rule replaced the row, so "where did they live before?" became unanswerable.
Now `lives_in Shanghai` gets `valid_to = 2026-05` and `lives_in Berlin` gets
`valid_from = 2026-05`, so both questions are answerable.

### Evidence

Evidence is a first-class entity, not a blob of ids on the belief row:

```
Evidence
├── id, user_id, session_id
├── ref          (caller's own id for the source, unique per user)
├── content
├── source_type  (message / session / document / system)
├── extraction_method (k1_keyword / k1_regex / k2_llm / user_confirmed)
├── authority    (user / assistant / system / derived)
└── observed_at

BeliefEvidenceLink
├── belief_id
├── evidence_id
└── relation  →  support | contradict | verify | correct
```

The relation is what makes this more than a join table: the same message can
support one belief and contradict another, which is the primitive conflict
resolution needs. See `tests/test_evidence_graph.py`.

### The Publisher

Every state change is a `StateTransitionProposal` — decided, not committed.
The `Publisher` is the only component allowed to write, and it checks:

| Gate | Refusal reason |
|------|----------------|
| consent | `memory_disabled` |
| revision | `stale_revision(claimed=…, current=…)` |
| scope | `target_belief_scope_mismatch` |
| authority | `evidence_authority_mismatch` |
| invariants | `resurrection_guard`, `target_belief_superseded`, `target_belief_missing` |

A refusal is a no-op: the database is byte-for-byte what it was. See
`tests/test_publisher.py` and `tests/test_runtime_invariants.py`.

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
| `observed_at` | When the engine learned the fact |
| `valid_from` / `valid_to` | When the fact itself was true; open `valid_to` = "still true" |
| `temporal_scope` | current_state / persistent / episodic |
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
1. load_evidence    → Read active beliefs            (Compute)
2. formulate        → LLM candidate understanding   (Compute)
3. validate         → Deterministic checks          (Compute)
4. compile_policy   → Controlled policy enums       (Compute)
5. synthesize       → K3 understanding              (Compute — writes nothing)
6. publish          → StateTransitionProposal → Publisher
                       Publisher re-checks consent,
                       revision and authority, then
                       commits the snapshot
```

Nodes 1–5 are pure Compute. Only Node 6 writes, and it writes through the
Publisher — so a state change that lands during the compute is caught at the
gate rather than silently overwritten. A refusal leaves the database
byte-for-byte unchanged; see `tests/test_runtime_invariants.py`.

## Configuration

All domain-specific content is loaded from YAML files:

- `dimensions.yaml` — dimension definitions (7 dimensions)
- `anchors.yaml` — keyword anchors (41 anchors)
- `display.yaml` — user-facing labels
- `extraction.yaml` — keywords + regex patterns
- `identity_policy.yaml` — predicate → cardinality **and** temporal scope mapping (24 predicates)
- `prompts/` — K2/K3/verification/worker prompt templates

## State Revision Atomics

Every memory mutation (correct, forget) bumps `state_revision` atomically:

```
User corrects belief    Worker computes snapshot
         │                        │
         ▼                        ▼
  correct_belief()         compute_and_publish()
         │                        │
         ▼                        ▼
  bump_state_revision()    reads state_revision = 5
  → SQL: SET rev = 6       (before correction)
         │                        │
         ▼                        ▼
  commit → rev = 6         snapshot published with rev = 5
                                    │
                                    ▼
                             persist_snapshot() checks:
                             snapshot.revision (5) < current_revision (6)
                             → discard stale snapshot
```

Implementation: SQL `UPDATE SET state_revision = state_revision + 1 RETURNING state_revision` — guaranteed atomic, no lost updates even under concurrent transactions.

Toggling the memory switch (`set_memory_enabled`) also bumps the revision, so
a job that read evidence while memory was still enabled cannot publish after
the switch is turned off. Writing the *default* value (enabling a user with no
preference row) is a no-op and does not bump.

## Correct Belief Consistency

`correct_belief()` updates all cognitive triple fields atomically:

```python
old.claim_text = new_claim_text     # text
old.predicate = new_predicate       # triple: predicate
old.object = new_object             # triple: object
old.source = "user_corrected"
session.flush()                     # atomic commit
```

Before/after provenance is logged as a BeliefEvent:
```json
{
  "before": {"claim_text": "...", "predicate": "...", "object": "..."},
  "after":  {"claim_text": "...", "predicate": "...", "object": "..."},
  "correction": "new claim text",
  "note": "user correction reason"
}
```
