# Architecture

## Overview

mirror-memory is a structured memory engine for AI agents. It extracts, stores, retrieves, and renders user memories across conversations.

## Core Abstraction: Belief

A **belief** is a structured observation about a user:

| Field | Description |
|-------|-------------|
| `dimension` | Category (topic, preference, fact, event, goal, pattern, boundary) |
| `key` | Short identifier (e.g., `sleep`, `i_want`) |
| `claim_text` | The specific observation |
| `confidence` | 0.0-1.0, modulated by 7-factor weighting |
| `layer` | L1 (user stated) → L2 (confirmed) → L4 (extracted) → rejected |
| `status` | active / rejected |

## Three-Layer Extraction

```
User message
    │
    ├─► K1 Deterministic (every turn)
    │   └─ Keyword + regex matching → instant claims
    │
    ├─► K2 LLM Semantic (throttled)
    │   └─ LLM extracts structured claims → validated against anchor set
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
| Dimension boost | +4 | Dimension-specific bonus |
| Layer confirmed | +3 | L1/L2 user-confirmed |
| Activity | +2 | Confidence × time decay |
| Evidence | +1 | Multiple evidence items |
| Freshness | +1 | Recent evidence |

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
3. validate         → Deterministic checks
4. compile_policy   → Controlled policy enums
5. synthesize       → K3 understanding
6. persist_snapshot → Versioned snapshot
```