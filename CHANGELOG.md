# Changelog

## v0.2.0 — Identity Semantics v1 (2026-09-20)

### Architecture
- **Cognitive Triple Model**: Beliefs now carry `subject → predicate → object` structure
- **IdentityResolver**: SINGLE/MULTI/EVENT cardinality with CREATE/SUPPORT/UPDATE/CONTRADICT/NOOP
- **Canonicalization**: Predicate synonyms (loves→likes, moved_to→lives_in) + object normalization
- **config/identity_policy.yaml**: 24 predicates with cardinality + 20 synonym mappings

### Memory Safety
- **Atomic state revision**: `bump_state_revision()` uses SQL `UPDATE ... SET state_revision + 1`
- **Stale-write barrier**: Worker checks revision before publish, blocks on mismatch
- **All mutations bump revision**: CREATE/SUPPORT/UPDATE/CONTRADICT/CONFIRM/REJECT/CORRECT/FORGET
- **Targeted forget**: `forget_belief()` cascades to snapshots + cancels worker jobs
- **Correction with triple consistency**: `correct_belief()` requires `new_object` for structured beliefs

### Retrieval
- **Query-aware scoring**: predicate_hit(+4), object_hit(+3) in score_belief
- **Query triple extraction**: `_extract_query_triples()` extracts predicates/objects from queries
- **Claim priority assembler**: scarce dimensions first, triple-structured claims prioritized

### API
- `MemoryEngine.correct()` — user-initiated correction
- `MemoryEngine.explain()` — provenance chain
- `MemoryEngine.forget_belief()` — targeted belief deletion
- `MemoryEngine.forget_session()` — targeted session deletion
- `MemoryEngine.set_consent()` — feature-level consent
- `MemoryEngine.get_verification_candidates()` — verification candidates
- `MemoryEngine.run_verification()` — verification judgment

### Testing
- 368 tests, 0 failures
- Real two-session concurrency test (shared SQLite, threading)
- Identity resolution tests (SINGLE/MULTI/EVENT, temporal, contradiction)
- Integration tests (end-to-end extract→resolve→DB→recall)

### Documentation
- 5 ADRs: Memory Identity, Forget Semantics, Correction, Host vs Mirror, Async Worker
- Architecture docs updated with Triple model + IdentityResolver
- API reference covers all 13 public methods
- Bilingual README (English + Chinese)

## v0.1.0 — Initial Release (2026-09-18)

### Core Engine
- Belief lifecycle (CREATE/SUPPORT/CONTRADICT/REJECT/CONFIRM)
- 7-factor confidence weighting
- Activity half-life decay
- Dynamic budget rendering

### Extraction
- K1 deterministic (keyword + regex)
- K2 LLM semantic (throttled, configurable)
- K3 understanding synthesis (async worker)

### Retrieval
- Multi-signal weighted scoring
- Session summary fallback (dual-channel)
- Query-aware topic matching

### Configuration
- YAML-based config (dimensions, anchors, display, extraction, prompts)
- identity_policy.yaml for predicate cardinality
- Pydantic v2 schema validation

### Infrastructure
- FastAPI server with auth, CORS, /health, /observe, /recall, /panel, /beliefs, /delete, /stats
- OpenAILLM adapter (one-liner startup)
- SQLite + PostgreSQL compatible
- 248 tests, 0 failures
