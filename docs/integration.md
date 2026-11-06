# Integration Guide (external interface contract)

This document is the **external contract** for a host application (an
embedder / pilot product) integrating `mirror-memory` v0.2.0. It covers
every *stable* interface on the public `MemoryEngine`
(`src/mirror_memory/api.py::MemoryEngine`): parameters, exceptions,
whether the call **blocks**, and **thread-safety**. It also states the
deletion-promise boundary the product's own copy must repeat, and the
deployment constraints around network sync.

For a quick reference of the same methods, see
[api-reference.md](api-reference.md). For tenant plumbing, see
[tenancy-context.md](tenancy-context.md).

> **Version / tag.** This document targets **v0.2.0**. The version is set in
> `pyproject.toml` and `src/mirror_memory/__init__.py` (`__version__`). The
> git tag `v0.2.0` is applied **by the user after this branch is merged into
> `main`** — this change does not create or push a tag.

## 0. How to read the two columns

- **Blocking?** — Whether the call performs database I/O and/or an LLM round
  trip on the calling thread. All public methods are synchronous: they
  block the calling thread for the duration of their work. "Blocking" below
  flags methods whose blocking is unbounded-by-mirror (an LLM call) versus
  bounded by the configured DB timeouts.
- **Thread-safe?** — Whether one `MemoryEngine` instance may be shared by
  concurrent threads. In short: **yes for normal operation** (engine state
  that needs it is lock-guarded, and every tenant-bound session is opened
  inside its own `tenant_scope`), with the caveats noted per method. A
  fresh `MemoryEngine` per tenant is the recommended shape; the server
  shares one engine per tenant and one SQLAlchemy pool across tenants.

## 1. Construction and tenancy (init)

```python
from mirror_memory import MemoryEngine

engine = MemoryEngine(
    config_path="config/",
    database_url="sqlite:///mirror_memory.db",
    llm_api_key="sk-xxx",
    llm_model="deepseek-chat",
    llm_base_url="https://api.deepseek.com",
    tenant_id="default",
    app_id="default",
)
```

### Parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `config_path` | `str \| Path` | `None` | Config directory. |
| `config` | `MemoryConfig` | `None` | Pre-loaded config; takes precedence over `config_path`. **One of `config_path`/`config` is required.** |
| `database_url` | `str` | `sqlite:///mirror_memory.db` | SQLAlchemy URL. PostgreSQL selects the RLS posture. |
| `llm_client` | `object` | `None` | Pre-built LLM (must expose `generate()`); wins over the `llm_*` trio. |
| `llm_api_key` | `str` | `None` | Built-in OpenAI-compatible adapter (used only when `llm_client` is None). |
| `llm_model` | `str` | `gpt-4o-mini` | Model id. |
| `llm_base_url` | `str` | `None` | Base URL for non-OpenAI providers. |
| `tenant_id` | `str` | `"default"` | Tenant id; every session the engine opens is bound to `(tenant_id, app_id)`. |
| `app_id` | `str` | `"default"` | App id within the tenant. |
| `single_tenant_mode` | `bool \| None` | `None` (→ env `MM_SINGLE_TENANT`) | End-side single-tenant mode with fail-closed startup checks. |
| `default_purpose` | `str \| None` | `None` | Engine-level default purpose (single-tenant hosts set it once). **`None` means no default** — callers must pass a purpose; there is no implicit one. |
| `deletion_log` | `DeletionLog \| None \| sentinel` | config-derived | Default builds the log from config. An explicit `None` means "no log" → deletions are refused (fail-closed). |
| `sync_transport` | `object \| None` | `None` | In-process transport to a sync peer (needs `.db_id` and `.export_tombstones(cursor, limit)`); the network is the host's business. |
| `engine` / `schema_ready` | pre-built engine / `bool` | `None` / `False` | Share one SQLAlchemy pool across tenant wrappers (PostgreSQL/server); a wrapper never disposes a pool it does not own. |

### Exceptions

- `ValueError` — neither `config_path` nor `config` provided.
- `ConfigError` — invalid/missing configuration (raised while loading).
- Startup (on **first entry**, not in `__init__`): `RuntimeError` when
  single-tenant startup checks fail (multi-tenant DB, stored tenant
  mismatch, or mutual exclusion with the server tenant keys);
  `ReplayPending` / `DeletionLogUnavailable` / `StorageError` can surface
  during schema setup and deletion-log reconciliation.

### Blocking / thread-safety

- `__init__` itself does **no** database work; schema setup, deletion-log
  reconciliation, and the startup catch-up run lazily on the **first
  entry** under a lock (`_ensure_db`). Concurrent first calls on the same
  engine are safe: one thread initializes, others wait for the verdict.
- Sharing one engine across threads is supported. A **failed**
  initialization leaves the engine uninitialized so the next call retries
  the whole init (it never bypasses it).

## 2. Purpose authorization: `grant_purpose` / `revoke_purpose`

Purpose-based authorization gates every read and write. A call carries a
`purpose`; the engine resolves it to the engine default if omitted, and
refuses (`PurposeDenied`) if neither exists or the grant does not allow the
operation.

### `grant_purpose`

```python
info = engine.grant_purpose(
    user_id="u1", purpose="care", operations=["recall", "observe"],
    issuer="admin", expires_at=None, expected_version=None, reason="",
)
```

| Parameter | Type | Notes |
|---|---|---|
| `user_id` | `str` | Subject of the grant. |
| `purpose` | `str` | Purpose name. |
| `operations` | `list[str]` | Allowed operations (e.g. `observe`, `recall`, `export`, `explain`, `correct`, `extract`). |
| `issuer` | `str` | Who issues it (audited). |
| `expires_at` | `Any \| None` | Optional expiry instant. |
| `expected_version` | `int \| None` | **First issuance: omit.** Updating an existing grant requires it — the optimistic lock; a stale issuer is refused. |
| `reason` | `str` | Free-text, audited. |

- **Returns**: `dict{user_id, purpose, version, allowed_operations, revoked}`.
- **Exceptions**: `ValidationError`; a version mismatch / concurrency conflict
  raises (optimistic-lock refusal) from `core.authorization`.
- **Blocking**: yes (one transaction; DB-bounded).
- **Thread-safety**: yes. Concurrent issuers cannot clobber each other — the
  CAS on `version` lets exactly one win. Every accepted change appends a
  row to `mm_purpose_grant_events`.

### `revoke_purpose`

```python
info = engine.revoke_purpose(user_id="u1", purpose="care", actor="admin",
                            reason="offboarding", expected_version=None)
```

- **Returns**: `dict{user_id, purpose, version, revoked, existed}`;
  `existed=False` when there was no grant (idempotent no-op).
- **Exceptions**: `ValidationError`; optimistic-lock refusal on a stale
  `expected_version`.
- **Blocking**: yes (DB-bounded). **Thread-safety**: yes.
- Revocation is **immediate**: the next `authorize` re-reads the row, so a
  revoke bites within the same session after commit and on any concurrent
  reader's next transaction.

## 3. `observe` (ingest a turn)

```python
count = engine.observe(user_id="u1", session_id="s1", text="I love painting",
                       turn_count=1, purpose=None)
```

- **Parameters**: `user_id`, `session_id`, `text`, `turn_count=1`,
  `purpose=None`, plus `**kwargs` forwarded to the extraction pipeline.
- **Returns**: `int` — number of claims extracted this turn.
- **Exceptions**: `ValidationError` — empty `user_id`/`session_id`/`text`, or
  `text` longer than 50 000 characters. `PurposeDenied` when the purpose is
  missing or not authorized for `observe`. Sync-gate refusals
  (`SyncCatchingUp`, `SyncCatchupRequired`) when network sync is enabled and
  the edge is catching up / out of grace.
- **Fail-open**: extraction errors never propagate to the caller (the turn
  simply yields fewer/zero claims).
- **Blocking**: yes — includes an **unbounded LLM round trip** when an LLM
  client is configured (bounded only by the LLM client's own timeout).
- **Thread-safety**: yes. Writes are one transaction per call.

## 4. `recall` (read for prompt injection)

```python
block = engine.recall(user_id="u1", query="What are the user's hobbies?",
                      language="en", tail_load=0, trace_hook=None, now=None,
                      purpose=None)
```

- **Parameters**: `user_id` (required), `query=""`, `language="en"`
  (`"zh"`/`"en"`), `tail_load=0`, `trace_hook=None` (additive observability),
  `now=None` (reference clock for freshness/telemetry), `purpose=None`.
- **Returns**: `str | None` — the rendered memory block, or `None` when no
  memory is available.
- **Exceptions**: `ValidationError` (empty `user_id`); `PurposeDenied`;
  `SyncCatchingUp` / `SyncCatchupRequired`.
- **Fail-closed read**: the access-audit row rides the same transaction as
  the read. If the audit cannot be written, **no block is returned**.
- **Blocking**: yes (DB-bounded; no LLM needed for a read).
- **Thread-safety**: yes.

## 5. Forget / delete

Three scopes: whole user, one belief, one session. All of them write the
deletion log **before** the main-DB commit (fail-closed: a log failure rolls
the deletion back), so the deletion can be replayed after a restore.

### `delete_memories` (whole user scope)

```python
result = engine.delete_memories(user_id="u1", purpose=None)
```

- **Returns**: `DeleteResult` — per-table deletion counts (`result.as_dict()`).
- **Exceptions**: `ValidationError` (empty `user_id`); a purpose is
  **required** (a revoked user can still have data erased, but the
  invocation must carry a purpose — missing → `PurposeDenied`);
  `DeletionLogUnavailable` when no deletion log is configured.
- **Blocking**: yes (DB-bounded). **Thread-safety**: yes.
- **Does NOT delete authorization/config**: consent rows, memory preferences,
  and purpose grants survive a delete. "Delete AND revoke" is an explicit
  two-step for the caller.

### `forget_belief` (one belief + everything derived)

```python
ok = engine.forget_belief(user_id="u1", belief_id=123, purpose=None)
```

- **Returns**: `bool` — `True` if found and deleted.
- **Exceptions**: `ValidationError`; purpose required (`PurposeDenied`);
  `DeletionLogUnavailable`.
- **Blocking**: yes. **Thread-safety**: yes.

### `forget_session` (one session summary)

```python
ok = engine.forget_session(user_id="u1", session_id="s1", purpose=None)
```

- **Returns**: `bool`. **Exceptions**: `ValidationError`; purpose required;
  `DeletionLogUnavailable`. **Blocking/thread-safety**: as above.

### The deletion-promise boundary (COPY THIS INTO PRODUCT TEXT)

> **Deleting a belief removes that belief and everything derived from it, but
> evidence that is still shared by at least one surviving belief is retained
> (its raw source text included).** A source message only becomes garbage —
> and is pruned — once its **last** referencing belief is gone. One message
> can back several beliefs, so deleting *one* belief never erases the
> provenance of what remains.

Consequences an embedder must state to its users:

- To guarantee a specific source message is not retained, the product must
  delete **every belief derived from it**, or the whole user scope
  (`delete_memories`) — not just one belief.
- `forget_belief` prunes **orphan** evidence only. This is the M4 deletion
  boundary and is enforced by the code
  (`core/repository._prune_orphan_evidence`: "Evidence is shared: one
  message can back several beliefs, so a row only becomes garbage once its
  last link is gone") and pinned by the deletion tests.

## 6. Export

### `export_page` / `export`

```python
page = engine.export_page(user_id="u1", purpose=None, cursor=None, page_size=500)
for page in engine.export(user_id="u1", purpose=None, page_size=500):
    ...
```

- **Parameters**: `user_id` (required), `purpose` (engine default; **must
  carry the `export` operation**), `cursor` (previous page's `next_cursor`;
  `None` starts), `page_size=500` (1..5000).
- **Returns**: `ExportPage` — `{schema_version: "mm-export/1", user_id,
  section, items, next_cursor, revision, exported_at, authorization}`.
  Sections run `beliefs` → `evidence` → `events`; `next_cursor is None` on
  the last page. `export()` is a generator over `export_page`.
- **Exceptions**: `PurposeDenied` (missing `export` op, or revoked/expired —
  including **between pages**); `InvalidCursor` / `ValidationError` for a
  malformed cursor or out-of-range `page_size`.
- **Blocking**: yes per page (each page is its own short transaction).
- **Thread-safety**: yes per call.
- **Consistency caveat**: pages are **not** one consistent snapshot — each is
  re-authorized and audited independently and memory can change between
  pages. Compare the `revision` field between pages; if it moved, restart the
  export. A grant revoked mid-export refuses the next page (a stream never
  outlives its grant).

## 7. Explain / history

### `explain`

```python
chain = engine.explain(user_id="u1", belief_id=123, purpose=None)
```

- **Parameters**: `user_id` (required), `belief_id`, `purpose` (needs the
  `explain` operation).
- **Returns**: `dict | None` — belief state, event history, evidence sources,
  supersession chain, plus `authorization` (purpose + grant version) and
  `evidence_source_purposes`. `None` if the belief is not found (or not
  visible under the purpose — identical to not-found, no write).
- **Exceptions**: `ValidationError`; `PurposeDenied`.
- **Blocking**: yes (DB-bounded). **Thread-safety**: yes.

### `belief_history`

```python
versions = engine.belief_history(user_id="u1", belief_id=123, purpose=None)
```

- **Returns**: `list[dict]` — the version chain oldest-first; empty list when
  not found. Requires the `explain` operation. Same exceptions/blocking/
  thread-safety as `explain`.

## 8. `metrics.snapshot()`, `snapshot_full()` and per-call cost

```python
from mirror_memory.core import metrics

counters = metrics.snapshot()          # -> dict[str, int]  (historical shape)
full = metrics.snapshot_full(storage=stored)   # everything, one call
cost = metrics.entry_cost("observe")           # per-call averages
recent = metrics.recent_costs("observe")       # ACTUAL per-call records
```

- **`snapshot()`** keeps its historical contract exactly: a plain
  `dict[str, int]` of counters, nothing else. Every existing caller
  (`.get(name, 0)`, iteration over counter names) is unaffected.
- **`snapshot_full(storage=...)`** is the complete telemetry interface, with
  fixed sections that never collide with counter names:
  - `counters` — identical to `snapshot()`;
  - `gauges` — `{name: float}`;
  - `latency` — `{entry: {count, p50, p95, p99}}` (fixed exponential buckets);
  - `cost` — `{entry: per-call averages}` (see `entry_cost` below);
  - `recent` — `{entry: [per-call records]}` (see `recent_costs` below);
  - `storage` — `{(tenant_id, app_id, user_id): {table: {rows, bytes}}}`,
    ONLY when the caller passes the on-demand `storage=` value (it needs SQL);
  - `enabled` — whether recording is on (`metrics.set_enabled`).
  `metrics.report(storage=...)` returns the same data.
- **`entry_cost(entry)`** — the AVERAGE cost of one invocation (LLM calls,
  measured tokens, failed/partial/unknown usage, SQL statements, divided by
  the invocation count). `None` averages for an entry that never ran — never
  a fabricated zero.
- **`recent_costs(entry)`** — the ACTUAL per-call records (bounded ring, last
  256 invocations, oldest first): `duration_ms` plus that invocation's own
  `llm_calls` / `llm_failed` / token / usage-unknown / usage-partial /
  `sql_statements` deltas. This is real per-observe cost data, not an average.
  Library value only; never emitted to Prometheus; no content or identifiers.
- **Measurement boundary**: one latency sample per invocation covering the
  whole span — including database initialization and the sync gate. During
  cold start the entry scope binds to the engine as soon as it exists
  (`metrics.bind_scope_engine`), so initialization SQL genuinely belonging to
  THIS engine is attributed, while SQL of unrelated engines on the same thread
  never is (an unbound scope attributes nothing — `engine=None` never means
  "all engines"). The one-time initialization cost lands in the first sample.
  The background worker records one sample per REAL claimed extraction job,
  not per idle poll tick.
- **LLM call accounting** (via the instrumented client proxy): calls are
  `llm_calls_total`; provider failures are `llm_calls_failed_total` — this
  includes `ok=False` calls that were swallowed into a fallback return; usage
  is `llm_input_tokens` / `llm_output_tokens` when really reported,
  `llm_usage_partial_total` when only one side was reported,
  `llm_usage_unknown_total` when neither side was. The host's own
  `set_on_call` callback is preserved (chained, never replaced); replacing or
  clearing it through the proxy — or through the raw client reference — is
  honoured without losing either side.
- **Blocking**: no — in-process; no query unless you pass `storage=`.
- **Thread-safety**: yes (module lock; attribution is per-thread).
- **Privacy contract**: counters, gauges, latency, cost and recent records
  never carry memory content, user text, user ids or tenant ids. User/tenant
  ids appear only in the storage section returned by the library (never in a
  Prometheus label). All dynamic Prometheus labels are closed sets
  (`entry`, tombstone `reason` — unknown reasons bucketed to `other`) and
  label values are escaped. `/metrics` exposes only `entry` / `reason` / `le`.

### 8.1 `metrics.storage_stats(engine)` (on-demand, admin/worker only)

```python
with tenant_scope("tenant-a", "app-1"):      # the bound context is the authz
    stats = metrics.storage_stats(eng._engine)
```

- **Authorization**: the tenant is taken from the **bound tenancy context**
  (`core.tenancy.current_tenant()`), never from a caller argument. With no
  bound context the call raises `TenantContextMissing` (fail closed) — it does
  **not** fall back to a default tenant, and it cannot be pointed at another
  tenant. The SQL also carries an explicit
  `WHERE tenant_id = :t AND app_id = :a`, and PostgreSQL+RLS applies on top.
- **Returns**: `{(tenant_id, app_id, user_id): {table: {"rows", "bytes"}}}` —
  all three key parts, so the same `user_id` under another app or tenant can
  never overwrite a row in the map.
- **No cross-tenant variant exists**: a capacity report across tenants would
  need its own verifiable administrator authorization and a dedicated
  privileged connection, which the library does not provide. Do not
  reintroduce one behind `system_scope`.
- **Blocking**: yes — one aggregate query per table; call it from admin tooling
  or a low-frequency worker task, never from a request path.

## 9a. Tenant trust boundary and the PostgreSQL app-role/RLS prerequisite

**Where the tenant identity comes from — and what it is allowed to trust.**

- `tenant_scope(tenant_id, app_id)` is a **trusted in-process boundary**, not
  an authentication mechanism. It is opened by (a) the engine entry point with
  the engine's own pinned `(tenant_id, app_id)`, (b) the server's tenant-key
  middleware AFTER a Bearer key has been matched to a registered tenant, or
  (c) a maintenance/administration path that is deliberately cross-tenant
  (`system_scope`, allowlisted). Application code must never open
  `tenant_scope` with a value that originates from a client-supplied header,
  payload field, or envelope claim.
- Everything tenant-scoped reads its identity from this context
  (`core.tenancy.current_tenant()`). `metrics.storage_stats(engine)` is an
  example: it refuses to run unbound (`TenantContextMissing`, fail closed) and
  takes **no tenant argument**, so a caller cannot widen or redirect the
  report.
- **PostgreSQL prerequisite**: the trust boundary is only as strong as the
  database login behind it. The hosted shape runs tenant work through the
  **app role** (`core.rls.app_role()`) with row-level security ENABLE+FORCE on
  every tenant table and both tenant policies installed
  (`MM_DATABASE_URL` + `MM_PG_RLS=1`; the owner login is only for
  migrations/DDL). A deployment that points tenant work at the owner/superuser
  login silently loses the second line of defence — the in-process context
  still scopes every statement, but a context bug would no longer be caught by
  the database. The RLS posture is verified at startup (`prepare_runtime_engine`)
  and by the dedicated RLS tests.
- On SQLite there is no RLS; the in-process context (and the engine's own
  explicit predicates) is the only defence. SQLite is the end-side/embedded
  shape — treat the process boundary as the security boundary.
- Two engines may share one PostgreSQL pool (the server does); tenants are
  separated by the per-transaction tenant GUC + RLS, never by separate pools.

## 9. Deployment constraints: sync routes and transport

`mirror-memory` ships **two** deployment shapes:

1. **No network sync** (the default library / single-box server). There are
   no sync HTTP routes and **no transport is needed**. If
   `config.sync.enabled` is `true` but the host injected **no**
   `sync_transport`, the engine is in **no-transport gate mode**
   (`_gate_without_transport`): the stored channel history in *this*
   database is the only trustworthy source — reads serve under the standard
   offline grace measured from the stored completion and **refuse**
   (`SyncCatchupRequired`) when the grace is out, when there is none, or when
   the completed peer cannot be located unambiguously. Writes (offline
   deletions) stay available. No catch-up can run; the stored instant is
   never refreshed or invented.

2. **Network sync enabled** (end↔cloud). The sync channel carries
   cross-device *tombstones* (content-free deletion records). When you
   enable it and add routes that move a `TombstoneBatch` between databases,
   **all of the following are mandatory**:

   - **Routes MUST authenticate the tenant.** The tenant is derived from the
     authenticated credential (server tenant-key middleware), never from a
     client-supplied tenant header. Client headers are never read.
   - **Routes MUST pass `auth_tenant`** to `apply_tombstones(..., auth_tenant=(tenant_id, app_id))`.
     A receiver engine *is* its tenant; a mismatch between `auth_tenant` and
     the engine's tenant is a routing bug and is **refused**
     (`TombstoneBatchRejected("tenant", ...)`), never guessed.
   - **The transport MUST bind its peer `db_id` to the authenticated peer**
     at the protocol layer. The channel's cursors, the last-complete-catchup
     instant, `sync_status`, and every expiry error are keyed by **one** peer
     identity that the network layer authenticated. A transport that cannot
     declare `db_id` fails closed (`SyncTransportIdentityMissing`); the
     envelope's self-claimed sender is **never** a substitute for an
     authenticated identity (an envelope/transport sender mismatch is
     refused).
   - **The transport MUST carry its own network timeout.** The engine's
     initialization wait (`_ensure_db`, including the startup catch-up) and
     the catch-up loop have **no internal time cap**; they bound themselves
     only by the transport failing fast. A transport without its own timeout
     can wedge initialization.

   Additional receiver-side behavior to plan for: applying a batch runs each
   entry through the *normal local deletion* path (fingerprints collected,
   local log written, local generation advanced) in **one transaction** — any
   failure rolls the whole batch back. Applied entries are deduplicated by
   `(origin, origin_seq, origin_hash)`, so re-delivery is idempotent. A
   multi-tenant receiver (the cloud) refuses tombstones for users its
   authenticated tenant does not know.

## 10. Exceptions quick reference

Base: `MirrorMemoryError`; subclasses `ConfigError`, `ExtractionError`,
`LLMError`, `StorageError`, `ValidationError`. Purpose refusals raise
`PurposeDenied` (`core.authorization`). Sync refusals raise
`SyncCatchingUp`, `SyncCatchupRequired`, or `SyncTransportIdentityMissing`
(`api.py`). Deletion plumbing raises `DeletionLogUnavailable` /
`TombstoneBatchRejected` / `TombstoneCursorPruned` (`core.deletion`,
`core.tombstones`).

## 11. Minimum dependency floors

- **Python ≥ 3.12** (tested; the package metadata still allows ≥3.11).
- **SQLAlchemy ≥ 2.0.49** (2.0.x line).
- **FastAPI ≥ 0.135.3** (HTTP server extras).
- **Pydantic 2.x**.

`tests/test_dependency_matrix.py` asserts the mirror import and a basic
engine smoke-run work under these floors (it skips, with a clear message,
when the *installed* versions are below them, so it is informative in CI
legs that pin the floors).
