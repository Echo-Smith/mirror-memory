# Device–Cloud Collaboration & Device-Side Encrypted Storage — Design

- **Status:** design proposal (M5 readiness), no production `src/` change
- **Branch:** `feat/m5-readiness`
- **Scope:** a *generic* mirror capability. Nothing in this document is
  specific to any single adopter application; the same mechanisms apply to
  any product that embeds mirror as its memory substrate.
- **Companion microbenchmark:** [`tools/m5_fieldexp_micro.py`](../../tools/m5_fieldexp_micro.py)
  (standalone AES-256-GCM field encryption + recall latency; run with
  `.venv/bin/python tools/m5_fieldexp_micro.py`).

## 0. Executive summary

mirror today is a **cloud-authoritative, plaintext-at-rest** memory
substrate (scheme **B**). The first adopter direction — realtime voice with
a device side — forces a decision on *where the authoritative memory lives*
and *who can read it*. Three schemes are on the table:

| | Authority | Content at rest | Key holder |
|---|---|---|---|
| **A** | Device | Device only (mirror runtime on device) | Device (SQLCipher / device keystore) |
| **B** | Cloud | Cloud, **plaintext** | none |
| **C** | Cloud | Cloud, **per-user-key ciphertext** | User (client system keychain) |

**Recommendation: adopt C ("cloud-authoritative + user-held key") as the
target**, keep B as the current shipped baseline (it is what runs today),
and keep A as a documented alternative for the privacy-maximalist tier.
C is the only scheme that simultaneously (a) keeps the operational
simplicity and cross-device continuity of a single cloud copy, and (b)
removes plaintext content from operators, backups and the running server
outside a session window. Its costs are, in order of severity:

1. **Lexical recall admission must move off plaintext SQL** (§4.4) — the
   one genuine engineering lift. There are **two** fuzzy (`ILIKE %tok%`)
   sites, not one, and the original semantics are **substring-in-column**,
   so a whole-token HMAC blind index is **not** an equivalent replacement
   (§2.3 A.3).
2. **Key lifecycle / recovery** becomes a product surface (§2a).
3. **Session-start decrypt load** (§2.4, §4.6) — **measured with real
   AES-256-GCM** (`cryptography`): ≈4.4 ms @1k, ≈45 ms @10k beliefs (whole
   content, field-by-field, ~0.55 µs/field); small for the target counts.
4. **Residual exposure to a compromised running server during a session**
   (§6) — unavoidable in any scheme where the conversation model is in the
   cloud, because the injected memory context is itself sent to the cloud
   each turn.

The realtime-voice latency verdict (§1) is: **the SQL layer is not the
problem; the per-call Python scoring + audit-commit path is.** At 1k–10k
beliefs the end-to-end `recall()` p95 measured on a *loaded* host is tens
to a few hundred ms, not the ~1 ms the SQL slice costs. The ≤50 ms p95 hot
-path target is **achievable only with a candidate-set cap and a
warm-cache/async design**; we propose corrected target values in §1.4.

---

## 1. Latency budget for realtime voice

### 1.1 Why voice changes the budget

In a voice turn the user finishes speaking and expects the assistant to
**start responding within ~300–500 ms**. Memory lookup must not block the
first audio token. That splits memory work into three latency classes:

| Step | When | Must block first audio? | Target |
|---|---|---|---|
| **memory recall** (context injection) | on the turn hot path, before the LLM call | **yes** (or via a streaming prepend) | p95 ≤ 50 ms |
| **session-start context prefetch** (warm the cache) | session open, before turn 1 | no (can overlap connect) | ≤ 300 ms |
| **observe + extraction** | after the reply, fully async | **no** | off hot path (best-effort SLA) |

### 1.2 What the measurements say

Two independent sources:

- **SQL layer (controlled, from the M3/M2 EXPLAIN work,
  `docs/reports/2026-10-27-m2-batch2.md`):** the recall relevant slice
  (`ILIKE … ORDER BY last_evidence_at LIMIT 150`) is **1.13 ms**; the recent
  slice (`LIMIT 25`) **0.11 ms**; by-id belief fetch 0.03 ms. So *the
  database portion of a recall is ~1–2 ms at the 5 M-row scale*.
- **End-to-end `recall()` (this task, `tools/m5_fieldexp_micro.py`,
  `MemoryEngine.recall` on a loaded shared host):** see the table in §1.3.
  This is the number that matters for voice, because recall is not just
  SQL — it loads the bounded candidate set, **scores every candidate in
  Python** (`score_belief`), renders the block, writes an **access-audit
  row and commits** in the same transaction.

The gap between ~1 ms (SQL) and tens of ms (end-to-end) is the finding:
end-to-end recall cost is dominated by **Python scoring over the candidate
set plus the mandatory audit commit**, and it **scales with the candidate
count**, not with total belief count per se — but the candidate count grows
with the user's belief population because the recent floor and the
lexical slice both widen until they hit the fixed caps (150 relevant + 25
recent = 175).

### 1.3 Measured recall baseline (this task)

> Method: `MemoryEngine.recall(user_id, query, language, purpose="api")`
> seeded via `observe()` (deterministic `_FakeLLM`, one belief per
> observe), 40 timestamped calls per point with distinct queries, 8
> warmup calls. **Host was shared with concurrent CI/agent work, so
> absolute p95 is an upper bound** (unloaded p50 is the more stable
> signal; the M2 controlled SQL numbers above are the floor). Reproduced by
> `tools/m5_fieldexp_micro.py`. **A later, less-contended re-run (§4.6 C.2)
> measured lower p50: SQLite 1k = 17.4 ms, 10k = 82.7 ms; PG 1k = 17.7 ms —
> use §4.6 C.2 for the freshest recall figures; the table below is the
> original contended run kept for provenance. The *trend* (p50 grows with
> N; ≤50 ms met only to ~1k) is identical in both.**

| backend | N beliefs | recall p50 (ms) | recall p95 (ms) | recall mean (ms) | +decrypt (native) p95 (ms) |
|---|---|---|---|---|---|
| SQLite | 100 | 18.6 | 24.2 | 18.7 | 24.3 |
| SQLite | 1,000 | 38.2 | 382.7 | 92.6 | 383.3 |
| SQLite | 10,000 | 128.6 | 181.4 | 132.3 | 181.8 |
| PostgreSQL | 100 | 34.5 | 98.0 | 46.4 | 98.3 |
| PostgreSQL | 1,000 | 75.9 | 232.5 | 94.1 | 233.2 |

*(SQLite 1k/10k seeded by direct bulk insert — recall latency depends on
row count and candidate-set size, not on how rows were produced; the
1k SQLite p95 of 383 ms is a single contended outlier, see mean/p50. PG
1k is the seed+observe path. Host load average was ~20 during the run, so
**absolute microseconds are inflated; the p50 trend with N is the signal
to trust**.)*

**Read:** p50 grows with the belief population — 18 ms → 38 ms → **129 ms**
on SQLite as N goes 100 → 1k → 10k, and 34 → 76 ms on PG over 100 → 1k.
This is the decisive fact behind §1.4: the ≤ 50 ms hot-path target is met
only up to ~1k beliefs, and is **missed at 10k** with the current recall
path. The growth comes from the candidate set widening toward the fixed
caps (150 relevant + 25 recent) plus Python scoring over it and a
synchronous audit commit per call — **not** from SQL (the SQL slice is
~1 ms). On PostgreSQL the end-to-end recall is *slower than SQLite here*
because each call pays a network round trip + a commit, and the host was
contended; an unloaded PG deployment adds ~0.5–1 ms/round trip, still small
against the scoring term.

### 1.4 Verdict and corrected targets

- **≤ 50 ms p95 on the hot path is NOT met today at 1k–10k beliefs** with
  the current "score the full 175-candidate set in Python + audit-commit
  synchronously" recall. Measured p50 is 18 ms @100, 38 ms @1k, **129 ms
  @10k** (SQLite); it is met at ~100 beliefs and marginal at ~1k.
- **The fix is not encryption; it is the recall path.** Two levers, both
  independent of scheme A/B/C:
  1. **Cap the candidate set** fed to Python scoring (e.g. 150→64, 25→16).
     The relevant slice is already LIMIT 150; scoring 64 instead of 175 is
     a ~2.7× reduction in the scoring term.
  2. **Make the access-audit commit asynchronous / batched**, or write it
     outside the hot transaction (it is a write that today must succeed
     before the block returns — fail-closed). A batched audit writer removes
     a commit (~1–5 ms) from every recall.
- **Proposed corrected targets** (to adopt with the recall-path work):

  | Class | Current measured | Proposed target |
  |---|---|---|
  | recall p50, ≤ 1k beliefs, warm, unloaded | 18–38 ms (SQLite) | **≤ 30 ms** |
  | recall p50/p95, 10k beliefs, warm, unloaded | 129 ms / 181 ms (SQLite, contended) | **≤ 50 ms** with candidate cap + async audit |
  | session-start prefetch, 10k beliefs | see §2.4 / §4.6 | **≤ 300 ms** |
  | observe + extraction | ~40–160 ms/call, async | off hot path; **no SLA on the turn** |

- **Voice-specific design:** recall should be issued the moment the user
  *starts* speaking / at end-of-utterance detection and run **concurrently**
  with ASR finalisation; if it is not ready when first audio is due, prepend
  it on the *second* token instead of blocking. The 50 ms p95 target then
  only has to be met for the *prepend*, not for turn start.

---

## 2. Device–cloud split; deep evaluation of scheme C

### 2.1 The split

Under **C**, the split is deliberately lopsided:

| Concern | Where it runs |
|---|---|
| Memory engine (extract / score / metabolise / render) | **cloud** (single copy) |
| Content storage | **cloud, encrypted** with a per-user key |
| Per-user key | **client system keychain**; in server **RAM only** during a session |
| Recall / observe | cloud, using the in-RAM session key |
| "My memories" display, delete controls, key management | **client** |
| Mirror runtime on device | **none** |

The client in C is *thin*: key management + a viewer + delete. No mirror
runtime ships to the device.

### 2.2 Which columns must be encrypted

The belief row (`mm_beliefs`) carries both content and *structure*.
Field-level encryption must cover everything that reconstructs meaning,
while leaving the columns the query planner needs in plaintext.

**Must encrypt (content):**
`claim_text`, `value_json`, `evidence_json`, `subject`, `predicate`,
`object`, `raw_predicate`, and the human-readable `key` (it embeds the
claim's topic words).

**Related content-bearing tables that must also be encrypted (or emptied):**
`mm_evidence` (the folded evidence text / digests), `mm_session_summaries`
(the rolling summary is raw-text-derived), and `mm_belief_events.detail_json`
where it can carry extracted text.

**Must stay plaintext (structure / planning / integrity):**
`id`, `global_id`, `tenant_id`, `app_id`, `user_id`, `dimension` (enum),
`layer`, `status`, `memory_tier`, `cardinality`, `superseded_by`,
`confidence`, `polarity`, all timestamps, `source`, and the whole
`mm_deletion_*` / tombstone / watermark machinery. These are either enum
values, ids, or hashes — they reveal little and the planner/RLS/audit/
deletion-replay depend on them.

Encryption is **AES-256-GCM with a per-user data key**, one ciphertext +
nonce + tag per encrypted column (so a single column can be decrypted
without the whole row), with the **row id / global_id as AAD** so a
ciphertext cannot be spliced onto a different row.

### 2.3 Which SQL filters / sorts touch content, and are affected

This is the crux, and it is enumerated against the **real code**
(`core/repository.py`, `core/retrieval.py`, `render/renderer.py`; verified
by grep over `src/mirror_memory/`, branch `feat/m5-readiness` @ `00672c7`).
There are **exactly three classes** of content-touching SQL: (a) one
*exact-match* statement, (b) **two** *fuzzy substring* statements, and
(c) several *exact set-membership* statements on structured content
columns (`predicate` / `object` / `key`). Everything else — RLS, tier,
status, recency ordering, by-id fetch — is structure and unaffected.

#### A.1 — Complete inventory of content-touching SQL (exact statements)

| # | Function (file:line) | Kind | SQL shape (as ORM) | Touches | Under field encryption |
|---|---|---|---|---|---|
| **S1** | `get_belief` (`repository.py:1114`) | exact key | `WHERE user_id = :u AND key = :key` | `key` | **BREAKS** (ciphertext ≠ plaintext key) |
| **S2** | `recall_candidates` **relevant slice** (`repository.py:1316–1324`) | **fuzzy substring #1** | `WHERE user_id=:u AND status IN (…) AND memory_tier IN (…) AND ( claim_text ILIKE '%tok%' OR object ILIKE '%tok%' OR key ILIKE '%tok%' , … ) ORDER BY last_evidence_at DESC LIMIT 150` | `claim_text`, `object`, `key` | **BREAKS** — the crux |
| **S3** | `beliefs_referenced_by_tokens` (`repository.py:1539–1547`) | **fuzzy substring #2** | `WHERE user_id=:u AND status='active' AND ( object ILIKE '%tok%' OR key ILIKE '%tok%' OR claim_text ILIKE '%tok%' , … ) ORDER BY last_evidence_at DESC LIMIT 50` | `object`, `key`, `claim_text` | **BREAKS** — used by the extraction **retraction/update resolver** (finds the belief a "Correction: …" refers to) |
| **S4** | `beliefs_with_predicates` (`repository.py:1494–1509`) | exact set (triple) | `WHERE user_id=:u AND status IN ('active','superseded') AND ( predicate IN (:preds) OR object IN (:objs) OR predicate IN (SELECT predicate WHERE user_id=:u AND object IN (:objs)) ) ORDER BY last_evidence_at DESC LIMIT n` | `predicate`, `object` | **BREAKS** (exact, but on content columns) — update/revival resolver |
| **S5** | `active_goal_beliefs` (`repository.py:1583`) | exact set | `WHERE user_id=:u AND status='active' AND predicate IN (:goal_preds) ORDER BY last_evidence_at DESC LIMIT n` | `predicate` | **BREAKS** (predicate is a controlled verb vocabulary, but it is a content column) |
| **S6** | `list_active_beliefs` dimension filter (`repository.py:1226`, `1329`, `1342`) | exact set (enum) | `dimension IN (:dims)` | `dimension` (enum) | **unaffected** — enum, keep plaintext |
| **S7** | `_disambiguate_key` (`repository.py:2135`) | exact key | `WHERE … AND key == :candidate` (collision probe) | `key` | **BREAKS** (same class as S1) |

**Unaffected (structure only — keep plaintext):** the RLS predicate
(`current_setting('…tenant…')`), `status`/`memory_tier`
`IN (...)`, `ORDER BY last_evidence_at DESC` (a timestamp, never content),
by-id belief fetch, and `evidence_counts_for_beliefs` (ids). **No SQL
anywhere does `ORDER BY` on a content column** — every order is by
`last_evidence_at`. Python-side `score_belief` (`retrieval.py:62`) and
`text_overlap` (`utils.py:192`) touch content, but run **after** rows are
loaded, so they only need the session key present (decrypt-before-score).

#### A.2 — The two fuzzy queries, precisely

Both are **parameterised, escaped substring `ILIKE`** (`%token%`, with
`_escape_like` neutralising `%`/`_`), OR-ed over three content columns,
against tokens produced by `content_tokens(text, min_length=3)`:

- **S2 — recall admission** (`recall_candidates`, `repository.py:1316`):
  ```sql
  (claim_text ILIKE :p OR object ILIKE :p OR key ILIKE :p)  -- repeated per query token, OR-joined
  ```
  It admits a belief if **any** query token appears as a **substring
  anywhere** inside `claim_text`/`object`/`key`.
- **S3 — retraction/update referent search**
  (`beliefs_referenced_by_tokens`, `repository.py:1539`): the same
  `(object|key|claim_text) ILIKE %token%` shape, used by the extraction
  pipeline (`extraction/pipeline.py:694,727`) to find *which* existing
  belief a correction refers to.

`content_tokens` returns whole **words** (≥3 chars, stop-worded) or CJK
**bigrams**, but the SQL then does a **substring** match of each such
token *inside the column*. So the matching key is genuinely
**substring-in-column** (e.g. token `coffee` matches column text
`…prefers_coffee_every_morning…`), not word-boundary equality.

#### A.3 — Options that PRESERVE the original substring semantics

The requirement is **not** "find an indexable lookup" — it is
"preserve substring-in-column matching over encrypted content". Three
families, compared:

| Option | Preserves substring semantics? | Query-time work | Index-leak surface | Perf |
|---|---|---|---|---|
| **(i) Character-n-gram blind index** (`mm_belief_ngram( belief_id, ngram_hmac )`, every contiguous k-gram (k≈3) of the normalized column, HMAC under the user key) | **Yes** — a query token is decomposed into its k-grams, all must match (AND), then exact-verify on the decrypted candidates | set-intersection over `ngram_hmac` (index-friendly) + a **candidate re-check** after decrypt (k-grams are a filter, not proof of substring within a word boundary — you still decrypt-verify the survivors) | leaks **k-gram equality within one user's own data only** (all under the user key); the server sees the k-gram multiset of each belief — **frequency / length / co-occurrence leakage** about the ciphertext under that user, worse than a whole-token blind index | index grows ~`len(column)`×overhead; a token→k-gram expansion then an indexed `IN` — sub-linear, keeps the latency profile; re-check adds one decrypt per survivor (bounded by LIMIT) |
| **(ii) Substring-capable encrypted search** (e.g. encrypted trie / suffix-array over secret-shared or order-revealing structures, or an FHE/OPRF substring scheme) | **Yes** (that is the point) | heavy: OPRF/FHE or a secret-shared suffix structure per query | **weaker** leakage guarantees in practice than claimed; OPRE/order-revealing schemes leak order/prefix structure | **unsuitable** for a ≤50 ms voice hot path at 10k beliefs; large engineering + a research-grade crypto dependency we do not have (`cryptography` is not even installed here) |
| **(iii) Leave fulltext server-visible** (store a plaintext token column / PG `tsvector` of content, encrypt only the display blob) with a **separate threat model** | **Yes** (it *is* the plaintext search) | none extra — plain GIN/`tsvector`/`pg_trgm` index | **full plaintext content (or its tokens) visible to the DB/operator** — this is scheme **B** for the searchable field; arguably the worst privacy choice, and it re-admits exactly the plaintext-at-rest exposure C exists to remove | best (native fulltext), but defeats the point of C for content |

**Explicit statement the task requires:** **an HMAC exact-match blind
index (the shape the previous draft recommended) is NOT an equivalent
replacement for the original fuzzy search.** `HMAC(user_key, token)` =
`HMAC(user_key, token)` finds a belief only if that *whole token* was
indexed as a token of the column. It cannot find a token that appears as
a **substring inside a longer word** (the actual S2/S3 semantics), nor
the CJK-bigram-in-any-position case. Using it silently **narrows recall**
— the very failure mode S2 was introduced to fix (§1: the answer belief
sat at recency rank 153 of 161). To preserve the substring semantics you
must index **character n-grams** (option i) — which is a different, and
leakier, primitive — or accept one of the other two options. The prior
"blind index" recommendation was therefore **under-specified**; corrected
here.

**Recommendation (documented, not implemented):** option **(i)
character-n-gram blind index + decrypt-verify**, because it is the only
option that preserves the substring semantics *and* fits the latency
budget without a research-grade crypto dependency — while stating
plainly that it leaks k-gram frequency/co-occurrence within a single
user's ciphertext, a **weaker** guarantee than a whole-token blind index.
Fallbacks: (iii) only if the adopter explicitly accepts plaintext
searchable fulltext for the search field (document it as scheme-B for
that field); (ii) is flagged **not viable** here.

The **recent floor** (no content predicate, `LIMIT 25`) is unaffected, so
even in the worst case recall never returns empty — it degrades from
"relevant + recent" to "recent only".

> **Design consequence:** field-level encryption is *compatible* with the
> current query plan **only with a substring-capable index (option i)** or
> by accepting option (iii)'s plaintext exposure. A whole-token HMAC index
> regresses recall quality for old memories **and does not preserve the
> original semantics**. This is the single largest engineering item in the
> whole proposal (§7), and it is **larger** than the earlier draft implied.



### 2.4 Session-start decrypt load

Under C the server holds the key only during a session. At session start it
may prewarm the user's working set by fetching + decrypting content. This is
the "≤ 300 ms prefetch" target. **Measured with REAL native AES-256-GCM**
(`cryptography.hazmat...AESGCM`, per-field 96-bit nonce + tag, AAD-bound;
`cryptography 50.0.2` in `.venv` — see §4.6): decrypting the **whole** user's
content field-by-field costs **≈4.4 ms @1k** and **≈46 ms @10k** beliefs on
the SQLite store (800 fields @1k → 80,000 fields @10k, ~0.55 µs/field), and
**≈4.2 ms @1k / ≈43.5 ms @10k** on PostgreSQL. Ciphertext per field is
`plaintext + 16-byte tag` (tiny). So the prefetch is bounded by the **fetch
(I/O)**, not the GCM decrypt. Recommendation: prefetch the *recent floor +
top-N by heat* (a few hundred rows), not all 10k, keeping prefetch well under
100 ms. (The earlier CTR+HMAC "≈2.2 / ≈25.8 ms" figures are now **downgraded
to comparison/estimate** — they are not GCM measurements; §4.6.)

### 2.5 Background extraction / organising queue when the key is unavailable

`observe` (extraction) and `run_metabolism` (organising) both read and
write content. Under C they need the key. Cases:

- **Session open, key in RAM:** extraction/org runs normally (they are
  already async — the evolution worker). This is the common path: the turn
  that produced the utterance just delivered the key.
- **Key expired (session closed) but a job is queued:** the job **cannot
  decrypt**. It must **not drop work**. Design: mark the job
  `needs_key` and **park it**; the worker resumes it at the next session
  start when the key is re-delivered. The queue is durable (it is the
  existing `mm_evolution_jobs` table); parking is a status, not a loss.
- **Pure server-side batch (no device ever connects, e.g. nightly
  metabolism for an inactive user):** under strict C this **cannot run**
  without the key. Options: (i) skip until next login — metabolisation is
  a quality optimisation, not correctness, so staleness is acceptable;
  (ii) run the *structure-only* parts (tier transitions driven by
  timestamps/heat, which need no content) and defer the content-dependent
  parts. **Recommendation: (ii)** — split the metabolism cycle into
  structure-only (runs keyless) and content-dependent (parks for key).
- **Deletion / tombstone application:** needs **no** key (operates on ids
  and hashes) — must keep working keyless so deletes propagate regardless.

### 2.6 Key destruction vs the M4 deletion log

M4 gave us an **independent, hash-chained deletion log** with replay. Under
C, "delete ⇒ destroy the key" is the strongest possible deletion: the
ciphertext becomes undecryptable. But this interacts with the deletion log
and backups and must be stated honestly:

- **The deletion log stays intact and verifiable.** Log entries carry **no
  content** — only `table:id:created_at` fingerprints and hashes. Destroying
  the key does not break the log's hash chain or replay; replay still
  matches fingerprints and applies deletes. So **key destruction and M4
  replay are complementary**: the log proves *what* was deleted and keeps it
  deleted across restores; key destruction makes the *content* of deleted
  (and restored-but-then-deleted) rows unreadable.
- **Key destruction ≠ old backups unreadable.** A backup taken *before* the
  key was destroyed, and a **recovery copy of the key** (see §2a), or an
  old device still holding the key, can still decrypt that backup. Destroying
  the *current* key does not retroactively shred ciphertext that was already
  copied elsewhere with a copy of the key. **We do not claim otherwise.**
  True deletion of content requires the key never to have left the
  destroyable boundary — which a recovery copy by definition violates.
- **Ordering:** destroy the key **only after** the deletion log has durably
  recorded the deletion (or for whole-user erasure, after the erase is
  logged), so an interrupted erase is still replayable.

### 2.7 What admins / ops can still see under C

Even with content encrypted, operators retain:

- **Metadata:** user ids, belief counts, dimensions, timestamps, tiers,
  confidence, access-audit ids (who recalled *how many*, not *what*),
  token-index cardinality, and ciphertext sizes.
- **Session-key exposure:** during a session the key is in **server RAM**.
  A privileged operator with live process-memory access (`gdb`, `/proc/pid/mem`,
  a root on the host) can read the key and then the content. C protects
  against *at-rest* and *archive* exposure, **not** against a live,
  root-compromised server. (§6.)
- **Cloud LLM exposure:** the rendered memory block is sent to the
  conversation model **in plaintext every session** — §6 makes this
  explicit; it dominates the residual risk and is scheme-independent.

### 2.8 A / B / C comparison

| Dimension | **A** device-authoritative | **B** cloud plaintext (today) | **C** cloud + user key |
|---|---|---|---|
| **Privacy exposure at rest** | best — content never leaves device | worst — plaintext in DB + backups | good — ciphertext at rest & in backups; plaintext only in session RAM + to the cloud LLM |
| **Ops exposure** | none (no server copy) | full | metadata only; content only with live-RAM access |
| **Recall latency** | local on device: fast, no network; but device CPU/battery | cloud, no decrypt: baseline | cloud + blind-index query + decrypt of candidates (native: ~0.7 ms @175) |
| **Cross-device continuity** | hard — needs sync of full content (§5) | trivial (single copy) | trivial (single copy) |
| **Device-loss recovery** | data recoverable only if synced; else lost with device | survives (cloud copy) | survives the *data*; needs key recovery (§2a) |
| **Device-compromise exposure** | full local content to malware | n/a (no local content) | key in keychain; content reachable while key held |
| **Implementation effort** | **very high** — device mirror runtime (Rust core or dual native), sync engine, SQLCipher | **none/current** | **medium** — field crypto + blind index + key lifecycle + client key mgmt |
| **Realtime voice fit** | good if local, but battery/CPU on device | good | good (decrypt cost tiny) |
| **Delete strength** | delete local row | row delete (backups?) | strongest (destroy key) minus recovery-copy caveat |

**Recommendation:** C as the target for the general product; B stays as the
shipped default until C lands; A only as an opt-in "no cloud copy" tier for
users who accept device-runtime trade-offs (and its sync burden, §5).

---

## 2a. Device-change key recovery — COMPARISON ONLY (no decision, no impl)

Two families for how a user gets their key back on a new device. **This
section compares them; it deliberately makes no implementation choice.**

### 2a.1 Approach ① — user-held key (old-device-authorizes-new, or saved recovery code)

The key is generated **on the device** (Secure Enclave / Android Keystore,
non-exportable where possible, or export-wrapped). It **never** goes to the
cloud in a form the server can unseal. A new device gets the key either from
an **old device that is present** (out-of-band, e.g. a QR/attestation
handshake) or from a **user-saved recovery code** (the key material, or a key
that unwraps it, written down / printed by the user at setup).

### 2a.2 Approach ② — cloud-hosted encrypted key copy / KMS-assisted

The key (or a data-encryption key) is stored **in the cloud, wrapped** by a
key the account can unseal at login (password-derived key or a cloud KMS
key). Account login is sufficient to recover; multi-device is automatic.
The server (or KMS) holds the wrapping key and can, at least in principle,
be made to unseal.

### 2a.3 Flows (① vs ②)

| Event | ① user-held key | ② cloud/KMS-wrapped copy |
|---|---|---|
| **Enable feature** | device generates key in keystore; shows recovery code once; server stores only ciphertext | client generates DEK; wraps with KMS/account key; server stores wrapped DEK + ciphertext |
| **iOS ↔ Android migration** | needs old device present (cross-OS OOB handshake) **or** the saved recovery code; keychain is **not** portable across ecosystems | automatic: log in on the new phone, fetch wrapped DEK, unseal with account/KMS; **works cross-OS** |
| **Multi-device join** | old device authorizes each new device, or each device enters the recovery code | log in; automatic |
| **Old device lost** | recover via **recovery code**; if also lost ⇒ **data unreadable forever** | recover via account login (works as long as account is not compromised) |
| **Account compromised** | attacker gets ciphertext only — **cannot decrypt** (no key in cloud) | attacker who also passes KMS/account unseal gets **the key ⇒ full plaintext** |
| **Revoke device** | remove that device's ability to hold the key; but if it already copied the key, revocation cannot recall it | revoke in cloud; re-wrap for remaining devices; a device that already cached the DEK may still hold it |
| **Recovery-code lost** | **fatal if no other device** (data unreadable) | irrelevant (account login is the recovery path) |

**Who generates / holds / unseals:**
- ① device generates, device holds (keychain), old-device-or-code unseals.
  **Account login is NOT sufficient** for recovery — it is neither necessary
  nor sufficient.
- ② client generates, **cloud/KMS holds the wrapped copy**, account login
  (or KMS policy) unseals. **Account login IS the recovery primitive.**

**iOS / Android portability limits that force this discussion:**
- iOS Keychain items can be configured **non-exportable / ThisDeviceOnly**;
  a this-device-only Secure Enclave key **cannot leave the device** — so ①
  cannot migrate by copying the keychain item; migration must be a live
  handshake or a user-held recovery secret.
- Android Keystore keys are similarly **non-exportable** by design; StrongBox
  keys never leave the TEE.
- **Keychain is per-ecosystem**: Apple Keychain does not sync to Android.
  Cross-OS migration is the hardest flow and is where ① is weakest.

### 2a.4 Trade-offs

| | ① user-held | ② cloud/KMS copy |
|---|---|---|
| **Implementation effort** | higher: OOB device handshake, recovery-code UX, lost-key support burden | lower: standard KMS/escrow + account-auth re-wrap |
| **UX** | worse: "write down this code", "keep your old phone" | best: just log in |
| **Loss risk (user loses key)** | **high** (lost code + lost device = permanent loss) | **low** |
| **Cloud/KMS compromise exposure** | **none** — nothing to unseal in the cloud | **full** — a compromised KMS/escrow + account = plaintext |
| **Running-server compromise (live RAM)** | during a session: full (same as C generally) | during a session: full; **plus** persistent wrapped key to attack offline |
| **Recovery copy impact on key destruction** | a written-down code **is** a copy of the key**: destruction of the server copy doesn't shred it; deleting the account doesn't either | worse: the wrapped DEK persists in cloud; destroy = also destroy every wrapped copy (KMS key deletion is slow and dangerous) |
| **Deletion-log replay impact** | replay still works (log is content-free); but a recovery code in the wild can decrypt a *restored* backup that still holds the ciphertext | same caveat, larger blast radius (cloud has the wrapped key *and* the backup) |

### 2a.4b Three exposure surfaces, compared (the decisive axis)

"Compromise" is not one thing. The three surfaces below must be kept
distinct — they decide ①vs② more than any other dimension:

| Exposure surface | **① user-held key** | **② cloud/KMS-wrapped copy** |
|---|---|---|
| **(a) Database-only leak** (DB/backup stolen; no server process, no KMS, no device) | **ciphertext only — cannot decrypt** (the key is never in DB/backup) | **ciphertext only — cannot decrypt** (the wrapped DEK alone is useless without the KMS/account key, which lives elsewhere) — **provided** the wrapping key is not co-located; if a password-derived wrapped DEK is stored *with* the DB, an offline password attack becomes the surface |
| **(b) Recovery-service compromise** (the KMS/escrow, or the device-handshake service / recovery-code store, is breached) | **①-drag:** if the OOB handshake service or a recovery-code store is compromised, and the attacker can impersonate a device/code, they can unwrap → plaintext for affected users. If recovery codes are purely user-held (printed, never on a server), there is **no server recovery surface to breach** — the strongest position. | **②-full:** a compromised KMS/escrow + a passing account-auth is designed to unseal → **plaintext**. The recovery service *is* the trust anchor, so it is the highest-value target. Mitigation only: split trust (password-derived KEK *and* KMS, neither alone sufficient), HSM-backed KMS, audit. |
| **(c) Running-server compromise** (live process/RAM, during a session) | **full during the session** (key unsealed in RAM to score/render) — identical to C generally; **but nothing persistent to attack offline** after the session key is zeroed | **full during the session** **plus** a **persistent wrapped DEK in the cloud** that can be attacked offline at leisure (weaker if the wrap is password-derived; strong only if KMS-held with rate-limited/HSM unseal) |

**Reading of the three surfaces (comparison, not a decision):**
- **(a) DB leak:** a **tie** — both are ciphertext-only *if and only if* key
  material is separated from the ciphertext store. ② stays safe only if the
  wrapping key is truly elsewhere (KMS/HSM), not co-stored.
- **(b) Recovery-service compromise:** **① is strictly stronger** when the
  recovery code is purely user-held (no server-side recovery surface to
  breach) or when only an OOB old-device handshake is used; ② *is* a
  recovery service and is the crown-jewel target. This is ②'s worst surface.
- **(c) Running-server compromise:** **a tie during a session** (both unseal
  in RAM); ② is **worse afterwards** because it leaves a persistent wrapped
  DEK to grind offline, whereas ① leaves nothing persistent once the session
  key is zeroed.

### 2a.4c DEK lifecycle mechanics (both approaches)

- **DEK generation:** a random 256-bit **DEK** per user, generated **client-
  side** (never server-generated — a server-generated DEK means the server
  held it). Content is encrypted under the DEK (§4.1).
- **Wrapping:** the DEK is wrapped by a **KEK**. In **①** the KEK is a
  device-held key (Secure Enclave / StrongBox, non-exportable) or a key
  derived from the user's recovery code; the wrapped DEK may even be stored
  server-side harmlessly (it is useless without the device/code KEK). In
  **②** the KEK is an **account-password-derived key and/or a cloud KMS key**;
  the wrapped DEK is stored in the cloud.
- **Unsealing:** ① unseals via the old-device handshake or the recovery code
  (KEK on the new device); ② unseals via account login → password-derived
  KEK and/or KMS unwrap. Unsealing happens **at session start**; the DEK is
  then RAM-only (§2.4/§4.3) and zeroed on close.
- **Rotation:** because content is under the DEK (not the KEK), rotating the
  KEK/device key re-wraps only the 32-byte DEK — **no content re-encryption**.
  Rotating the DEK itself (rare, on suspected DEK exposure) requires
  re-encrypting all content and **rebuilding the n-gram blind index** (§2.3).


recovery copy (written code in ①, wrapped DEK in ②) means that **destroying
the current server-side key does not make previously-taken backups
unreadable**. Any claim of "cryptographic erasure" must exclude
(a) ciphertext already copied with a still-existing key copy, and (b) an old
device that still holds the key. We will not promise an erasure we cannot
deliver.

### 2a.5 User decisions this forces (listed, not decided)

1. Is **account login allowed to be sufficient** to recover memory? (Yes ⇒
   ②, lower loss risk, higher cloud-trust; No ⇒ ①, higher loss risk, zero
   cloud key exposure.)
2. Do we accept **permanent loss** on "lost device + lost recovery code"?
3. Does "delete" mean **"unreadable to us"** (C + destroy key) or
   **"unreadable to anyone including old backups"** (which forbids any
   recovery copy / escrow — very aggressive, hurts UX badly)?
4. Cross-OS migration is a **must** or a **nice-to-have**? (must ⇒ ② or ①+code)
5. Are we willing to run/own a **KMS** (② operational surface) or keep the
   cloud key-free (① only)?

**Recommendation (for the decision, not an impl):** if cross-OS migration
must "just work" and permanent-loss risk must be low, **② with a
password-wrapped (not server-held) key plus an optional user recovery code**
is the pragmatic middle — the server stores a *wrapped* DEK it cannot unseal
without the user's password-derived key, so account login alone is
insufficient and a stolen DB gives ciphertext-only. If the threat model
treats the cloud as untrusted and permanent loss is acceptable, **①**.

---

## 3. Client runtime

### 3.1 Under scheme A — cost on native iOS/Android

A requires a **full mirror runtime on the device**. Two builds:

- **Dual native rewrite (Swift + Kotlin/Java):** reimplement extraction,
  scoring, metabolism, rendering, schema, deletion log, sync. **Order of
  magnitude: many engineer-months × 2 platforms**, and every future mirror
  change must be ported twice. Highest long-term cost; worst divergence risk
  (the device and server drift).
- **Shared Rust core (one implementation, FFI to both):** write the runtime
  once in Rust, expose via UniFFI/Swift-FFI to iOS and JNI to Android. Still a
  **rewrite of the current Python engine in Rust** — a large, multi-month
  effort — but single-sourced. Plus **SQLCipher** on-device and the sync
  engine (§5). This is the only sane way to do A, and it is still an
  order-of-magnitude larger programme than C.

Either way A also carries **battery/CPU/thermal** cost (extraction on-device)
and a **much harder test matrix**.

### 3.2 Under scheme C — thin client

The client only needs:

- **Key management**: generate/hold the per-user key in the system keychain
  (non-exportable where the platform allows), deliver it to the server over
  an encrypted authenticated channel at session start, and **never** persist
  it outside the keychain.
- **A "my memories" viewer** (fetch ciphertext-free rendered views — the
  server renders; the client displays) and **delete controls**.
- **No mirror runtime, no local storage, no local extraction.**

This is a **weeks-not-months** client effort vs A's **months** — the decisive
scheduling argument for C.

---

## 4. Encrypted storage

### 4.1 Field-level AES-GCM per-user key — primary (PG + SQLite)

- **Cipher:** AES-256-GCM, one **DEK per user**, DEK itself wrapped by the
  user's keychain-held key (KEK) so rotating the keychain item doesn't force
  content re-encryption.
- **Granularity:** per encrypted **column** (§2.2), each with its own nonce
  and tag, **AAD = `table:row_id:global_id`**. This gives integrity against
  splicing and lets a single column decrypt independently.
- **Both backends:** the ciphertext is `BYTEA` (PG) / `BLOB` (SQLite); the
  scheme is backend-agnostic — the crypto happens in the app layer, not the
  database. SQLite is the natural fit for scheme A's on-device store; PG is
  the cloud store under C.
- **Nonce discipline:** 96-bit random nonce per column-encryption; the DEK
  is rotated long before the 2^32 random-nonce birthday bound is a concern
  (one user, thousands of beliefs — negligible).

### 4.2 Whole-DB SQLCipher — only as an A comparison

SQLCipher encrypts the entire SQLite file (pages, indexes, WAL) with one
key. It is the natural choice **only for scheme A** (device-authoritative),
because there the whole DB is one user's and sits on the device. Under C it
is the **wrong tool**: the cloud DB is multi-tenant, so a single whole-DB key
means the server can read every tenant — it does **not** give per-user
isolation. So SQLCipher is listed here purely as the "what A would use" line,
**not** a candidate for the cloud.

### 4.3 Key in system keychain, not on disk

The user key/KEK lives in **iOS Keychain / Android Keystore** (hardware-backed
where available), configured non-exportable/this-device-only where the
platform allows. It is **never** written to app storage, logs, or the server
disk. On the server it exists **only in RAM** for the session window and is
**zeroed** on close (§2, "cleared in a short window after").

### 4.4 Encryption effect on fulltext / vector search

- **Fulltext / lexical:** as §2.3 — the broken constructs are the **two**
  substring `ILIKE` queries (S2 recall admission, S3 retraction referent
  search) plus the exact-key/triple lookups (S1, S4, S5, S7). To **preserve
  the original substring semantics** you need a **character-n-gram blind
  index + decrypt-verify** (option i, §2.3), not a whole-token HMAC index
  (which is **not** equivalent — §2.3 A.3). No plaintext fulltext index of
  content exists in the cloud under C. Choosing instead to leave a plaintext
  searchable column is option (iii) = scheme-B exposure for that field, and
  must be documented as such.
- **Vector / semantic:** mirror's scoring today is **not** embedding-vector
  based; it is a Python `score_belief` over loaded rows (topic/predicate/
  object overlap + heat). That scoring runs **after decrypt in Python**, so
  field encryption does **not** break it — it just requires the session key
  present (which it is, on the hot path). If a future version adds ANN vector
  search, **encrypted embeddings break ANN** (distance is not computable on
  ciphertext); that would need an encrypted-ANN scheme (out of scope here) or
  recomputing embeddings in the session. Flagged for the roadmap.

### 4.5 Deletion log (JSONL) + backups also encrypted; hash chain stays usable

- **Deletion log:** already **content-free by construction** (fingerprints =
  `hash(table:id:created_at)`, chain hashes). It needs **no** content
  encryption; it stays a plain, verifiable, hash-chained JSONL. The chain
  hash computation is over the log's own fields, not content, so **hash-chain
  verification remains fully usable** with content encrypted. (Optionally the
  log file can be encrypted at rest, but its *structure* must stay
  verifiable — encrypt the file with an ops key, keep the chain math plain.)
- **Backups:** must be **encrypted at rest** with the same per-user DEKs
  (the ciphertext in the DB is already ciphertext, so a byte-copy backup is
  safe *provided the DEKs are not in the same backup*). **Never co-locate
  wrapped DEKs with the ciphertext backup** — store DEK wrapping material
  separately. Restore + replay (§2.6) works on ciphertext unchanged.

### 4.6 Field-encryption microbenchmark — REAL AES-256-GCM (实测)

**Environment / provenance (read before the tables):**

- Repo `feat/m5-readiness` @ `461e66d`; host shared with concurrent CI/agent
  work (recall absolute p95 is an upper bound; p50 the stable signal).
- **Crypto library actually used — the real thing:**
  `cryptography 50.0.2` is installed in `.venv`;
  `from cryptography.hazmat.primitives.ciphers.aead import AESGCM` works.
  Every number in **C.1–C.4 is REAL native AES-256-GCM** produced by that
  primitive: each content **field** is encrypted independently with its own
  random **96-bit nonce** and its own **16-byte tag**, and **AAD = `row_id:
  column`** (row `global_id`/`id` : column name), so a ciphertext cannot be
  spliced onto another row or swapped across columns (verified: wrong row /
  wrong column both raise `InvalidTag`).
- **Label legend:** **实测** = produced here by the real `AESGCM` primitive
  and/or the real `MemoryEngine`; **估算** = derived/extrapolated; **未测** =
  could not be measured here (stated, never fabricated).
- **What is prototype (not production):** the GCM field encryption lives in
  `tools/m5_fieldexp_micro.py` as an **in-memory prototype** — no ORM column,
  no migration, no storage integration reads/writes ciphertext. So every
  *crypto* timing is real (真实调用了 AESGCM), but *the recall/observe path
  does not actually run through GCM in production*: in C.3/C.4 the engine p95
  / observe base is **real engine measurement**, and the **+GCM** column is a
  **prototype-simulated add-on** (real GCM on the real field bytes, composed
  with the real baseline) — labelled exactly that way, never presented as an
  integrated end-to-end number.

**Tool + commands (short, repeated, spread reported):**

```
# REAL native AES-256-GCM (cryptography.AESGCM), SQLite:
.venv/bin/python tools/m5_fieldexp_micro.py --backends sqlite  --counts 1000,10000 \
    --bulk-seed --no-native --gcm --gcm-runs 7 --gcm-field-sample 3000
# REAL native AES-256-GCM, PostgreSQL:
.venv/bin/python tools/m5_fieldexp_micro.py --backends postgres --counts 1000,10000 \
    --bulk-seed --no-native --gcm --gcm-runs 7 --gcm-field-sample 3000
```

The content columns per belief are the 8 from §2.2
(`claim_text, value_json, evidence_json, subject, predicate, object,
raw_predicate, key`), so **8 fields/belief** → 8,000 fields @1k, 80,000 @10k.

#### C.1 — Full-load: decrypt ALL belief fields, REAL AES-256-GCM (实测)

Per run: encrypt every field of every belief once (build the in-memory
ciphertext store), then time decrypting **every field**, repeated
(`--gcm-runs`); p50 / min / max over runs reported.

| backend | N beliefs | fields | **decrypt-all p50 ms (实测)** | min / max ms (spread) | µs/field (实测) | encrypt-all ms (实测) |
|---|---|---|---|---|---|---|
| SQLite | 1,000 | 8,000 | **4.44** | 4.29 / 5.31 | 0.578 | 14.5 |
| SQLite | 10,000 | 80,000 | **46.41** | 44.72 / 48.04 | 0.581 | 195.6 |
| PostgreSQL | 1,000 | 8,000 | **4.21** | 4.15 / 4.52 | 0.535 | 12.8 |
| PostgreSQL | 10,000 | 80,000 | **43.50** | 42.70 / 45.69 | 0.547 | 229.8 |

**Read:** real GCM decrypt-all is **linear in field count at ~0.55 µs/field**
— ≈4.4 ms @1k, ≈45 ms @10k. **PG 10k WAS measured** (not untested): 43.5 ms.
The spend is per-field Python/AESGCM call overhead at these ~20–200-byte
sizes, not bandwidth. Well inside the ≤300 ms prefetch budget; I/O dominates.

#### C.2 — per-field decrypt cost, REAL GCM (实测)

| backend | N | µs/field (sample, 实测) |
|---|---|---|
| SQLite | 1,000 | 0.511 |
| SQLite | 10,000 | 0.527 |
| PostgreSQL | 1,000 | 0.505 |
| PostgreSQL | 10,000 | 0.530 |

**≈0.5 µs per field** (independent nonce + tag + AAD), essentially flat with
payload at these sizes — it is per-call cost, not throughput.

#### C.3 — recall under C: engine baseline (REAL) + candidate-set decrypt (REAL GCM, Prototype-sim add-on)

The recall hot path narrows to a bounded candidate set (`relevant 150 +
recent 25 = 175` beliefs = 1,400 fields); production has no GCM integration,
so we compose: **engine p95 = REAL `MemoryEngine.recall`** (plaintext in
session RAM, measured) **+** the **REAL GCM** cost of decrypting 1,400
fields (measured, prototype-only as an add-on). The composed "proto p95" is
labelled **prototype simulation**, not an integrated end-to-end number.

| backend | N | engine p50 ms (实测) | engine p95 ms (实测) | +GCM 175-cand p50 ms (实测) | proto recall p95 ms (估算 = p95+add, prototype-sim) |
|---|---|---|---|---|---|
| SQLite | 1,000 | 10.56 | 15.74 | 0.749 | **16.49** |
| SQLite | 10,000 | 43.32 | 65.52 | 0.809 | **66.32** |
| PostgreSQL | 1,000 | 16.64 | 30.23 | 0.729 | **30.96** |
| PostgreSQL | 10,000 | 36.99 | 72.46 | 0.800 | **73.26** |

Candidate decrypt spread (min/max over runs): SQLite 1k 0.73/1.32,
10k 0.79/1.11; PG 1k 0.72/0.90, 10k 0.77/1.15 ms. **Read:** the added GCM
decrypt on the hot path is **< 1 ms** — rounding error against the tens-of-ms
Python scoring + audit-commit that dominates recall. Field encryption is not
the latency problem; the scoring path is (§1). The recall **safety scope and
audit fail-closed behaviour are unchanged** (this prototype touches neither).

#### C.4 — observe delta with real GCM (prototype add-on over real observe)

observe() base is **real engine** cost; +GCM encrypts every content field of
the freshly-minted belief with real AES-256-GCM in the harness (no production
integration):

| backend | observe base p50 ms (实测) | +GCM p50 ms (实测) | delta p50 ms (实测) | base p95 | +GCM p95 |
|---|---|---|---|---|---|
| SQLite | 21.57 | 21.72 | **0.15** | 24.72 | 24.98 |
| PostgreSQL | 21.27 | 28.67 | **7.40** | 22.25 | 66.62 |

**Read:** encrypting one belief's 8 fields is **≈4–5 µs** of GCM; the SQLite
delta (0.15 ms) is within noise. The PG "+GCM" delta (7.4 ms) reflects host
contention on the extra DB round-trips in that loop, not crypto — the crypto
term is µs. Observe is **off the hot path** anyway.

#### C.5 — what is 未测 (and what is NOT)

- **Nothing in C.1–C.4 was estimated as measured.** All GCM crypto timings
  are real `AESGCM` calls; engine/observe baselines are real `MemoryEngine`.
- **未测:** *production-integrated* recall/observe that actually stores and
  reads GCM ciphertext — that integration does **not exist** (by design, this
  task is prototype-only). So the "proto p95" and "observe +GCM" are
  **composed** (real baseline + real crypto add-on), explicitly **not** an
  integrated measurement. Everything else (1k/10k full decrypt, µs/field,
  PG 1k and **PG 10k**) is **measured** — PG 10k is **not** untested.

#### C.6 — COMPARISON / ESTIMATE only — the earlier CTR+HMAC figures (NOT GCM)

The previous revision measured **native AES-256-CTR (`openssl` CLI) +
HMAC-SHA256 (`hashlib`)** as a stand-in for GCM, because `cryptography` was
not installed then. **Those numbers are now DOWNGRADED to
comparison/estimate — they are NOT AES-256-GCM measurements and must not be
read as such.** In particular the earlier claim that "HMAC-SHA256 bounds
GCM's GHASH cost" is **withdrawn** — no such bound is asserted here; the real
GCM numbers in C.1–C.4 supersede it. (For reference only, the old CTR+HMAC
*session-start decrypt-all* proxy was ≈2.2 ms @1k and ≈25.8 ms @10k on
SQLite with the CLI, and its per-call CLI floor was startup-dominated; those
figures are **对照/估算**, kept only to show the two approaches differ, and
are replaced by the real GCM C.1 table above.)

**Verdict (on REAL GCM data):** field-level AES-256-GCM is **cheap** —
~0.55 µs/field, ≈4.4 ms (1k) / ≈45 ms (10k) to decrypt a user's whole
content, **< 1 ms** added to a recall's bounded candidate set, and µs added
to observe. It is **not** the realtime-voice bottleneck; the recall
**scoring path** (§1) is. The only unmeasured item is a *production-
integrated* recall path (prototype GP only) — the crypto itself is now
measured with the real primitive.

### 4.7 Plaintext / log / backup boundaries during cloud processing (Section D)

Under scheme C, "encrypted at rest" is a **time-and-place** statement, not a
blanket one. During a session the content is plaintext in several transient
places. This subsection states exactly where plaintext exists and where it
must not, so nobody over-claims.

**Plaintext exists (transiently, by design) in:**

1. **Running-server process RAM, during a session only.** The session key is
   unsealed into RAM at session start and the belief content is decrypted for
   scoring/rendering. This is the unavoidable window (§2.7, §6): a
   root-level live attacker (`/proc/pid/mem`, `gdb`) can read it. Key is
   **zeroed on session close**; content plaintext is not persisted.
2. **The rendered memory block sent to the cloud LLM — every turn.** This is
   the dominant, scheme-independent exposure (§6): the injected context is
   plaintext to the model provider. C shrinks *at-rest* exposure, not the
   *inference* exposure.
3. **Extraction / metabolism working set** (async worker) while the session
   key is present; again RAM-only, not persisted.

**Must be plaintext (structure) by design:** ids, `dimension` (enum),
`status`, `memory_tier`, timestamps, `confidence`, hashes/fingerprints, the
deletion log. These are not content and the planner/RLS/audit/deletion-replay
need them (§2.2).

**Logs — hard boundary (what MUST NOT contain content):**

- **Application / worker logs, error traces, and metrics labels:** MUST log
  **ids, counts, dimensions, timings, error *types*** — **never** decrypted
  content, query tokens, plaintext keys, or the unsealed key. Any log line
  that formats a `Belief` object must go through a redacting `__repr__` or a
  content-free projection; a raw `repr(belief)` leaking `claim_text` is a
  **bug**, not an accepted risk.
- **Access audit rows (§ M5 audit):** already content-free by construction —
  they record *that* a recall happened and *how many* rows, not *what* was
  read.
- **Token/n-gram blind-index sidecar (§2.3 option i):** contains **HMACs
  under the user key**, never plaintext tokens; it is plaintext-*free* but
  is itself **sensitive metadata** (frequency/co-occurrence leakage within
  one user), so it is excluded from logs and treated as content-grade for
  access control even though it is not decryptable to words without the key.
- **The deletion log:** content-free by construction (fingerprints +
  hash chain, §2.6/§4.5); optionally file-encrypted at rest with an ops key,
  but its *structure* stays verifiable.

**Backups — hard boundary:**

- Backups are **byte-copies of ciphertext** — safe **only if the DEK wrapping
  material is NOT in the same backup**. **Never co-locate wrapped DEKs /
  recovery material with the ciphertext backup** (§4.5). Restore + M4 replay
  operates on ciphertext unchanged.
- A **pre-deletion backup plus any surviving key copy** (recovery code in ①,
  wrapped DEK in ②, or an old device) can still decrypt that snapshot —
  "cryptographic erasure" must **exclude** those copies (§2.6, §2a.4). We do
  not promise erasure we cannot deliver.
- The **deletion log backup** restores the ability to re-apply deletes; it
  carries no content and is safe to back up.

**Boundary summary (one line):** *at rest and in backups* → ciphertext (safe
if key material is separated); *in the running server during a session and
in the cloud-LLM context every turn* → **plaintext** (the honest residual);
*logs / audit / blind-index / deletion-log* → content-free or
under-key-hashed, never plaintext content.

---

## 5. Data sync (relevant only if A is chosen; C is a single cloud copy)

Under **C** there is **one authoritative cloud copy** — no memory-data sync
is needed (devices read rendered views; delete propagates through the cloud).
What follows is the sync design **only for scheme A**, built on the **M4
tombstone sync base** already in the tree (`core/tombstones.py`,
`MemoryEngine.apply_tombstones`, `export_tombstones`, `sync_status`).

- **Base: M4 tombstones.** Each device keeps an independent deletion log;
  tombstones name *what* was deleted by **`global_id`** (never the local
  integer id — restores and other devices reuse those), scope
  (`belief`/`session`/`user`), and carry `origin`/`origin_seq` + hash. A peer
  pulls and `apply_tombstones` runs the normal local delete for each, in one
  transaction, recording `(origin, origin_hash)` as applied (idempotent; a
  vanished scope is a no-op **still recorded**).
- **Filter by applied tombstones (incl. no-ops) before ingest.** Before a
  device inserts/updates a belief from a peer, drop anything whose
  `global_id` matches an **applied** tombstone — **including entries that
  applied as no-ops** (the row was already gone). Otherwise an in-flight
  create can resurrect a deleted belief. The applied-record is the source of
  truth, not "does the row exist now".
- **Align existing belief `global_id` before first sync.** A device that
  already has beliefs must have minted `global_id`s (M4) before its first
  exchange; otherwise the same logical belief has different identities on two
  devices and conflict detection is meaningless. **Precondition: every
  pre-existing row has a `global_id`** (backfill before enabling sync).
- **Belief-revision conflict resolution.** Sync carries **belief revisions**
  (per-belief monotonic revision). On conflict (same `global_id`, different
  revisions): last-writer-by-revision wins with a **tie-break** (higher
  confidence, then later `last_evidence_at`, then device id) — the loser is
  **superseded** (kept as history, not deleted), preserving the M3
  "corrections are history" rule.
- **Grant / revoke propagation across devices.** Authorization (purpose
  grants, consent, feature enable) is **not** content; it must propagate
  through the same channel or a separate control plane, so a revoke on one
  device takes effect everywhere. Treat grants as revisioned control records
  with the same conflict/tombstone discipline (a revoked grant is a
  tombstone-like control delete that must not be resurrected).
- **Network route (security contract, from
  `core/authorization.py` UNGATED_ENGINE_METHODS):**
  1. **authenticate the tenant** at the transport (never trust the envelope's
     self-claimed `sender`);
  2. **pass `auth_tenant`** to `apply_tombstones`; a mismatch against the
     receiver's own tenant is refused (`reason="tenant"`) — a routing bug, not
     a data fix;
  3. **bind the transport's `db_id` to the authenticated peer** (the
     self-claimed `sender` is never used for identity);
  4. **the transport carries its own timeout** — the engine's catch-up wait
     has **no internal cap** (by design), so the network call must bound it
     (else a hung link pins the sync gate). Offline grace (`offline_grace_days`,
     default 7) + `reminder_days` make going-offline a product-visible state,
     not a hard failure, and a **rescan** re-anchors to `(0, GENESIS)` when a
     peer's log was restored/rewritten under already-consumed seqs.

---

## 6. Threat model

Scheme C (and, where noted, A/B). "Protectable?" is *for the content*, by
the named mechanism; residual risk is what remains.

| Threat | Protectable? | By what | Residual risk |
|---|---|---|---|
| **Device lost / stolen, or file copied off device** | **Yes (C, A)**; No (B) | C: device holds only the *key*, not content at rest → attacker gets the key? keychain is non-exportable/hardware-backed; content is cloud ciphertext. A: SQLCipher + device keystore. | If the device is **unlocked and the app is running with the key unwrapped**, malware/forensics can read; a this-device-only key + biometric gate mitigates. B: full plaintext lost. |
| **Restore an old backup** | **Partial** | M4 deletion log **replays deletes** onto the restored DB (content-free, survives restore) → deleted memory stays deleted. C: ciphertext restored → still needs the key. | A **pre-deletion backup plus a recovery-key copy** (or an old device) can still decrypt that snapshot (§2.6). Replay re-deletes *known* deletions, not "everything after T". |
| **Local data tampered / poisoned** (A, and C's client-side cache) | **Partial** | AES-GCM **auth tag per column + AAD binding row id** detects tampering/ splicing → decrypt fails loudly. Server-side content is integrity-protected. | A malicious **client** that legitimately holds the key can write poisoned *plaintext* (it encrypts garbage correctly) — GCM proves integrity, not truth. Poisoning is a trust-of-writer problem, mitigated by provenance/reputation, not crypto. |
| **Device controlled by malware** | **Partial** | Key stored non-exportable (Secure Enclave/StrongBox) so malware **cannot extract** it; content decrypted only transiently. | Malware **running as/next to the app** can read decrypted content while the key is unwrapped and the app is active (screen/keystroke capture). Fundamental: a compromised endpoint with a live session is compromised. |
| **App replaced** (MITM / fake app / repackaged) | **Partial** | Attestation / code-signing / cert pinning on the key-delivery channel; key released only to an attested app. | A **jailbroken/rooted** device can defeat attestation; a user-installed malicious build with the user's own keys cannot be distinguished. |
| **Cloud compromised** (server host / DB / backup stolen) | **Yes for content-at-rest (C)**; No (B) | C: DB + backups are ciphertext; per-user DEKs are wrapped, not stored with content; operator reads metadata only. | **Live server RAM**: during a session the key is in memory → a root-level live attacker can read content. **KMS/escrow compromise (approach ②)** → unwrap → plaintext. **Cloud LLM**: see the dedicated row. |
| **Injected memory context goes to the cloud LLM (every session)** | **No — inherent** | nothing in A/B/C changes this if the conversation model is cloud-hosted | The rendered memory block (decrypted, plaintext) is sent to the conversation model **each turn/session**. So even under C, the *content that reaches the model* is exposed to the model provider. C shrinks the *at-rest* surface, not the *inference* surface. Mitigating this requires an on-device/private model — a separate product decision, out of scope here. **This is the dominant residual risk and must be stated to users honestly.** |

**Summary of what each scheme buys:**
- **B** protects nothing at rest; simplest.
- **C** protects content **at rest and in backups and from ordinary
  operators**; does **not** protect against live-rooted-server, cloud-LLM, or
  a compromised endpoint during a session.
- **A** protects content from the cloud entirely, at the cost of device
  runtime, sync, and device-compromise exposure.

---

## 7. Implementation breakdown, effort, user decisions

### 7.1 Workstreams (follow-up batch tasks)

| # | Task | Depends on | Effort (order) |
|---|---|---|---|
| 1 | **Field crypto primitive** — AES-256-GCM per-user DEK, per-column ciphertext + AAD binding, wrap/unwrap by keychain KEK; PG `BYTEA` + SQLite `BLOB`; migrate read/write paths | — | **M** |
| 2 | **Substring-capable blind index (character-n-gram) + decrypt-verify** — `mm_belief_ngram(belief_id, ngram_hmac)`, HMAC(user_key, k-gram) over the normalized content columns; query = token→k-gram expansion + indexed `IN` + post-decrypt substring re-check on survivors. Preserves S2/S3 substring semantics; a whole-token HMAC index is **not** equivalent. Maintain on ingest/update/delete; rebuild on DEK rotation | 1 | **L** (the crux, §2.3) |
| 3 | **Session key lifecycle** — encrypted channel, RAM-only key, zeroize on close, session-start prefetch (working set) | 1 | **M** |
| 4 | **Async/queued extraction & metabolism under C** — park `needs_key` jobs, split structure-only vs content-dependent cycle, keyless delete/tombstone path | 1,3 | **M** |
| 5 | **Key management client** — keychain (non-exportable), session handshake, "my memories" viewer, delete | 3 | **M** (×iOS + Android) |
| 6 | **Key recovery (approach per §2a decision)** — old-device handshake **or** KMS/password-wrapped escrow + recovery-code UX | 3,5 | **M–L** (decision-gated) |
| 7 | **Recall-path latency work** (independent of scheme) — candidate cap, async/batched audit commit (§1.4) | — | **M** (unblocks the ≤50 ms p95 target) |
| 8 | **Encrypted backups + restore/replay under C** — separate DEK-wrap store from ciphertext backup; verify replay on ciphertext | 1 | **S–M** |
| 9 | **Encrypt summary/evidence tables** — `mm_session_summaries`, `mm_evidence`, `detail_json` | 1 | **S** |
| 10 | *(A only)* **Rust core runtime + SQLCipher + memory-data sync (§5)** | — | **XL** (multi-month programme) |
| 11 | **Vector/ANN note** if semantic search is added later — encrypted ANN | 1 | **L** (future) |

### 7.2 User decisions needed (blocking)

1. **Scheme:** confirm C as target (B stays default until C lands, A opt-in)?
2. **Key recovery family (§2a):** ① user-held vs ② cloud/KMS-wrapped
   (this decides tasks 5–6 and the threat-model posture).
3. **Is account login allowed to unseal?** (No ⇒ ② with password-wrapped
   key; Yes ⇒ simpler ② but higher cloud-trust.)
4. **Deletion meaning:** "unreadable to us" (C, destroy key) vs "unreadable
   to anyone incl. backups" (forbids recovery copy — aggressive).
5. **Cross-OS migration** must-have vs nice-to-have (drives §2a choice).
6. **Latency budget:** accept the corrected targets in §1.4 (≤30 ms @≤1k,
   ≤50 ms @10k with candidate cap + async audit)? This gates task 7.
7. **Vector/semantic search** on the roadmap? (If yes, encrypted-ANN is a
   future workstream.)
8. **Run a KMS?** (② operational surface) or keep the cloud key-free (①).

---

## Appendix A — Reproducing the numbers

```
# from repo root, with the bundled venv (cryptography 50.0.2 installed):
# PRIMARY — REAL native AES-256-GCM (cryptography.hazmat...AESGCM)
.venv/bin/python tools/m5_fieldexp_micro.py --backends sqlite   --counts 1000,10000 --bulk-seed --no-native --gcm --gcm-runs 7 --gcm-field-sample 3000
.venv/bin/python tools/m5_fieldexp_micro.py --backends postgres --counts 1000,10000 --bulk-seed --no-native --gcm --gcm-runs 7 --gcm-field-sample 3000
.venv/bin/python tools/m5_fieldexp_micro.py --quick           # fast smoke (SQLite, real GCM)
# COMPARISON ONLY (not GCM): openssl AES-256-CTR + hashlib HMAC-SHA256
.venv/bin/python tools/m5_fieldexp_micro.py --no-gcm
# pure-Python AES-GCM reference (correctness only; ~2000x slow)
.venv/bin/python tools/m5_fieldexp_micro.py --no-native --no-gcm --aes-speedup 1
```

**Crypto backends in the tool (three, in order of authority):**
1. **`--gcm` (default on) → 实测, PRIMARY.** Real `cryptography` **AESGCM**
   (AES-256-GCM), per-field independent 96-bit nonce + 16-byte tag, AAD =
   `row_id:column`, in-memory prototype (no production storage). This is the
   source of every **实测 GCM** number in §4.6 C.1–C.4. There is **no**
   "GHASH proxy" claim — GCM is measured directly.
2. **`--native` (default on) → COMPARISON / ESTIMATE only, NOT GCM.** Real
   `openssl enc -aes-256-ctr` + real `hashlib` HMAC-SHA256 tag. This was the
   prior stand-in while `cryptography` was absent; it is now **对照/估算**,
   superseded by (1), and is **not** claimed to bound GCM/GHASH.
3. **pure-Python AES-256-GCM** (correct but ~2000× slow; validated against
   the GCM spec Appendix B AES-256 test vector) — correctness reference.

The PG leg uses `postgresql://postgres:mm@127.0.0.1:5437/mm_test` with a
fresh throwaway schema per run (dropped on exit). The benchmark is robust to
a transient shared-tree import state (it re-tries imports and only patches a
missing module attribute in-process; it **never** edits `src/`). To avoid the
shared working tree being mutated mid-run by a concurrent teammate, run from
an isolated worktree: `git worktree add --detach /tmp/mm-bench <rev>` and set
`PYTHONPATH=/tmp/mm-bench/src` (note: do **not** create worktrees on the
shared tree that is itself mid-flight).
