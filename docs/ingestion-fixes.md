# Ingestion fixes: key collision and K2 call volume

Two failures surfaced by the LongMemEval run (500 users, DeepSeek): 19 users
lost to a `UNIQUE` constraint during ingestion, and K2 consuming >80% of
runtime. Both are fixed here, with the measurements that justify each change.

## 1. Key collision — 19 LongMemEval users lost

### Symptom

```
Ingestion failed at LongMemEval user 482 while persisting claim
dim=fact key=works_as_digital_marketing_specialist
sqlite3.IntegrityError: UNIQUE constraint failed: mm_beliefs.user_id, mm_beliefs.key
```

The benchmark skipped the failed user, so 481/500 users were ingested and the
missing 19 silently degraded every downstream score.

### Cause

`mm_beliefs` has `UNIQUE(user_id, key)` with no status filter. When the
extractor produced the *same key* for two different values of a
single-cardinality predicate — "works as a digital marketing specialist" then
"works as a data analyst", both keyed `works_as_digital_marketing_specialist` —
the resolver correctly chose `TEMPORAL_UPDATE`, and `update_belief_by_id` then
inserted the new row under a key the row being superseded still held.

The old code looked for a *superseded* row to revive, found none (the old row
was still `active`), and inserted straight into the constraint.

### Fix

Two guards in `core/repository.py`:

- `_target_key_for_update()` — a caller-supplied `new_key` is respected only
  when it is not the key the superseded row still holds. A value change moves
  to a derived key (`base:object`).
- `_disambiguate_key()` — before the insert, the key is made free for the
  user regardless of status, suffixing if another row already holds it.

The revival path (Shanghai → Berlin → Shanghai) still works, because
disambiguation runs *after* the revival check.

### Verification

`tests/test_longmemeval_ingest.py` and `tests/test_longmemeval_regressions.py`
reproduce the failure and confirm the fix. Reverting the two guards makes 8 of
those tests fail, including the real 500-user dataset ingest.

## 2. K2 call volume — the throttle was a no-op

### Symptom

K2 LLM semantic extraction consumed >80% of ingestion runtime.

### Cause

`_belief_dimensions()` mapped a belief key onto its dimension by splitting on
`"."`:

```python
covered.add(key.split(".", 1)[0])
```

But K1 namespaces its keys with `":"` — `topic:sleep`, `age:30`,
`always:abc123`. Splitting `"topic:sleep"` on `"."` yields `["topic:sleep"]`,
never `"topic"`. So a stored belief was **never** mapped back to its
dimension, and:

- `compute_extraction_value()` returned the same score with and without
  beliefs (measured: `0.85` either way for the same text),
- the suppression gate could never fire,
- the early-trigger gate fired on **928 of 928** signal turns.

The throttle was structurally incapable of throttling.

### Fix

`_key_namespaces()` splits on both separators and offers each prefix, so
`topic:sleep` resolves through the `topic` anchor and the `topic` dimension id.
`age:30` → `fact`, `always:abc123` → `pattern`.

### Effect

Measured over 3 LOCOMO conversations (1451 ingested turns, 928 with a keyword
signal):

| | before | after |
|---|---|---|
| LLM calls | 928 | **78** |
| calls / turn | 0.64 | 0.05 |
| early-triggered turns | 928 / 928 | 9 / 928 |
| suppressed turns | 0 | 817 |
| `extraction_value` spread | pinned ≥ 0.70 | 0.00 – 0.89 |

**91.6% fewer LLM calls**, with `extraction_value` finally discriminating
between turns instead of saturating.

### The honest caveat

This makes the *configured* throttle work as designed. It does not come free,
and the cost is worth stating plainly: with a mock extractor that returns a
claim for every callable turn, answer recall over LOCOMO single-hop drops from
0.635 to 0.355 (127 → 71 facts captured). That is the trade-off the config
already chose via `llm_min_keyword_hits=2` / `llm_every_turns=5` — it simply
never took effect before, because the gates were blind to stored beliefs.

The remaining loss is structural: suppression is **dimension-level**, so a new
fact in an already-mature dimension ("started a new job" after "work is busy")
is suppressed along with genuine re-statements. Making it anchor-level would
recover that, but it changes behaviour the current tests encode
(`test_mature_coverage_suppresses_extraction`) and cannot be validated here
without a real LLM, so it is left alone deliberately.

### LongMemEval results are unaffected

The benchmark runs with `llm_min_keyword_hits=0` — the documented "always
extract" escape hatch — and `should_extract` returns before any value gate in
that configuration. `test_benchmark_escape_hatch_is_untouched` pins it, so the
43.8% LongMemEval figure is measured under unchanged extraction behaviour.

## Reproducing

```bash
python -m pytest tests/test_longmemeval_ingest.py tests/test_longmemeval_regressions.py -q
```

---

# Update: measured with a real LLM (mimo-v2.5)

The numbers above were produced without an answering model, which left two
questions open: does the throttle's call reduction cost extraction quality,
and is the renderer really the bottleneck? Both are now answered with a live
extractor and answerer.

## A third bug: the benchmark CLI wired the extractor wrong

`_build_extractor` and `_build_answerer` took their parameters in **different
orders**, and the extractor was called with `(model, key)` in the `(key, model)`
slots. Every K2 call authenticated with the model name, got a 401, and the
`fallback` in `SemanticExtractor.extract` returned `{"claims": []}`.

The symptom was not an error — it was a run where `identity_accuracy` was
`0.0` and every identity action read `SKIPPED_NO_TRIPLE`, i.e. "K2 found
nothing". A silent misconfiguration that looks exactly like a model finding
nothing. The two builders now share one signature, and
`test_answerer_and_extractor_agree_on_argument_order` fails if they drift.

## Reasoning models need thinking disabled

With `max_tokens=800` (the adapter default), mimo-v2.5 returned an **empty
completion** for every K2 call: the thinking tokens consumed the whole budget.
`OpenAILLM` now accepts `extra_body`, and the CLI exposes `--no-thinking`
(`thinking.type=disabled`). This is not a tuning detail — without it, K2
extracts nothing at all and the funnel reports a clean-looking zero.

## The funnel, with K2 actually working

LOCOMO single-hop, 40 questions, 2 conversations, mimo-v2.5 as both extractor
and answerer:

| stage | K1 only | **K2 working** |
|---|---|---|
| Extraction recall | 0.725 | **0.775** (31/40) |
| Identity accuracy | 0.000 | **1.000** (31/31) |
| Retrieval recall@5 | 1.000 | 1.000 |
| Context coverage | 0.050 | **0.000** (0/40) |
| Answer F1 | 0.070 | 0.070 |

Failure attribution moved from "identity 72%" to **"context 78%, extraction
22%"**. Identity and extraction are no longer the problem.

## Where the remaining 78% actually goes

Chasing the context number down, in order:

1. **Extraction → storage.** 31 facts reach extraction, but only **16 of 40**
   answers exist in any stored belief. `assemble_claims` sorts by dimension
   *scarcity* and truncates to `max_claims_per_turn`, so a fact in an
   already-populated dimension sorts last and is dropped.
2. **Storage → scored set.** Of the 16 stored, only **9** appear in the scored
   set at all.
3. **Scored → rendered.** Raising `max_per_dimension` from 3 to 20 moves
   context coverage from 0.075 to 0.100 — **the diversity cap is not the
   lever.** Median stored confidence is 0.90, so the L4 watermark is not the
   lever either.

The answer to "is it extraction or retrieval?" is now: **neither**. With a
working extractor, retrieval recall is 1.000 and the loss is concentrated
between extraction and storage, and then in the renderer's selection of which
ranked beliefs to emit. The next investigation should start at
`assemble_claims`, not at the scorer.

## LongMemEval ingestion, re-run with the real extractor

Users 470–500 — a 30-user window containing user 482, the exact point the
prior run died — re-ingested end to end with mimo-v2.5 as the K2 extractor:

```
users attempted : 30
users ingested  : 30
users FAILED    : 0
turns ingested  : 682
beliefs stored  : 969
elapsed         : 3395s
```

Zero failures across the whole window, including the originally-failing user.
The 19-lost-users failure mode is closed with a live extractor, not only with
the mock that reproduces it. The mock-driven `TestRealDatasetIngestion` covers
all 500 users and is what CI runs; a live sweep costs roughly two minutes per
user at this provider's latency, so it is a spot check rather than a gate.
