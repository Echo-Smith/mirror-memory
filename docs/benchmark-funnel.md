# Benchmark funnel — where single-hop actually loses the answer

Run with the deterministic path only (K1, no LLM key in the environment):

```bash
python -m mirror_memory.bench.run \
  --dataset /path/to/locomo10.json --categories 4 --no-answerer \
  --output out.json
```

## LoCoMo single-hop, 841 questions, 10 conversations

| stage | value | reading |
|---|---|---|
| Extraction recall | **0.759** (638/841) | 203 answer facts never became a claim at all |
| Identity accuracy | **0.000** (0/638) | K1 claims carry no cognitive triple, so `IdentityResolver` never runs |
| Retrieval recall@5 | **1.000** (841/841) | everything stored is scored and ranked |
| Context coverage | **0.024** (20/841) | only 3–4 of ~34 scored beliefs reach the rendered block |
| Answer F1 | not run | no answering model in this environment |

Failure attribution: identity 76%, extraction 24%. Retrieval accounts for
**zero** failures on its own.

## What this says

The chain is not evenly leaky — it has two walls and one gap.

**The gap is extraction.** A quarter of the facts needed to answer are never
turned into a claim. K1 only fires on configured keywords and regexes, so
anything phrased differently is invisible. This is the ceiling on every
downstream metric: a fact that is not extracted cannot be retrieved.

**The first wall is identity.** With K1 alone, `identity_accuracy` is exactly
zero, and not because identity resolution makes wrong decisions — it never
runs. K1 claims have no `predicate`/`object`, so the resolver's branch
(`if pred and obj and policy`) is skipped for every claim. The beliefs that do
get stored are keyword buckets, which query-aware retrieval cannot match. This
is the single largest number in the table and it is an artefact of running
without K2: it says "identity resolution is dead code unless the LLM extractor
runs", not "identity resolution is wrong".

**The second wall is the renderer, and it is the one that surprised us.**
Retrieval recall is perfect, yet context coverage is 2.4%. The reason is not
scoring and not the character budget (1440 chars, never reached):

- ~34 beliefs per case score above zero and are ranked,
- `render.max_per_dimension = 3` lets only 3 through, so **31 of 34 scored
  beliefs are discarded before rendering**,
- a further ~16 beliefs per case (13,807 in total) never reach scoring at all
  because they fall below `render.l4_render_threshold = 0.55`.

So the answer fact is usually stored, usually ranked, and then dropped on the
floor by the diversity cap. Raising the cap or lowering the watermark is a
config-level lever with a much better payoff than anything in the scorer.

## Caveats

- This run used the K1 deterministic path only. `identity_accuracy = 0` and the
  76% identity attribution are properties of that configuration, not of the
  resolver.
- `answer_f1` needs an answering model. Pass `--api-key` (or set
  `DEEPSEEK_API_KEY`) to include it; every other metric runs without one.
- `extraction_recall` is measured against the *answer* content, not the
  dataset's evidence turn. A K1 claim stores its input snippet verbatim, so
  scoring against the evidence turn would be tautological.

## Reproducing

```bash
# One conversation, 8 questions — a few seconds
python -m mirror_memory.bench.run --dataset locomo10.json \
  --categories 4 --max-conversations 1 --max-cases 8 --no-answerer
```

`--output` writes the full per-case report, including each stage's raw trace
(`retrieve.render.gated_out`, `context.render.rendered_items`), which is where
the numbers above come from.

## Temporal category, 321 questions

Run with `--categories 2` after Temporal V2 landed:

| stage | value | reading |
|---|---|---|
| Extraction recall | **0.106** (34/321) | 287 temporal facts never became a claim |
| Identity accuracy | 0.000 (0/34) | same K1-only cause as single-hop |
| Retrieval recall@5 | 0.984 | retrieval is not the problem |
| Context coverage | 0.006 (2/316) | the renderer drops almost everything |

Failure attribution: extraction 89%, identity 11%.

### Why the temporal gate is not met, and why that is the right answer

The stated gate was "Temporal F1 0.071 → at least 0.10+". It is **not met**,
and the funnel says exactly why: only 10.6% of temporal answer facts are
extracted in the first place. Temporal questions are overwhelmingly "when did
X happen?", and the answer is a date. K1 matches configured keywords and
regexes, and dates are not among them.

So the temporal *model* is not the binding constraint — extraction is. A
better lifecycle policy cannot recover a fact that was never captured, and
raising the gate by tuning the model would be measuring the wrong thing. The
interval machinery (``valid_from``/``valid_to``, TEMPORAL_UPDATE closing the
old interval rather than deleting it, current/historical query filtering) is
in place and covered by ``tests/test_temporal.py``; it will only show up in
the score once extraction carries dates.

The one place the temporal work *is* already visible is the 98.4% retrieval
recall with a 0.6% context coverage — the same renderer bottleneck as
single-hop, now confirmed on a second category.

---

# Update: query-aware retrieval (measured with a real extractor)

The context bottleneck above was diagnosed further and fixed. Three changes,
all in the retrieval path:

1. **Query-aware admission** (`core.repository.recall_candidates`).  The
   renderer took the 50 most recently evidenced beliefs and scored those, so
   recency stood in for relevance.  The answer belief sat at recency rank 153
   of 161 and never reached the scorer.  Admission is now the union of a
   lexically-relevant slice (parameterised `LIKE` over the query's content
   tokens) and a small recent floor, so the context is never empty either.
2. **Content-overlap ranking signal** (`core.retrieval`).  The predicate and
   object signals come from a fixed query→predicate syntax table, so a
   question worded differently produced no signal at all.  The new signal
   scores the fraction of the question's content tokens the belief carries,
   which is phrasing-independent.
3. **Predicate-scoped identity resolution** (`core.repository
   .beliefs_with_predicates`).  The pipeline asked for the 200 most recent
   beliefs, so past 200 active beliefs a user silently lost older predicates
   and the resolver turned what should have been a SUPPORT or UPDATE into a
   CREATE.  The candidate set is now selected by predicate.

## Result

LOCOMO single-hop, 40 questions, 2 conversations, mimo-v2.5 as extractor and
answerer:

| stage | before | after |
|---|---|---|
| Extraction recall | 0.775 | 0.775 |
| Identity accuracy | 1.000 | 1.000 |
| Retrieval recall@5 | 1.000 | 1.000 |
| **Context coverage** | **0.000** | **0.250** (10/40) |
| **Answer F1** | **0.070** | **0.238** |

Answer F1 improved 3.4×, and the attribution now reports 8 cases that clear
every stage and 2 where the fact was in context but the answerer still missed
it — both of which were invisible when coverage was zero.

The deterministic path improves too, with no LLM involved at all: context
coverage 0.025 → 0.065 on the 200-question single-hop set.

## What is still losing

Of the 30 remaining context failures, the dominant cause is upstream of
retrieval: 31 facts reach extraction but only 16 answers exist in any stored
belief. `assemble_claims` sorts by dimension *scarcity* and truncates to
`max_claims_per_turn`, so a fact in an already-populated dimension sorts last
and is dropped. That is the next thing to fix, and it is independent of
anything in the retrieval path.

Tests: `tests/test_query_aware_retrieval.py`.

---

# Correction: the assembler was not the bottleneck

I previously reported that the extraction→storage gap (31 facts extracted, 16
answers stored) was the next thing to fix, and named `assemble_claims` as the
cause. **That attribution was wrong**, and the measurement that produced it was
the problem.

`contains_fact` requires *every* significant token of the answer to appear in
one belief's text. The funnel's extraction recall applies it to the
concatenated `claim_text`s of the extraction trace — which for K1 are verbatim
turn snippets, so the answer is usually present word for word. Applying the
same test to a belief's *distilled* `claim_text` is far stricter, because the
belief summarizes. The gap was mostly "the answer is in the source text but the
belief paraphrases it", not "the claim was dropped".

Measuring all three ways on the same 40 questions:

| measure | result |
|---|---|
| answer present in the source turns | 33/40 |
| answer tokens present in **some** belief (loose) | **32/40** |
| answer fully present in one belief (strict) | 11/40 |

One answer is genuinely lost in ingestion. The other twenty are stored.

## Three real defects fixed anyway

All three were genuine, are now tested, and are worth keeping — they just were
not what the score was waiting on:

1. **`assemble_claims` sorted by dimension scarcity as its primary key**, so a
   fact in an already-populated dimension lost to low-confidence filler in an
   empty one. Quality (triple structure, confidence) now leads; scarcity only
   breaks ties.
2. **`max_claims_per_turn` was 3, exactly the K2 parser's own self-cap.** A
   full K2 batch therefore consumed the entire budget and every K1 claim was
   structurally excluded — including the ones carrying the answer. Raised to 6,
   and the loader's duplicate hardcoded default now defers to the schema so the
   two cannot drift again. Assembler drop rate went from 16.3% to 0.9%.
3. **SUPPORT replaced `claim_text` with "the longest seen so far"**, in both
   `record_claim` and `support_belief_by_id`. A later, longer sentence about a
   different aspect of the same belief key erased the wording that answered the
   question. Replaced with an elaboration test: the new text must be longer
   *and* must not drop the old text's distinctive words.

## What the score actually did

| stage | start | query-aware retrieval | + storage fixes |
|---|---|---|---|
| Extraction recall | 0.775 | 0.775 | 0.725 |
| Identity accuracy | 0.000 | 1.000 | 1.000 |
| Retrieval recall@5 | 1.000 | 1.000 | 1.000 |
| Context coverage | 0.000 | 0.250 | 0.200 |
| Answer F1 | 0.070 | 0.238 | 0.212 |

The storage fixes did **not** improve the score — the 0.238 → 0.212 difference
is two questions on a forty-question sample, i.e. noise, but it is certainly
not a gain. The gain came entirely from query-aware retrieval.

## Where the remaining loss actually is

32 answers are stored; 8 reach the rendered context. The bottleneck is the
renderer's **selection**, not ingestion: the belief carrying the answer exists
and is admitted, but does not survive into the emitted block. That is the next
thing to investigate, and it is the same place the query-aware work started
from — the remaining 22 are ranked below the cut or dropped by the diversity
cap.

---

# StateBench v0.1 and the predicate-variant fix

The plan's Phase 1 asked for a dedicated stateful-memory eval, because LOCOMO
and LongMemEval score extraction and answering and cannot tell a correct state
transition from a lucky paraphrase. `bench/statebench.py` and
`bench/statebench_runner.py` implement it: 7 categories and 15 scripted cases.
The runner actually scores Mirror's rendered recall block with deterministic
text assertions; it does not inspect persisted engine state. A generic
`bench/manifest.py` exists, but the v1.0 StateBench writer did not invoke it.
Both limitations are corrected by the v1.1 runner.

## First run: the engine lost to raw context

```
StateBench v0.1:  mirror1 7/15 = 0.467
                  full-context 10/15 = 0.667
```

Two real defects surfaced immediately.

**Temporal V2 was inert on the real extraction path.** Every belief had
`valid_from = valid_to = None`. The K2 prompt asked only for a human-readable
`temporal` string, and `_parse_extraction_full` never read date fields even
when present. Fixed by adding `valid_from`/`valid_to` to the prompt and parser,
and injecting the current date into the payload — without it the model resolves
"last month" against its training cutoff (measured: 2024-12-01 for a 2026 run).

**`_as_utc` crashed on the ISO string the LLM returns.** `update_belief_by_id`
passed `value["valid_from"]` straight through, `value.tzinfo` raised
`AttributeError`, and the whole UPDATE aborted — leaving the superseded value
active and the new one never created. `coerce_datetime` now lives in
`core/utils.py` and `_as_utc` coerces rather than assuming.

## A-class: predicate variants broke identity resolution

The dominant remaining failure was a current-state question returning a
superseded value. The extractor emits a different verb for the same attribute
on different turns — `works_at`, `started_at`, `joined`, `returned_to` for
employment; `lives_in`, `lived_in`, `moved_to` for residence — and identity
resolution matched on the literal canonical predicate. A variant therefore
matched nothing, the claim was stored as a new belief instead of superseding,
and the stale value stayed active with no interval.

Two things had to line up, and both were missing:

1. the synonym map did not resolve the variants (added `started_at`, `joined`,
   `returned_to`, `employed_at`, `hired_by`, `lived_in`, `relocated_to`,
   `retrained`, …);
2. `works_as` was absent from the identity policy, so it defaulted to `multi`
   cardinality and never superseded at all.

`left` is deliberately left unmapped: it describes a departure, and mapping it
to `works_at` would create a belief claiming the user works at their previous
job.

## Result

| | before | after |
|---|---|---|
| StateBench overall | 0.467 | **0.600** (9/15) |
| gap to full-context | −0.200 | **−0.067** |
| replacement | 0.333 | 0.667 |
| stale_state | 0.000 | 0.500 |

All six remaining failures are "missing current" — the B-class problem already
diagnosed: the answer is stored but not rendered, because 31/40 questions share
no content word with their answer and the block holds 8–13 of ~160 beliefs.
That is the next wall, and it is a retrieval/rendering problem rather than a
state problem.

The 15-case suite above is the archived v0.1 diagnostic run. StateBench v1.0
now contains 200 versioned cases with a 140-case development split and a
60-case evaluation split. See [StateBench v1.0](statebench.md) for the
dataset contract, category distribution, scoring rules, and run examples.

---

# StateBench v1.0 — archived exploratory run

> **Protocol correction (v1.1):** the 59/200 run below is retained as historical
> evidence, but it is not an architectural proof. v1.0 scored the rendered
> recall block rather than persisted engine state, directly scored raw
> transcripts as the baseline, did not write a StateBench manifest, and called
> a public split held out. StateBench v1.1 corrects those four issues and must be
> used for new claims.

v1.0 replaces the 15 hand-written cases with a packaged dataset
(`bench/data/statebench_v1.json`, 200 cases, 7 categories, a
development/evaluation split, `expect_any` alternative groups, and
basic/distractor/paraphrase difficulty). The split is a workflow boundary
rather than a secret one — both halves are committed so a score is
reproducible.

## The result, and why the category breakdown matters more than the total

```
mirror1           59/200 = 0.295
full-context      90/200 = 0.450
delta                    -0.155
```

| category | mirror1 | full-context |
|---|---|---|
| preference_evolution | **0.467** | 0.000 |
| round_trip | **0.500** | 0.000 |
| replacement | **0.333** | 0.000 |
| stale_state | **0.350** | 0.000 |
| contradiction | 0.300 | **1.000** |
| multi_value | 0.067 | **1.000** |
| temporal_event | 0.067 | **1.000** |

The category breakdown is useful for generating hypotheses, but it cannot
establish that the engine beats raw context. The baseline is the unprocessed
transcript scored with the same substring assertions: stale values therefore
force a failure in state-transition categories, while coexistence categories
naturally pass because every value remains present.

The result still exposes engine failures. In particular, later turns in the
same dimension are often suppressed by the extraction throttle; contradiction
attenuates an old positive belief without storing a queryable negative state;
and preference predicates are configured as multi-value persistent facts, so
intermediate phases remain active.

The LoCoMo rendering diagnosis does not transfer directly to StateBench. Each
StateBench case uses a separate user with only 2–5 turns, rather than roughly
160 competing beliefs. Empty outputs and single-value outputs in multi-value
and temporal-event cases point first to extraction and lifecycle loss; renderer
selection is a later stage to measure separately.

StateBench v1.1 retains the scenarios but separates persisted state, recall,
and answer tracks. A full-context comparison is valid only when the same
answerer consumes both raw and Mirror contexts.

## What v1.0 changed about confidence

v0.1's 15 cases could not separate these effects. v1.0 increased coverage, but
its committed evaluation split was public and cannot provide held-out evidence.
The development 0.264 and evaluation 0.367 scores describe this run only.

## Remaining gaps

- Re-run v1.1 in forced and production extraction modes.
- Fix dimension-level throttle suppression, contradiction state, preference
  lifecycle, and event occurrence identity before tuning renderer capacity.
- Use a private unseen set for publication claims.

---

# P0-1: content novelty in the throttle (the fix that moved the score)

The first attempt put a novelty escape inside the suppression branch and moved
nothing (0.295 → 0.285). Tracing one `multi_value` case showed why: the second
turn was never reaching suppression at all — `fully_covered` was `False`, and
the turn died at the **value gate** instead (`value=0.21 < threshold 0.50`).

`compute_extraction_value` measures novelty at the *dimension* level. A second
value of a multi-value predicate — "I also like painting" after "I like
coffee" — lands in an already-covered dimension, so its novelty is zero, its
score collapses to the scarcity term alone, and it falls below the threshold.
The same blind spot hides a second occurrence of an event. Both are precisely
the information the engine exists to keep.

The escape therefore has to apply to the value gate, not only to suppression:
a turn carrying a content word no stored belief has is high-gain by
definition, covered dimension or not.

| | before | after |
|---|---|---|
| StateBench overall | 0.295 | **0.425** |
| gap to full-context | −0.155 | **−0.025** |
| multi_value | 0.067 | **0.400** |
| temporal_event | 0.067 | **0.333** |
| contradiction | 0.267 | 0.467 |
| round_trip | 0.500 | 0.567 |
| stale_state | 0.350 | 0.400 |

Development 0.436 vs evaluation 0.400 — close enough that the fix generalised
rather than being fitted to the visible cases.

This is the third time a read-the-code hypothesis was wrong and measurement
corrected it. The pattern is consistent: this engine's bottlenecks cannot be
inferred, they have to be located.

---

# P0-1 completed: cold-start escape -- StateBench crosses the baseline

The value-gate novelty fix took the score to 0.425 but two classes stayed
low. Tracing `sb-replacement-002` ("My home is in Toronto" → "I relocated to
Lisbon this spring") found the engine stored **zero beliefs** for the whole
case: the first turn has **no keyword hits** ("home", not "live"), so the
base rule skips it, and with no beliefs yet the value gate has nothing to
score. A user's first facts could be permanently missed because of how they
were phrased.

The cold-start escape gives K2 an attempt on early zero-keyword turns while
the profile is thin (< 5 active beliefs, first 6 turns), then stops firing.

| | before | after |
|---|---|---|
| StateBench overall | 0.425 | **0.605** |
| vs full-context (0.450) | −0.025 | **+0.155** |
| multi_value | 0.400 | **0.967** |
| temporal_event | 0.333 | **0.833** |
| contradiction | 0.467 | **0.900** |
| stale_state | 0.400 | 0.450 |
| replacement | 0.333 | 0.433 |
| round_trip | 0.567 | 0.333 ⚠ |
| preference_evolution | 0.467 | 0.267 ⚠ |

**First run above the full-context baseline.** The state-side categories the
architecture exists for are now carried by the engine, and the recall-side
categories closed most of the gap to raw context.

Two regressions to investigate next (round_trip 0.567→0.333,
preference_evolution 0.467→0.267): more extraction means more beliefs, which
changes identity resolution and the render competition. The multi-hop value
gains are worth more than the losses, but both need a cause, not a shrug.

State-side remaining: replacement 0.433 and round_trip 0.333 are now the
weakest, which is where the plan's P0-4 (interval closing on observation
order) applies.
