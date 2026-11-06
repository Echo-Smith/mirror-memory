# StateBench v1.0

> Archived protocol. v1.0 is retained for score provenance and regression
> comparison. New runs should use [StateBench v1.1](statebench-v1.1.md), which
> scores persisted state, recall context, and final answers separately.

StateBench is Mirror's deterministic evaluation for **evolving memory state**.
Public long-memory benchmarks test whether a system can answer questions from a
conversation. StateBench isolates the behavior that distinguishes a stateful
memory engine from a text archive:

- replacing a single current-state value;
- returning to an earlier value without creating a duplicate identity;
- retaining several valid values of a multi-value predicate;
- handling explicit contradiction without strengthening the old claim;
- keeping repeated events distinct;
- tracking preferences and goals that change over time;
- preventing stale values from leaking into current-state answers.

## Dataset

The packaged dataset is:

`src/mirror_memory/bench/data/statebench_v1.json`

It contains exactly 200 English cases.

| Category | Cases | Contract |
|---|---:|---|
| replacement | 30 | The newer current-state value replaces the old one |
| round_trip | 30 | A → B → A returns A as current and excludes B |
| multi_value | 30 | Several valid values coexist |
| contradiction | 30 | The explicit opposite is surfaced instead of the old positive claim |
| temporal_event | 30 | Separate dated occurrences remain retrievable |
| preference_evolution | 30 | The final preference or goal is current |
| stale_state | 20 | Current and historical queries select the right interval |
| **Total** | **200** | |

The cases are split into:

- `development`: 140 cases for diagnosis and iteration;
- `evaluation`: 60 cases reserved for final reporting.

Both splits are committed and therefore reproducible, but the evaluation split
is not secret. Do not tune against it when reporting the final score. A private
held-out set can be added later for publication claims.

Difficulty labels are balanced:

- `basic`;
- `paraphrase`;
- `distractor`.

## Case schema

Each case contains:

| Field | Meaning |
|---|---|
| `case_id` | Stable unique id |
| `category` | State transition category |
| `split` | development or evaluation |
| `difficulty` | basic, paraphrase, or distractor |
| `turns` | Conversation turns ingested in order |
| `question` | Query issued after ingestion |
| `expect_current` | Every value must appear |
| `expect_history` | Every historical value must appear |
| `expect_any` | At least one alternative in each group must appear |
| `forbidden` | No value may appear |
| `tags` | Domain and behavior labels |
| `note` | Human-readable reason for the assertion |

The `expect_any` field is necessary for contradictions and preference changes.
For example, either “do not like coffee” or “dislike coffee” is acceptable,
while the old positive claim alone is not.

## Load the dataset

```python
from mirror_memory.bench.statebench import (
    category_counts,
    load_statebench,
    split_counts,
)

all_cases = load_statebench()
development = load_statebench(splits={"development"})
temporal_eval = load_statebench(
    {"temporal_event", "stale_state"},
    splits={"evaluation"},
)

assert len(all_cases) == 200
print(category_counts(all_cases))
print(split_counts(all_cases))
```

## Run the raw-context diagnostic baseline

```python
from mirror_memory.bench.statebench_runner import (
    render_report,
    run_full_context_baseline,
)

baseline = run_full_context_baseline()
print(render_report(baseline))
```

Raw context is diagnostic rather than a production baseline. It naturally
contains stale values, so it should fail many replacement and stale-state
assertions even when it contains every fact.

## Run Mirror

The default path uses Mirror's configured deterministic extraction behavior:

```python
from mirror_memory.bench.statebench_runner import (
    render_report,
    run_full_context_baseline,
    run_statebench,
    write_report,
)

report = run_statebench(
    splits={"development"},
    config_path="config/",
    database_url="sqlite://",
)
baseline = run_full_context_baseline(splits={"development"})

print(render_report(report, baseline))
write_report(report, "artifacts/statebench-development.json", baseline)
```

For model-assisted extraction, construct the provider client and pass it as
`llm_client`. Record the model, provider, prompt hashes, generation settings,
code revision, and dataset hash beside the report. Scores from deterministic
and model-assisted extraction are different tracks and must not be combined.

## Reporting rules

A StateBench report should include:

1. overall score;
2. score by category;
3. development and evaluation scores separately;
4. extraction mode and model, if any;
5. case-level failures;
6. code revision and dirty-tree state;
7. StateBench version and dataset hash.

Do not tune on the evaluation split and then present that score as held out.
Do not compare two runs if the extractor model, prompts, context budget, or
dataset version changed at the same time.

## Release gate (definition)

The release gate is the fixed set of numbers a claim of architectural advantage
must clear. It lives in `src/mirror_memory/bench/gate.py`
(`check_release_gate`) and is applied to a **forced** StateBench v1.1 run
compared against the matching **production** run:

| Check | Threshold |
|---|---|
| state overall | `>= 0.90` |
| state, per category | `>= 0.85` |
| production vs forced gap | `<= 0.05` |
| recall overall | `>= 0.85` |
| answer overall (fixed answerer) | `>= 0.85` |
| reports missing a manifest | `0` |

A forced run is *not* sufficient on its own: the gate also requires full v1.1
case and category coverage, `extraction_mode == "forced"`, and clean K2
extraction health (every dataset turn attempted, zero failed / empty / unparsed
/ schema-rejected calls). A run whose extraction degraded is measuring the
fallback, so the gate rejects it rather than reporting model-labelled numbers.

### Forced vs production semantics

`check_release_gate` evaluates the **forced** and **production** tracks
independently:

- **forced** is the threshold-bearing track. Only the forced run is required to
  prove an every-turn K2 attempt (full coverage + clean extraction health).
- **production** is the engine as configured. Its *true* coverage and failures
  are counted as-is and compared to forced through the fixed gap check
  (`forced - production <= 0.05`). A production throttle never relaxes or edits
  the forced thresholds; a low production score is measured against the same
  fixed numbers.
- A production `not_run` / `error` is surfaced, but it never overwrites the
  forced track's already-computed verdict, and the forced track's failure never
  overwrites a completed production run.

### Four track states

Every track is evaluated on its own and ends in exactly one of four states:

| state | meaning | counts as pass? |
|---|---|---|
| `pass` | ran and cleared its threshold | yes |
| `fail` | ran and did not clear its threshold | no |
| `not_run` | not attempted; a prerequisite (e.g. API key) was absent | **never** |
| `error` | attempted, but execution raised | **never** |

`error` is an **infrastructure failure**, not a measurement failure and not a
`not_run`; it is surfaced loudly and must never be read as a pass or silently
downgraded to `not_run`. Isolation is enforced: a completed track's result is
never overwritten because a *different* track errored.

### Per-track edges (no key / construction failure / run failure)

- **No API key**: `forced` **and** `production` are each their **own** `not_run`
  track — both appear in `verdict.json` and in the summary table; one never
  stands in for the other. (The `answer` track is likewise `not_run`.)
- **Client-construction failure**: if the extractor/answerer client cannot be
  built, every LLM-owned track is `error` whose reason says
  `client construction/init failed: ...`. It is never `not_run` and never pass.
- **Track-run failure**: if a track starts and then raises, that track is
  `error` whose reason says `track run failed: ...`. This is distinct from a
  construction failure so an operator can tell a credentials/config problem
  from a run problem.

In both failure modes the *other* track's completed per-case report and its own
manifest are still written (forced and production each have their own file
pair), so one track’s failure never discards the other’s evidence.

The gate is **fail-closed**: a `not_run` or `error` track does not clear its
threshold. Only a `pass` on every required track (plus coverage, mode, health,
manifest, and the production gap) clears the gate.


Run it locally:

```bash
PYTHONPATH=src python -m mirror_memory.bench.statebench_v1_1_run \
  --mode forced --with-answerer \
  --output artifacts/forced.json \
  --gate --production-report artifacts/production.json
```

## The CI gate (how to read a run)

`.github/workflows/statebench-gate.yml` runs the gate weekly, on manual
dispatch, and on pull requests that touch
`src/mirror_memory/extraction/**`, `src/mirror_memory/core/**`, or
`src/mirror_memory/bench/**`.

It is **report-only**. The workflow conclusion is never red: the gate step is
`continue-on-error` and the verdict is surfaced as data, not as a CI failure.

### Dataset version

The runner loads `statebench_v1_1.json` (`statebench_v1_1_path()`), whose in-file
`version` is `1.2.0` and whose `case_count` is 200. The legacy
`statebench_v1.json` (`version 1.0.0`, 200 cases) is **not** used by the v1.1
runner. The report carries `dataset_version`; the summary prints it explicitly
so v1 and v1.1 results are never conflated.

### Artifacts (distinct, non-overwriting paths)

The CI helper writes — and the workflow uploads — the following files. No path
overlaps another, so a failed track can never erase a completed one's evidence.
`forced` and `production` are saved **separately**, each with its **own**
manifest:

| artifact | contents |
|---|---|
| `statebench-gate.json` | raw per-case report of the gate subject (the deterministic base run, or the forced run when it succeeded) |
| `statebench-gate.manifest.json` | commit, dataset SHA-256 + dataset_version, config/prompt hashes, model |
| `statebench-gate.forced.json` | **forced** run raw per-case report (only when the LLM tracks ran) |
| `statebench-gate.forced.manifest.json` | forced run's own provenance manifest |
| `statebench-gate.production.json` | **production** run raw per-case report (only when the LLM tracks ran) |
| `statebench-gate.production.manifest.json` | production run's own provenance manifest |
| `statebench-gate.verdict.json` | gate verdict + per-check table + four-state verdict per track |
| `statebench-gate-summary.md` | Markdown summary (first line `PASS:` / `FAIL:`) |

A forced-track failure or error does **not** discard the production run's output,
and a production failure does **not** discard the forced run's output: both are
written before the gate is evaluated, and the failure is reported as that
track's `error` in the verdict.

Read the run in `$GITHUB_STEP_SUMMARY`:

- the **first line** is `PASS: ...` or `FAIL: <failing items>` (infrastructure
  `error`s are prefixed so they cannot hide);
- a per-check table lists each check's value, threshold, and pass state;
- a per-track table lists `state`, `recall`, and `answer` with their four-state
  label (`pass` / `fail` / `not_run` / `error`) shown explicitly.

LLM-required tracks (forced extraction and the answerer) run **only** when the
`MM_LLM_API_KEY` secret is present. When it is absent those tracks are recorded
as `not_run`, the report says so, and the gate treats them as failures. If a
track is attempted and raises, it is recorded as `error` (not `not_run`). A
`not_run` track is **not** a pass and must never be read as one.

## When an external "advantage" claim is permitted

An external architectural-advantage claim is permitted only when **all** of the
following hold:

1. the **forced** run clears every gate above (state overall, every category,
   recall, answer, manifest present, and clean extraction health); **and**
2. the forced run matches the **production** run within the gap threshold
   (`forced - production <= 0.05`), i.e. the throttling escapes are not costing
   real recall; **and**
3. no track is `not_run` — a `not_run` track is never a pass, so a run with a
   missing answer or extraction track cannot support a claim.

Until these clear, the only honest external statement is a reproducible
measurement (report the numbers, the mode, and the manifest), not an advantage.

## Validation

The dataset's structural contract is covered by
`tests/test_statebench_dataset.py`:

- exactly 200 unique cases;
- declared category counts;
- 140/60 split;
- balanced difficulty labels;
- no duplicate scenarios;
- non-empty observable assertions;
- filter behavior;
- alternative-answer scoring;
- full-context coverage of all cases.
