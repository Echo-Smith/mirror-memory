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

The gate is **fail-closed**: any track that is not evaluated is a failure, never
a pass. In particular a track that did not run (`not_run`) does not clear its
threshold.

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
The `statebench-gate` artifact carries the JSON report, the provenance manifest
(commit, dirty flag, dataset SHA-256, model, prompt/config hashes, generation
params), and a Markdown summary.

Read the run in `$GITHUB_STEP_SUMMARY`:

- the **first line** is `PASS: ...` or `FAIL: <failing items>`;
- a per-check table lists each check's value, threshold, and pass state;
- a per-track table lists `state`, `recall`, and `answer` with `not_run`
  shown explicitly.

LLM-required tracks (forced extraction and the answerer) run **only** when the
`MM_LLM_API_KEY` secret is present. When it is absent those tracks are recorded
as `not_run`, the report says so, and the gate treats them as failures. A
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
