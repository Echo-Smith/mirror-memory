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
