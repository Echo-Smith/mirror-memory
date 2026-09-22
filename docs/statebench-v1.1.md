# StateBench v1.1

StateBench v1.1 is Mirror's 200-case evaluation for evolving memory state. It
keeps every v1.0 scenario and case id, but replaces the single rendered-text
score with three independent tracks.

## What changed

| Track | Question | Evaluated artifact |
|---|---|---|
| state | Did Mirror store the correct lifecycle? | belief rows, status, polarity, validity |
| recall | Did query-time recall surface useful state? | rendered memory block |
| answer | Did the answerer use the context correctly? | final answer text |

This separation prevents an extraction failure from being reported as a
renderer failure, and prevents a correct internal state from being hidden by
answer wording.

The frozen v1.0 dataset remains at
`src/mirror_memory/bench/data/statebench_v1.json`. The v1.1 dataset is:

`src/mirror_memory/bench/data/statebench_v1_1.json`

Both contain the same 200 case ids and seven category counts.

## Splits

- `development`: 140 public cases for diagnosis and iteration.
- `public_evaluation`: 60 public cases used as a reporting boundary.

`public_evaluation` is reproducible and visible. It is not a private held-out
set. A publication claim requires a separately controlled private set.

## Case contract

Every case declares:

- `transition_type`: the lifecycle behavior under test;
- `query_mode`: `current`, `history`, or `all_occurrences`;
- `state_contract`: required and forbidden structural atom assertions;
- `recall_contract`: assertions on the rendered memory block;
- `answer_contract`: assertions on the answer text.

State assertions can constrain belief status, surface values, polarity,
temporal mode, and match count. Historical queries deliberately allow recall
to include both past and current state; only the final answer must select the
period asked for.

## Extraction modes

Report each mode separately:

| Mode | Purpose |
|---|---|
| `deterministic` | K1-only smoke and regression checks |
| `forced` | K2 on every turn; isolates lifecycle quality from throttling |
| `production` | Current production throttle; measures deployed behavior |

Do not combine scores from different modes. `forced` and `production` require
an extractor client.

## Run

Deterministic smoke:

```bash
PYTHONPATH=src python -m mirror_memory.bench.statebench_v1_1_run \
  --mode deterministic \
  --output artifacts/statebench-v1.1-deterministic.json
```

Forced semantic extraction:

```bash
export MM_LLM_API_KEY=...
export MM_LLM_MODEL=deepseek-chat
export MM_LLM_BASE_URL=https://api.deepseek.com/v1

PYTHONPATH=src python -m mirror_memory.bench.statebench_v1_1_run \
  --mode forced \
  --with-answerer \
  --output artifacts/statebench-v1.1-forced.json
```

Each output automatically creates a sibling
`*.manifest.json` containing the code revision, dirty-tree flag, dataset and
config hashes, prompts, model, generation settings, and extraction mode.

## Baseline rule

Raw conversation text is not scored directly. A full-context baseline must
pass the transcript through the same answerer used for Mirror context, then
score the answer against `answer_contract`. This avoids predetermining category
results from the presence of stale values in the transcript.

## Reporting

Report all of the following:

1. state, recall, and answer scores separately;
2. category, transition, and split breakdowns;
3. extraction mode and extractor model;
4. answerer model when the answer track is run;
5. dataset SHA-256 and run manifest;
6. case-level missing and violated assertions.

