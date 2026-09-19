# Configuration Schema

## dimensions.yaml

- `dimensions`: list of dimension objects
  - `id`: str (required) -- unique identifier (also accepts `dimension_id`)
  - `name`: dict -- `{"zh": "...", "en": "..."}`
  - `description`: str -- human-readable description
  - `render_priority`: int (default 0) -- display ordering

## anchors.yaml

- `anchors`: list of anchor objects
  - `dimension`: str (required) -- parent dimension id
  - `key`: str (required) -- snake_case key
  - `phrases`: list[str] (required) -- bilingual trigger phrases
  - `identify_only`: bool (default false) -- if true, triggers identification only (no automatic claim creation)

## display.yaml

- `labels`: list of display label objects
  - `key`: str (required)
  - `dimension`: str (required)
  - `zh`: str
  - `en`: str

## extraction.yaml

- `keywords`: dict[str, list[str]] -- category to keyword list mapping
- `patterns`: list of regex pattern objects
  - `regex`: str (required) -- Python regex pattern
  - `dimension`: str (required)
  - `key`: str (required)
- `context_tags`: list[str]
- `max_claims_per_turn`: int (default 3) -- max claims per extraction turn
- `llm_every_turns`: int (default 5) -- LLM extraction fires every N turns at minimum
- `llm_min_keyword_hits`: int (default 2) -- minimum keyword hits to trigger LLM regardless of turn count (0 is rejected here; set at runtime for always-extract benchmarks)
- `high_value_dimensions`: list[str] -- dimensions that get a scoring boost for extraction priority (information-gain throttle)
- `extraction_value_threshold`: float in [0, 1] (default 0.5) -- information-gain score at or above which extraction fires early; turns whose hit dimensions are all covered by high-confidence beliefs are suppressed

## config.yaml (optional top-level)

- `budget`: dict -- render budget overrides
  - `base`: int (default 480)
  - `floor`: int (default 160)
  - `cap`: int (default 720)
- `question_value_tiers`: dict[str, int] -- dimension -> priority tier for verification question candidates (higher = more valuable to verify)

## prompts/

- `k2_system.txt` -- K2 LLM semantic extraction system prompt
- `k3_system.txt` -- K3 understanding synthesis system prompt
- `verification.txt` -- verification prompt (consumed by the verification loop to judge confirm/deny/unclear)
- `formulate.txt` -- Worker formulate prompt
