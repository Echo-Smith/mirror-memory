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
  - `key`: str (required) -- base key; a predicate-declaring pattern's stored key becomes `predicate:object`
  - `predicate`: str (optional, default "") -- canonical predicate the pattern asserts. When set, the pattern yields a cognitive triple (predicate + `object_group` capture) and flows through identity resolution like a K2 claim, so a new value supersedes the old one. Empty keeps the claim triple-less, and such claims are no longer persisted as beliefs.
  - `object_group`: int (optional, default 1, >= 1) -- capture group holding the object text; must not exceed the regex's group count
- `context_tags`: list[str]
- `max_claims_per_turn`: int (default 6) -- max claims per extraction turn; must exceed the K2 parser's self-cap of 3 or K1 claims are structurally excluded when K2 returns its maximum
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

## metabolism.yaml

Rule-based memory lifecycle. No model decides what to cool, compact, or delete.

- `retention_classes`: dict[str, dict] -- retention class -> cooling schedule
  - `cool_after_days`: int | null -- days without use before the belief may leave the hot tier; null = never moved on age alone
  - `archive_after_days`: int | null -- days before the belief may leave the default retrieval scan; null = never. Archiving is not deletion.
  - known classes: `canonical` / `preference` / `behavioral` / `episodic` / `transient`
- `retention_predicates`: dict[str, str] -- explicit predicate -> class overrides; beat every derived rule (default `{}`)
- `default_retention_class`: str -- fallback for claims with no usable identity signal (default `preference`)
- `heat`: dict -- heat-score parameters
  - `weights`: dict -- `freshness` / `access` / `evidence_strength` / `authority` / `importance`, should sum to ~1.0
  - `half_life_days`: float (default 60)
  - `thresholds`: dict -- `hot` (0.70) / `warm` (0.40) / `dormant` (0.15), must be strictly ordered
  - `protected_floor`: float (default 0.70) -- heat floor applied to protected beliefs
- `planner`: dict -- cycle safety rails
  - `max_transitions_per_run`: int (default 200, >= 1) -- upper bound on tier transitions + compactions proposed per run; excess cold beliefs are handled on the following run
- `compaction`: dict -- evidence compaction thresholds
  - `min_support_links`: int (default 12, >= 1) -- compaction starts strictly above this many support links on one belief
  - `keep`: dict -- `oldest` (1) / `recent` (3) / `highest_authority` (2) -- size of the live representative sample; `correct` / `contradict` / `verify` links are never folded regardless of these values

Default class derivation (when no override matches): goal lifecycle state or a `wants_to`-style predicate -> `behavioral`; event cardinality -> `episodic`; single cardinality -> `canonical`; anything else -> `default_retention_class`.

## prompts/

- `k2_system.txt` -- K2 LLM semantic extraction system prompt
- `k3_system.txt` -- K3 understanding synthesis system prompt
- `verification.txt` -- verification prompt (consumed by the verification loop to judge confirm/deny/unclear)
- `formulate.txt` -- Worker formulate prompt
