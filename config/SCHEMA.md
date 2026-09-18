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

## config.yaml (optional top-level)

- `budget`: dict -- render budget overrides
  - `base`: int (default 480)
  - `floor`: int (default 160)
  - `cap`: int (default 720)

## prompts/

- `k2_system.txt` -- K2 LLM semantic extraction system prompt
- `k3_system.txt` -- K3 understanding synthesis system prompt
- `verification.txt` -- hypothesis verification prompt
- `formulate.txt` -- Worker formulate prompt
