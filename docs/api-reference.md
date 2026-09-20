# API Reference

## MemoryEngine

```python
from mirror_memory import MemoryEngine

engine = MemoryEngine(
    config_path="config/",
    llm_api_key="sk-xxx",
    llm_model="deepseek-chat",
    llm_base_url="https://api.deepseek.com",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_path` | `str\|Path` | None | Path to config directory |
| `config` | `MemoryConfig` | None | Pre-loaded config (takes precedence) |
| `database_url` | `str` | `sqlite:///mirror_memory.db` | SQLAlchemy DB URL |
| `llm_client` | `object` | None | Custom LLM client (must expose `generate()`) |
| `llm_api_key` | `str` | None | API key for built-in OpenAI adapter |
| `llm_model` | `str` | `gpt-4o-mini` | Model identifier |
| `llm_base_url` | `str` | None | API base URL for non-OpenAI providers |

### observe()

```python
count = engine.observe(
    user_id="u1",
    session_id="s1",
    text="I love painting",
    turn_count=1,
)
```

**Returns**: `int` — number of claims extracted.

**Raises**: `ValidationError` if user_id/session_id/text is empty or text exceeds 50000 chars.

### recall()

```python
context = engine.recall(
    user_id="u1",
    query="What are the user's hobbies?",
    language="en",
    tail_load=0,
)
```

**Returns**: `str | None` — rendered memory block, or None if no memory available.

**Raises**: `ValidationError` if user_id is empty.

### panel()

```python
data = engine.panel(user_id="u1", language="en")
```

**Returns**: `PanelData` — structured panel with `memory`, `sections`, `avatar`.

### get_beliefs()

```python
beliefs = engine.get_beliefs(user_id="u1", limit=50)
```

**Returns**: `list[BeliefInfo]` — active beliefs sorted by last evidence time.

### set_enabled()

```python
engine.set_enabled(user_id="u1", enabled=False)
```

**Returns**: `None`. Raises `ValidationError` if user_id is empty.

### delete_memories()

```python
result = engine.delete_memories(user_id="u1")
```

**Returns**: `DeleteResult` — per-table deletion counts. Use `result.as_dict()` for dict form.

### forget_belief()

```python
result = engine.forget_belief(user_id="u1", belief_id="b-123")
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | `str` | User identifier |
| `belief_id` | `str` | Target belief ID to delete |

**Returns**: `DeleteResult` — deletion counts for the targeted belief and its associated events.

**Raises**: `ValidationError` if user_id or belief_id is empty.

### forget_session()

```python
result = engine.forget_session(user_id="u1", session_id="s-456")
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | `str` | User identifier |
| `session_id` | `str` | Target session ID whose summary to delete |

**Returns**: `DeleteResult` — deletion counts for the session summary.

**Raises**: `ValidationError` if user_id or session_id is empty.

### correct()

```python
result = engine.correct(
    user_id="u1",
    belief_id="b-123",
    new_claim_text="I prefer oil painting over watercolor",
    correction_note="User clarified preference",
    new_predicate="prefers",
    new_object="oil_painting",
    new_value="oil_painting",
)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | `str` | User identifier |
| `belief_id` | `str` | ID of the belief to correct |
| `new_claim_text` | `str` | Corrected claim text |
| `correction_note` | `str` | Explanation of the correction |
| `new_predicate` | `str` | Updated predicate |
| `new_object` | `str` | Updated object |
| `new_value` | `str` | Updated value |

**Returns**: `BeliefInfo` — the corrected belief.

**Raises**: `ValidationError` if user_id or belief_id is empty, or if the belief does not belong to the user.

### explain()

```python
history = engine.explain(user_id="u1", belief_id="b-123")
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | `str` | User identifier |
| `belief_id` | `str` | Target belief ID |

**Returns**: `dict` — full provenance chain including creation events, evidence items, supersession history, and correction records.

**Raises**: `ValidationError` if user_id or belief_id is empty.

### set_consent()

```python
engine.set_consent(user_id="u1", feature="memory", granted=True)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | `str` | User identifier |
| `feature` | `str` | Feature name (e.g., `memory`, `extraction`) |
| `granted` | `bool` | Whether consent is granted |

**Returns**: `None`.

**Raises**: `ValidationError` if user_id or feature is empty.

### get_verification_candidates()

```python
candidates = engine.get_verification_candidates(user_id="u1", limit=5, session_id="s1")
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_id` | `str` | — | User identifier |
| `limit` | `int` | `5` | Maximum number of candidates to return |
| `session_id` | `str` | `None` | Optional session scope |

**Returns**: `list[BeliefInfo]` — beliefs flagged as needing clarification, sorted by confidence uncertainty.

**Raises**: `ValidationError` if user_id is empty.

### run_verification()

```python
result = engine.run_verification(
    user_id="u1",
    session_id="s1",
    user_text="Yes, I still enjoy painting",
)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | `str` | User identifier |
| `session_id` | `str` | Current session ID |
| `user_text` | `str` | User's response to the verification prompt |

**Returns**: `dict` — verification outcome including updated beliefs and resolved contradictions.

**Raises**: `ValidationError` if user_id, session_id, or user_text is empty.

## OpenAILLM

```python
from mirror_memory.llm import OpenAILLM

llm = OpenAILLM(
    api_key="sk-xxx",
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
)
```

## Exceptions

| Exception | Description |
|-----------|-------------|
| `MirrorMemoryError` | Base exception |
| `ConfigError` | Invalid or missing configuration |
| `ExtractionError` | K1/K2/K3 extraction failure |
| `LLMError` | LLM client or API failure |
| `StorageError` | Database or persistence failure |
| `ValidationError` | Input validation failure |