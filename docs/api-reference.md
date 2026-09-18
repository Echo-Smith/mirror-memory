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