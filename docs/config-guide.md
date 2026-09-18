# Configuration Guide

All domain-specific content is loaded from YAML files in the `config/` directory.

## Files

### dimensions.yaml

Defines the dimension schema:

```yaml
dimensions:
  - dimension_id: topic
    name: { zh: "话题", en: "Topic" }
    description: "Recurring topics the user mentions"
    render_priority: 1
```

### anchors.yaml

Keyword anchors for K1/K2 extraction:

```yaml
anchors:
  - dimension: topic
    key: work
    phrases: ["work", "job", "career", "工作", "上班"]
```

### display.yaml

User-facing labels:

```yaml
labels:
  - key: sleep
    dimension: topic
    zh: "睡眠"
    en: "Sleep"
```

### extraction.yaml

Keywords and regex patterns:

```yaml
keywords:
  topic: ["work", "family", "health"]
  preference: ["like", "love", "hate"]

patterns:
  - regex: "\\d+ years old"
    dimension: fact
    key: age

context_tags: ["work", "family", "health"]
```

### prompts/

LLM prompt templates:
- `k2_system.txt` — K2 semantic extraction
- `k3_system.txt` — K3 understanding synthesis
- `verification.txt` — Hypothesis verification
- `formulate.txt` — Worker formulate

## Custom Configuration

Create a new directory with your YAML files:

```python
engine = MemoryEngine(config_path="my_config/")
```

Missing files use empty defaults. Only provide what you need to customize.