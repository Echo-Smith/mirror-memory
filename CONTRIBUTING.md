# Contributing to mirror-memory

## Development Setup

```bash
git clone https://github.com/Echo-Smith/mirror-memory.git
cd mirror-memory
python3.12 -m venv .venv
.venv/bin/pip install -e ".[server,llm,dev]"
```

## Running Tests

```bash
.venv/bin/python -m pytest tests/ -q
# 368 passed in 3.5s
```

## Architecture

```
src/mirror_memory/
├── core/          # Generic engine (zero domain coupling)
├── extraction/    # K1/K2/K3 extraction pipeline
├── render/        # Memory block rendering + panel
├── worker/        # Async evolution worker
├── memory/        # Identity resolver + canonicalization
├── config/        # YAML config loader + schema
├── api.py         # MemoryEngine public API
├── server.py      # FastAPI server
├── llm.py         # OpenAI-compatible LLM adapter
├── models.py      # Public data models (BeliefInfo, DeleteResult, PanelData)
└── exceptions.py  # Exception hierarchy
```

## Design Principles

1. **Domain-agnostic core**: No psychology-specific code in `src/`. All domain content lives in `config/`.
2. **Fail-open**: Extraction failures never block the caller.
3. **State revision**: Every memory mutation bumps `state_revision` for stale-write detection.
4. **Test before commit**: All 368 tests must pass.

## Adding a New Dimension

1. Add to `config/dimensions.yaml`
2. Add anchors to `config/anchors.yaml`
3. Add display labels to `config/display.yaml`
4. (Optional) Add to `config/identity_policy.yaml` if it has cardinality semantics

## Code Style

- Python 3.11+
- Ruff for linting/formatting
- Type annotations on all public functions
- Docstrings on all public classes/methods (numpy style)
