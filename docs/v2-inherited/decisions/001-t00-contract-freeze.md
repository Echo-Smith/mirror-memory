# Decision Record 001: T00 Contract Freeze

Date: 2026-09-17
Status: Accepted

## Context

Mirror Memory v2 has 8 design specification documents but zero implementation.
T00 freezes the behavioral contracts into a working repository with testable interfaces.

## Decisions

### 1. Repository Layout
- Monorepo: docs/ (specifications) + src/mirror_memory/ (code) + tests/ (C01-C20, Q01-Q08)
- Python package under src/mirror_memory/ with modules: core, application, domains, adapters, runtime

### 2. Interface Names (frozen)
All 8 operations as defined in design.md §2:
- `observe(ObserveInput) -> ResultEnvelope`
- `get_operation(GetOperationInput) -> ResultEnvelope`
- `recall(RecallInput) -> ResultEnvelope`
- `correct(CorrectInput) -> ResultEnvelope`
- `forget(ForgetInput) -> ResultEnvelope`
- `explain(ExplainInput) -> ResultEnvelope`
- `export(ExportInput) -> ResultEnvelope`
- `sync_authorization(SyncAuthorizationInput) -> ResultEnvelope`

### 3. Source/Deletion Semantics
- Source role: user | assistant | system (SourceRole enum)
- assistant-role sources cannot become user facts (R04)
- Deletion uses explicit ForgetSelector; empty selector is rejected (R08)
- Deletion barrier coordinated via scope_control.deletion_generation

### 4. Test Numbering
- C01-C20: Engineering contract tests (validation.md §1)
- Q01-Q08: Business quality scenarios (validation.md §2)
- All markers registered in pyproject.toml [tool.pytest.ini_options]

### 5. Technology Stack
- Python 3.11+, FastAPI, SQLAlchemy 2.0, psycopg 3, Pydantic 2, Alembic
- PostgreSQL 16 via Docker Compose (port 5433)
- pytest + pytest-asyncio for testing
- M1: deterministic extractor, fake model client, injectable clock

### 6. Environment Safety
- MIRROR_ENV=test required for destructive operations (schema reset)
- DATABASE_URL from environment variable only, never hardcoded
- Docker Compose provides isolated test database

### 7. 首个产品试点
- First integration product; no backward compatibility with old Mirror
- app_id="my_app", isolated from future products
- Modes: off / shadow / active_internal (M2 scope)

### 8. 第二产品
- Second product, uses independent app_id
- Validates cross-product reuse (R15)
- Cannot access first product data

## Not Decided (deferred to later tasks)
- HTTP transport layer (not M1 scope)
- Embedding/vector search (optional, not M1)
- Model selection and pricing (M2 scope)
- Production deployment topology (M3 scope)

## Exit Criteria
- All 8 interface names confirmed and implemented as abstract methods
- C01-C20 test files exist with proper markers and skip annotations
- Docker Compose starts PostgreSQL successfully
- `pytest --collect-only` discovers all test cases
- Core types importable and pass basic validation
