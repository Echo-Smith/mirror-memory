"""Dependency compatibility matrix (M5 readiness, Task 3).

This test asserts that ``mirror_memory`` imports and a *basic* engine
smoke-run works under the declared deployment floors:

    Python      >= 3.12
    SQLAlchemy  >= 2.0.49
    FastAPI     >= 0.135.3
    Pydantic    2.x

Design: the floors are a **lower bound**, so a CI leg that pins exactly these
versions exercises them.  When the *installed* versions are below a floor,
the relevant assertion is ``pytest.skip``-ped (with a clear message) instead
of failing -- an informative signal for a leg that has not yet pinned the
floors, not a red build for an unrelated reason.  ``bench-ci`` wires a CI leg
that pins the floors so this actually runs there.

The smoke-run uses an in-memory SQLite database and a stub LLM client, so it
needs no network and no PostgreSQL -- it only proves the import graph, the
schema bootstrap, and a minimal ``observe`` path survive on the installed
dependency set.
"""

from __future__ import annotations

import sys

import pytest


# -- floor constants --------------------------------------------------------

PY_FLOOR = (3, 12)
SQLALCHEMY_FLOOR = (2, 0, 49)
FASTAPI_FLOOR = (0, 135, 3)
PYDANTIC_MAJOR = 2


def _parse_version(text: str) -> tuple[int, ...]:
    """Parse the leading numeric components of a version string.

    Tolerates pre-release / local suffixes (``2.1.5rc1``, ``0.135.3.dev0``)
    by keeping only the leading ``[0-9.]`` run."""
    parts: list[int] = []
    current = ""
    for ch in text.strip():
        if ch.isdigit():
            current += ch
        elif ch == "." and current:
            parts.append(int(current))
            current = ""
        elif current:
            break
    if current:
        parts.append(int(current))
    return tuple(parts)


def _at_least(installed: tuple[int, ...], floor: tuple[int, ...]) -> bool:
    width = max(len(installed), len(floor))
    a = installed + (0,) * (width - len(installed))
    b = floor + (0,) * (width - len(floor))
    return a >= b


# -- the tests --------------------------------------------------------------


def test_python_version_floor():
    """Python >= 3.12."""
    installed = sys.version_info[:2]
    if installed < PY_FLOOR:
        pytest.skip(
            f"Python {installed[0]}.{installed[1]} is below the tested floor "
            f"{PY_FLOOR[0]}.{PY_FLOOR[1]}; the matrix leg pins 3.12"
        )
    assert installed >= PY_FLOOR


def test_sqlalchemy_floor():
    """SQLAlchemy >= 2.0.49."""
    import sqlalchemy

    installed = _parse_version(sqlalchemy.__version__)
    if not _at_least(installed, SQLALCHEMY_FLOOR):
        pytest.skip(
            f"sqlalchemy {sqlalchemy.__version__} is below the floor "
            f"{'.'.join(map(str, SQLALCHEMY_FLOOR))}; pin it in the matrix leg"
        )
    assert _at_least(installed, SQLALCHEMY_FLOOR)


def test_fastapi_floor():
    """FastAPI >= 0.135.3 (installed as part of the server in this leg)."""
    fastapi = pytest.importorskip(
        "fastapi",
        reason="fastapi is an optional server extra; the matrix leg installs it",
    )
    installed = _parse_version(fastapi.__version__)
    if not _at_least(installed, FASTAPI_FLOOR):
        pytest.skip(
            f"fastapi {fastapi.__version__} is below the floor "
            f"{'.'.join(map(str, FASTAPI_FLOOR))}; pin it in the matrix leg"
        )
    assert _at_least(installed, FASTAPI_FLOOR)


def test_pydantic_2():
    """Pydantic 2.x."""
    pydantic = pytest.importorskip("pydantic")
    major = _parse_version(pydantic.__version__)[0]
    if major != PYDANTIC_MAJOR:
        pytest.skip(
            f"pydantic {pydantic.__version__} is not 2.x; the matrix leg pins 2.x"
        )
    assert major == PYDANTIC_MAJOR


def test_mirror_imports_and_smoke_runs():
    """The package imports and a basic engine observe path works end-to-end.

    Proves the import graph (api, config, core, storage) resolves and that
    schema bootstrap + a minimal ``observe`` survive on the installed
    dependency set.  Uses in-memory SQLite and a stub LLM -- no network, no
    PostgreSQL.
    """
    from mirror_memory import MemoryEngine
    from mirror_memory.config.loader import load_config

    config = load_config("config/")

    class _StubLLM:
        """Minimal extractor: returns an empty claim list, so observe is a
        no-network no-op that still exercises the full write path."""

        def generate(self, *, system_prompt, payload_text, fallback=None):
            return '{"claims": []}'

    engine = MemoryEngine(
        config=config,
        database_url="sqlite://",
        llm_client=_StubLLM(),
        # a purpose default keeps the smoke independent of grant seeding;
        # the smoke is about dependency wiring, not authorization.
        default_purpose="smoke",
    )

    # Schema bootstrap is lazy; the first entry builds it.
    claims = engine.observe(
        user_id="smoke-user",
        session_id="smoke-session",
        text="I enjoy hiking on weekends.",
        purpose="smoke",
    )
    assert isinstance(claims, int)
    assert claims >= 0
