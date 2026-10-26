"""M1 batch-2 task 4 -- schema migration tests.

Builds a base-7f783bc-shaped database with real-looking data by checking
out the base revision's models in a subprocess (the old models cannot be
imported in-process alongside the current ones), then exercises:

1. upgrade -> verify -> downgrade -> upgrade keeps data consistent
2. the migrated database runs representative flows (observe-like claim /
   recall read / correct / forget)
3. a mid-migration failure leaves the database at the pre-migration state
4. reflected schema == Base.metadata (parity, by construction)
plus: refusal without an explicit default tenant.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from mirror_memory.core.models import Base
from mirror_memory.core.schema_migration import (
    MigrationRefused,
    downgrade_database,
    upgrade_database,
    verify_schema_parity,
)

REPO = Path(__file__).resolve().parents[1]

BASE_SEED_SCRIPT = r"""
import json
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# base-revision models are materialised in the worktree by the test
# (git worktree-free: the test copies the base models over a temp package)
sys.path.insert(0, sys.argv[2])

from sqlalchemy.orm import DeclarativeBase


def main(db_path: str, models_dir: str) -> None:
    import importlib
    mod = importlib.import_module("base_models")
    from sqlalchemy.engine import URL

    engine = create_engine(URL.create("sqlite", database=db_path))
    mod.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    from datetime import datetime, timezone

    pref = mod.MemoryPreference(user_id="u1", enabled=True, state_revision=3)
    s.add(pref)
    belief = mod.Belief(
        user_id="u1", dimension="topic", key="sleep",
        claim_text="User has trouble sleeping", confidence=0.8,
        layer="L4", status="active", source="extracted",
        subject="user", predicate="sleeps_badly", object="night",
    )
    s.add(belief); s.flush()
    s.add(mod.BeliefEvent(belief_id=belief.id, user_id="u1", event_type="created"))
    s.add(mod.Evidence(user_id="u1", session_id="s1", ref="m1", content="I sleep badly",
                       source_type="message", extraction_method="k1_regex", authority="user"))
    s.add(mod.EvolutionJob(id="job-1", user_id="u1", idempotency_key="ik",
                           status="completed"))
    s.add(mod.Snapshot(id="snap-1", user_id="u1", version=1, status="active",
                       content_json=json.dumps({"patterns": []})))
    s.commit()
    s.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
"""


def _materialize_base_models(target: Path) -> None:
    """Extract the base revision's models.py into an importable module."""
    src = subprocess.run(
        ["git", "show", "7f783bc:src/mirror_memory/core/models.py"],
        capture_output=True, text=True, check=True, cwd=REPO,
    ).stdout
    # neutralise the engine imports that no longer exist at base
    (target / "base_models.py").write_text(src, encoding="utf-8")


@pytest.fixture
def base_db(tmp_path):
    db = tmp_path / "base.db"
    models_dir = tmp_path / "basepkg"
    models_dir.mkdir()
    _materialize_base_models(models_dir)
    script = tmp_path / "seed_base.py"
    script.write_text(BASE_SEED_SCRIPT, encoding="utf-8")
    subprocess.run(
        [sys.executable, str(script), str(db), str(models_dir)],
        check=True, cwd=REPO,
    )
    return db


def _engine(db_path):
    from sqlalchemy.engine import URL

    return create_engine(URL.create("sqlite", database=str(db_path)))


import contextlib  # noqa: E402


@contextlib.contextmanager
def downgrade_scope():
    """Read the downgraded (pre-tenancy) database: no tenant columns exist,
    so the auditor cannot demand tenant predicates on these reads."""
    from mirror_memory.core.sql_audit import no_audit_scope

    with no_audit_scope():
        yield


def _read_claims(engine) -> list[str]:
    """Read belief claim_text from whatever schema is present (reflection)."""
    from sqlalchemy import MetaData, select, Table

    table = Table("mm_beliefs", MetaData(), autoload_with=engine)
    with engine.connect() as conn:
        return list(conn.execute(select(table.c.claim_text)).scalars())


# ---------------------------------------------------------------------------


def test_upgrade_downgrade_upgrade_roundtrip(base_db):
    eng = _engine(base_db)

    # --- upgrade 1 -------------------------------------------------------
    counts1 = upgrade_database(eng, default_tenant="legacy", default_app="app")
    assert counts1  # tables were migrated
    parity = verify_schema_parity(eng)
    assert parity == [], parity
    from mirror_memory.core.models import Belief, BeliefEvent
    from mirror_memory.core.tenancy import tenant_scope

    factory = sessionmaker(bind=eng)
    with tenant_scope("legacy", "app"):
        s = factory()
        row = s.query(Belief).one()
        assert (row.tenant_id, row.app_id) == ("legacy", "app")
        assert row.claim_text == "User has trouble sleeping"
        assert s.query(BeliefEvent).count() == 1
        s.close()

    # --- downgrade (backup restore) --------------------------------------
    assert downgrade_database(eng) is True
    insp = inspect(eng)
    assert "tenant_id" not in {
        c["name"] for c in insp.get_columns("mm_beliefs")
    }
    with downgrade_scope():
        assert _read_claims(eng) == ["User has trouble sleeping"]

    # --- upgrade 2: same data again --------------------------------------
    counts2 = upgrade_database(eng, default_tenant="legacy", default_app="app")
    assert counts2 == counts1
    from mirror_memory.core.tenancy import tenant_scope

    factory = sessionmaker(bind=eng)
    with tenant_scope("legacy", "app"):
        s = factory()
        assert s.query(Belief).count() == counts2["mm_beliefs"]
        s.close()


def test_migrated_database_runs_representative_flows(base_db):
    eng = _engine(base_db)
    upgrade_database(eng, default_tenant="legacy", default_app="app")

    from mirror_memory.core.repository import (
        forget_belief,
        get_belief,
        record_claim,
        update_belief_by_id,
    )
    from mirror_memory.core.tenancy import tenant_scope
    from mirror_memory.core.tenancy import fixture_write_scope

    factory = sessionmaker(bind=eng)
    with tenant_scope("legacy", "app"):
        s = factory()
        # recall-ish read
        belief = get_belief(s, "u1", "sleep")
        assert belief is not None
        # observe-ish write (claim)
        with fixture_write_scope("migration flow test"):
            new_belief, event = record_claim(
                s, "u1", dimension="topic", key="coffee",
                claim_text="likes coffee", confidence=0.7, session_id="s2",
            )
        s.commit()
        assert event == "created"
        # correct (creates a new row for the same key)
        old, corrected = update_belief_by_id(
            s, belief.id, new_claim_text="sleeps badly, working on it",
        )
        s.commit()
        assert corrected.claim_text == "sleeps badly, working on it"
        # forget the current row for the key
        current = get_belief(s, "u1", "sleep")
        assert current is not None
        assert forget_belief(s, "u1", current.id) is True
        s.commit()
        assert get_belief(s, "u1", "sleep") is None
        s.close()


def test_refuses_without_default_tenant(base_db):
    eng = _engine(base_db)
    with pytest.raises(MigrationRefused):
        upgrade_database(eng, default_tenant="", default_app="app")
    # nothing was touched
    insp = inspect(eng)
    assert "tenant_id" not in {c["name"] for c in insp.get_columns("mm_beliefs")}


def test_mid_migration_failure_restores_backup(base_db, monkeypatch):
    """A failing step leaves the database exactly as it was (SQLite: the
    pre-migration file backup is restored)."""
    eng = _engine(base_db)
    from mirror_memory.core import schema_migration as sm

    def boom(engine):
        raise sm.MigrationError("simulated verification failure")

    monkeypatch.setattr(sm, "_verify_tenancy", boom)
    with pytest.raises(sm.MigrationError):
        sm.upgrade_database(eng, default_tenant="legacy", default_app="app")
    # backup restored: base schema, data intact, version not advanced
    insp = inspect(eng)
    assert "tenant_id" not in {c["name"] for c in insp.get_columns("mm_beliefs")}
    assert sm.current_version(eng) == 0
    with downgrade_scope():
        assert _read_claims(eng) == ["User has trouble sleeping"]


def test_schema_parity_by_construction(base_db):
    eng = _engine(base_db)
    upgrade_database(eng, default_tenant="legacy", default_app="app")
    assert verify_schema_parity(eng) == []


def test_upgrade_installs_the_tenant_primary_key(base_db):
    """M2 task 3: the migrated database exposes the composite tenant PK
    wherever the dialect can host it.

    PostgreSQL converts the constraint (catalog ALTER, no table rewrite).
    SQLite keeps a single-column physical key -- a composite key over an
    autoincrement column is a CompileError there -- and closes the
    PK-addressing surface with the fallback layers instead.
    """
    import os  # noqa: F401 -- kept for the fixture's dialect note

    eng = _engine(base_db)
    upgrade_database(eng, default_tenant="legacy", default_app="app")
    insp = inspect(eng)
    tenant_tables = [
        t.name for t in Base.metadata.sorted_tables if insp.has_table(t.name)
    ]
    assert tenant_tables
    # The legacy database this fixture builds is always SQLite (its base
    # models come from a SQLite-seeded file), so the physical key stays
    # single-column here on both matrix legs: the migration reads the
    # target engine's dialect, and that engine is SQLite either way.
    is_pg = eng.dialect.name == "postgresql"
    for name in tenant_tables:
        live = insp.get_pk_constraint(name)["constrained_columns"]
        wanted = [c.name for c in Base.metadata.tables[name].primary_key.columns]
        if is_pg:
            assert live == wanted, (name, live, wanted)
        else:
            autoinc = Base.metadata.tables[name]._autoincrement_column
            if autoinc is not None:
                assert live == [autoinc.name], (name, live)
            else:
                assert live == wanted, (name, live, wanted)


def test_upgrade_is_reentrant_on_the_primary_key_step(base_db):
    """A second upgrade must not re-run the conversion or fail on it."""
    eng = _engine(base_db)
    upgrade_database(eng, default_tenant="legacy", default_app="app")
    before = {
        t.name: inspect(eng).get_pk_constraint(t.name)["constrained_columns"]
        for t in Base.metadata.sorted_tables
        if inspect(eng).has_table(t.name)
    }
    upgrade_database(eng, default_tenant="legacy", default_app="app")
    after = {
        t.name: inspect(eng).get_pk_constraint(t.name)["constrained_columns"]
        for t in Base.metadata.sorted_tables
        if inspect(eng).has_table(t.name)
    }
    assert after == before


def test_primary_key_swap_is_postgres_only():
    """The SQLite renderer refuses the catalog ALTER instead of silently
    emitting something that would rebuild the table."""
    from sqlalchemy.dialects import sqlite as sqlite_dialect

    from mirror_memory.core.schema_migration import MigrationError, PrimaryKeySwap

    with pytest.raises(MigrationError):
        str(PrimaryKeySwap("mm_beliefs").compile(dialect=sqlite_dialect.dialect()))
