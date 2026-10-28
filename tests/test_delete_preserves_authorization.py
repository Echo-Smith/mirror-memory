"""M3 batch-1 patch task 2: deleting memory must not delete the switches
or the authorizations.

A user who turns extraction off (or memory off entirely) and then has
their memory deleted must stay in that state: the feature consent rows,
the global memory preference, and the purpose grants are authorization /
configuration, NOT memory content.  Before this patch the cascade deleted
``mm_consent_grants``, so a disabled feature silently re-enabled itself on
the next request after a delete (the fallback "missing feature row =
global preference" kicked in).  "Delete AND revoke" is an explicit
two-step for the caller.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core.authorization import PurposeDenied, grant_purpose
from mirror_memory.core.models import Base, Belief, ConsentGrant, MemoryPreference, PurposeGrant
from mirror_memory.core.repository import delete_user_memories, set_memory_enabled
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope

PURPOSE = "api"


@pytest.fixture
def engine(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/delete-switches.db")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


def _seed(engine):
    F = sessionmaker(bind=engine)
    with tenant_scope("default", "default"), fixture_write_scope("seed a belief"):
        s = F()
        set_memory_enabled(s, "u1", True)
        s.add(Belief(user_id="u1", dimension="topic", key="k",
                     claim_text="remember this", confidence=0.9))
        s.commit()
        s.close()
    return MemoryEngine(
        config=load_config("config/"), database_url=str(engine.url),
        tenant_id="default", app_id="default",
    )


def _grant_via_engine(engine, user_id="u1", purpose=PURPOSE):
    F = sessionmaker(bind=engine)
    with tenant_scope("default", "default"):
        s = F()
        grant_purpose(s, user_id, purpose,
                      operations=["observe", "recall", "forget"], issuer="reviewer")
        s.commit()
        s.close()


def test_feature_consent_survives_delete(engine):
    """extraction=False -> delete -> observe must STILL be refused
    (feature_disabled): the consent row is authorization, not memory."""
    eng = _seed(engine)
    eng.set_consent(user_id="u1", feature="extraction", granted=False)
    eng.grant_purpose(user_id="u1", purpose=PURPOSE,
                      operations=["observe", "recall", "forget"], issuer="reviewer")
    result = eng.delete_memories(user_id="u1", purpose=PURPOSE)
    assert result.beliefs == 1  # memory content is gone
    # the switch survived the delete
    s = sessionmaker(bind=engine)()
    row = s.get(ConsentGrant, ("default", "default", "u1", "extraction"))
    assert row is not None and row.granted is False
    s.close()
    # ... so observe is still refused, with the feature reason
    with pytest.raises(PurposeDenied, match="feature_disabled"):
        eng.observe(user_id="u1", session_id="post-delete",
                    text="must not extract", purpose=PURPOSE)


def test_memory_disabled_survives_delete(engine):
    eng = _seed(engine)
    eng.set_enabled(user_id="u1", enabled=False)
    eng.grant_purpose(user_id="u1", purpose=PURPOSE,
                      operations=["observe", "recall", "forget"], issuer="reviewer")
    eng.delete_memories(user_id="u1", purpose=PURPOSE)
    # the global preference survived the delete
    s = sessionmaker(bind=engine)()
    row = s.get(MemoryPreference, ("default", "default", "u1"))
    assert row is not None and row.enabled is False
    s.close()
    with pytest.raises(PurposeDenied, match="feature_disabled"):
        eng.observe(user_id="u1", session_id="post-delete",
                    text="must not extract", purpose=PURPOSE)


def test_purpose_grants_survive_delete(engine):
    """Purpose grants (and their audit trail) are authorization: the delete
    cascade must leave them untouched, so re-enabled users keep their
    grant state exactly as it was."""
    eng = _seed(engine)
    eng.grant_purpose(user_id="u1", purpose=PURPOSE,
                      operations=["observe", "recall", "forget"], issuer="reviewer")
    eng.grant_purpose(user_id="u1", purpose="other", operations=["recall"], issuer="reviewer")
    result = eng.delete_memories(user_id="u1", purpose=PURPOSE)
    assert result.beliefs == 1
    s = sessionmaker(bind=engine)()
    grants = s.query(PurposeGrant).filter_by(user_id="u1").all()
    assert {(g.purpose, g.revoked_at is not None) for g in grants} == {
        (PURPOSE, False), ("other", False),
    }
    s.close()
    # the surviving grant still authorizes a post-delete recall (empty, not denied)
    ctx = eng.recall(user_id="u1", purpose=PURPOSE)
    assert ctx is None or isinstance(ctx, str)


def test_delete_counts_no_longer_include_consent_grants(engine):
    """The per-table deletion counts reflect what the cascade deletes;
    authorization tables are not in it."""
    eng = _seed(engine)
    eng.set_consent(user_id="u1", feature="extraction", granted=True)
    eng.grant_purpose(user_id="u1", purpose=PURPOSE,
                      operations=["forget"], issuer="reviewer")
    result = eng.delete_memories(user_id="u1", purpose=PURPOSE)
    as_dict = result.as_dict()
    assert "consent_grants" not in as_dict
    assert as_dict["beliefs"] == 1
