"""M1 closeout task 1 -- narrowed write-exemption counterexamples.

The blanket test-body fixture_write_scope is gone: the write guard is now
LIVE in test bodies (exemptions: fixture setup phase, and tests that
explicitly declare @pytest.mark.fixture_writes(reason=...)).

Counterexample required by the task: an ordinary test (no marker) calling
a production function that writes a protected table DIRECTLY (i.e. outside
its classified scope) must be intercepted.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.core.models import Base
from mirror_memory.core.tenancy import (
    WriteGuardViolation,
    fixture_write_scope,
    tenant_scope,
)


def session_at(url):
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def test_unmarked_test_body_cannot_construct_belief(tmp_path):
    """Ordinary test (no marker): constructing a protected row in the test
    body is a write the guard must intercept."""
    factory = session_at(f"sqlite:///{tmp_path}/ce.db")
    from mirror_memory.core.models import Belief

    with tenant_scope("default", "default"):
        s = factory()
        with pytest.raises(WriteGuardViolation):
            s.add(Belief(user_id="u1", dimension="d", key="k", claim_text="x"))
        s.close()


def test_unmarked_body_production_direct_write_is_intercepted(tmp_path):
    """The task's counterexample: a production-code path that writes a
    protected table without entering its classified scope, invoked from an
    ordinary (unmarked) test body, must be caught.

    We simulate the production side honestly: a helper -- of the kind a
    real engine module would contain -- that writes a belief row directly
    (no engine_write_scope).  Before the narrowing this was silently
    allowed by the blanket scope; now the guard fires at attribute-set
    time inside the test body.
    """

    def production_style_direct_writer(session, user_id: str, key: str):
        # No engine_write_scope: an unsanctioned writer (what a future
        # module might accidentally do).
        from mirror_memory.core.models import Belief

        session.add(Belief(user_id=user_id, dimension="topic", key=key,
                           claim_text="direct write"))
        session.commit()

    factory = session_at(f"sqlite:///{tmp_path}/ce2.db")
    with tenant_scope("default", "default"):
        s = factory()
        with pytest.raises(WriteGuardViolation):
            production_style_direct_writer(s, "u1", "k")
        s.rollback()
        s.close()


def test_marked_test_body_can_write(tmp_path):
    """The escape hatch works: fixture_writes(reason=...) opens the scope
    for that test's body."""
    from mirror_memory.core.models import Belief

    factory = session_at(f"sqlite:///{tmp_path}/ce3.db")
    with tenant_scope("default", "default"), fixture_write_scope("declared seed"):
        s = factory()
        s.add(Belief(user_id="u1", dimension="d", key="k", claim_text="ok"))
        s.commit()
        assert s.query(Belief).count() == 1
        s.close()


def test_marker_requires_a_reason():
    """A fixture_writes marker without a reason is a usage error (the
    declaration must say WHY the body writes) -- validated against the
    conftest's reason extraction logic."""
    import tests.conftest as conftest

    class _Mark:
        def __init__(self, kwargs, args):
            self.kwargs = kwargs
            self.args = args

    class _Node:
        def __init__(self, marker):
            self._marker = marker

        def get_closest_marker(self, name):
            return self._marker if name == "fixture_writes" else None

    class _Request:
        def __init__(self, node):
            self.node = node

    # reason from kwargs
    assert conftest._marker_reason(_Request(_Node(_Mark({"reason": "why"}, ())))) == "why"
    # reason from positional args
    assert conftest._marker_reason(_Request(_Node(_Mark({}, ("positional",))))) == "positional"
    # missing reason -> UsageError
    with pytest.raises(pytest.UsageError):
        conftest._marker_reason(_Request(_Node(_Mark({}, ()))))
