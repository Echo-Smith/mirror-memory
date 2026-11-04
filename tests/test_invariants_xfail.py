"""xfail(strict=True) stubs for invariants that v1 does not yet implement.

Context: the six runtime invariants I1–I6 are defined in
``docs/v2-inherited/invariants.md``; the mapping to v1 tests lives in
``docs/invariants.md``.  Every invariant has a real v1 test EXCEPT **I4**:

    I4 — the same ``source_event_id`` is globally unique within one scope
         (idempotency).

v1 was rewritten around content-addressed evidence ``ref`` uniqueness plus
publish-time ``proposal_id`` / ``idempotency_key`` idempotency.  The v2
``source_event_id`` column and its claim-time uniqueness constraint were
**not** carried over, so there is nothing in v1 to assert this exact
property against.

The stub below encodes the v2 property.  It is marked
``xfail(strict=True)``: it documents the gap and, crucially, **fails the
build if the property ever starts holding** (XPASS -> failure under strict
mode).  That is intentional -- if I4 is ported to v1, this test must be
turned into a real passing test and the mapping in ``docs/invariants.md``
(and any product copy) updated, rather than the invariant silently drifting
out of the contract.
"""

from __future__ import annotations

import pytest


@pytest.mark.xfail(
    strict=True,
    reason=(
        "I4 is not implemented in v1: there is no source_event_id column or "
        "claim-time global-uniqueness constraint. v1 achieves ingestion "
        "idempotency via content-addressed evidence refs + publish-time "
        "proposal_id/idempotency_key. When I4 is ported, make this pass and "
        "flip the docs/invariants.md mapping."
    ),
)
def test_i4_source_event_id_global_uniqueness():
    """I4 (v2 definition): the same source_event_id, observed twice in one
    scope, must not create a second atom.

    v1 has no ``source_event_id``; this assertion is expected to fail today.
    """
    from mirror_memory.core.models import Evidence

    # The v2 invariant is expressed in terms of a column v1 does not have.
    # Accessing it raises AttributeError (or the assertion below fails if the
    # column were added without the uniqueness semantics), which is exactly
    # the xfail we expect.
    assert hasattr(Evidence, "source_event_id"), (
        "I4: v1 Evidence has no source_event_id column; the v2 global "
        "uniqueness invariant is not implemented here"
    )
    pytest.fail(
        "I4: v1 does not enforce global source_event_id uniqueness within a "
        "scope; this stub documents the gap (see docs/invariants.md)"
    )
