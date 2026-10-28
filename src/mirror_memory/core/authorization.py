"""Purpose authorization (M3 batch 1): the use-limitation layer.

A tenant authorizes a *purpose* (why a caller touches memory) to a fixed
set of *operations*.  Every entry point that reads or writes memory calls
:func:`authorize` in the same transaction, before any business statement;
a refusal raises :class:`PurposeDenied` and nothing else runs.

Fail-closed by design, explicitly unlike v2: a missing purpose is a
refusal (``missing_purpose``), never an implicit allow.  ``authorize``
re-reads the grant row on every call -- nothing is cached beyond the
transaction -- so a committed revocation bites the very next request.

The function refuses to run under ``system_scope`` or without a tenant
context: on the maintenance engine the loader criteria are absent, the
query would see every tenant's grants, and a foreign grant could satisfy
the check (the "RLS makes the check silently empty" shape).  Call sites
run inside a tenant-scoped session on the tenant (app) engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

from mirror_memory.core.repository import _sanctioned_writer
from mirror_memory.core.tenancy import current_tenant, in_system_scope

#: The fixed operation vocabulary.  Unknown operations are a programming
#: error and raise instead of being refused data-driven.
OPERATIONS: tuple[str, ...] = (
    "observe",
    "recall",
    "correct",
    "forget",
    "export",
    "explain",
    "extract",
)

#: Which feature consent (``is_feature_enabled``) each operation inherits.
#: ``observe`` and ``extract`` are both gated by the extraction feature;
#: the rest use their own name, so operators can switch them individually
#: with ``set_consent`` (a missing feature grant falls back to the global
#: memory preference, the pre-M3 behaviour).
OPERATION_FEATURE: dict[str, str] = {
    "observe": "extraction",
    "extract": "extraction",
    "recall": "recall",
    "correct": "correct",
    "forget": "forget",
    "export": "export",
    "explain": "explain",
}

#: Refusal reasons, in the order ``authorize`` evaluates them.
REASON_MISSING_PURPOSE = "missing_purpose"
REASON_NO_GRANT = "no_grant"
REASON_REVOKED = "revoked"
REASON_EXPIRED = "expired"
REASON_OPERATION_NOT_ALLOWED = "operation_not_allowed"
REASON_FEATURE_DISABLED = "feature_disabled"


class UnknownOperation(ValueError):
    """An operation outside the fixed vocabulary was requested."""


class StaleGrantVersion(RuntimeError):
    """A grant change lost the compare-and-swap: the version moved on."""


class PurposeDenied(PermissionError):
    """An operation was refused by the purpose authorization layer."""

    def __init__(self, decision: "Decision", operation: str) -> None:
        self.decision = decision
        self.operation = operation
        super().__init__(
            f"{operation} denied for purpose {decision.purpose!r}: "
            f"{decision.reason} (grant version {decision.grant_version})"
        )


@dataclass(frozen=True)
class Decision:
    """The outcome of one authorization check."""

    allowed: bool
    reason: str
    grant_version: int | None
    purpose: str = ""

    def require(self, operation: str) -> "Decision":
        """Raise :class:`PurposeDenied` unless this decision allows."""
        if not self.allowed:
            raise PurposeDenied(self, operation)
        return self


def _now() -> datetime:
    """Naive UTC -- the engine stores naive datetimes (TIMESTAMP WITHOUT
    TIME ZONE); comparing an aware value against the stored one would
    raise on every expiry check."""
    return datetime.now(UTC).replace(tzinfo=None)


def _operations_json(operations) -> str:
    ops = list(operations)
    if not ops:
        raise UnknownOperation("a grant must allow at least one operation")
    unknown = [op for op in ops if op not in OPERATIONS]
    if unknown:
        raise UnknownOperation(
            f"unknown operation(s) {unknown}; the vocabulary is {list(OPERATIONS)}"
        )
    if len(set(ops)) != len(ops):
        raise UnknownOperation("duplicate operation(s) in the grant")
    return json.dumps(sorted(ops))


def _load_grant(session, user_id: str, purpose: str, *, lock: bool = False):
    """The grant row, re-read from the database on EVERY call.

    ``session.get`` is deliberately avoided: its identity map can serve an
    object loaded earlier in this session's lifetime, which would be a
    cache across transactions -- a committed revocation must bite the next
    request.  A fresh ``query`` always reaches the database.  With
    ``lock`` (PostgreSQL only) the row is selected ``FOR SHARE``: a
    concurrent revocation's UPDATE blocks until this transaction ends, so
    "revoke committed" can never slip between this check and this
    transaction's commit (M3 batch-1 task 4).  SQLite serializes writers
    by itself (rollback-journal EXCLUSIVE at commit); a ``FOR SHARE`` there
    is a syntax error, so it is not emitted.
    """
    from mirror_memory.core.models import PurposeGrant

    query = session.query(PurposeGrant).filter(
        PurposeGrant.user_id == user_id,
        PurposeGrant.purpose == purpose,
    )
    if lock and session.get_bind().dialect.name == "postgresql":
        # SQLAlchemy 2.x: read=True renders FOR SHARE (the 1.x-era
        # with_for_share name no longer exists on Query)
        query = query.with_for_update(read=True)
    return query.first()


def authorize(
    session,
    user_id: str,
    purpose: str | None,
    operation: str,
    *,
    lock: bool = False,
) -> Decision:
    """May *purpose* perform *operation* on this user's memory?

    Evaluation order (each refusal has its own reason): missing purpose,
    unknown grant, revoked, expired, operation not in the grant's list,
    then the existing feature consent (``feature_disabled``).  See the
    module docstring for the fail-closed and no-caching contracts.
    """
    if operation not in OPERATIONS:
        raise UnknownOperation(
            f"unknown operation {operation!r}; the vocabulary is {list(OPERATIONS)}"
        )
    if in_system_scope():
        raise RuntimeError(
            "authorize() must not run under system_scope: on the maintenance "
            "engine the tenant filter is absent and the check could be "
            "satisfied by another tenant's grant"
        )
    current_tenant()  # fail closed without a tenant context
    if not purpose or not str(purpose).strip():
        return Decision(False, REASON_MISSING_PURPOSE, None, "")
    purpose = str(purpose).strip()

    grant = _load_grant(session, user_id, purpose, lock=lock)
    if grant is None:
        # No implicit issuance, ever: an absent grant is a refusal.  Hosts
        # (and the test harness) grant explicitly through grant_purpose.
        return Decision(False, REASON_NO_GRANT, None, purpose)
    if grant.revoked_at is not None:
        return Decision(False, REASON_REVOKED, grant.version, purpose)
    if grant.expires_at is not None and grant.expires_at <= _now():
        return Decision(False, REASON_EXPIRED, grant.version, purpose)
    if operation not in json.loads(grant.allowed_operations):
        return Decision(False, REASON_OPERATION_NOT_ALLOWED, grant.version, purpose)
    from mirror_memory.core.repository import is_feature_enabled

    if not is_feature_enabled(session, user_id, OPERATION_FEATURE[operation]):
        return Decision(False, REASON_FEATURE_DISABLED, grant.version, purpose)
    return Decision(True, "granted", grant.version, purpose)


def authorize_or_raise(session, user_id: str, purpose: str | None, operation: str, *, lock: bool = False) -> Decision:
    """:func:`authorize`, raising :class:`PurposeDenied` on refusal."""
    return authorize(session, user_id, purpose, operation, lock=lock).require(operation)


@_sanctioned_writer("consent")
def grant_purpose(
    session,
    user_id: str,
    purpose: str,
    *,
    operations: list[str],
    issuer: str,
    expires_at: datetime | None = None,
    expected_version: int | None = None,
    reason: str = "",
):
    """Issue or update one purpose grant.

    First issuance (no row): ``expected_version`` must be ``None``; two
    concurrent first issuances collide on the composite primary key and
    the loser raises :class:`StaleGrantVersion` -- exactly one wins.
    Update (row exists): ``expected_version`` is mandatory and compared
    atomically (``UPDATE ... WHERE version = expected``), so a stale
    issuer is refused instead of clobbering.  Every accepted change
    appends a :class:`~mirror_memory.core.models.PurposeGrantEvent` row.

    Returns the grant row.  The caller commits.
    """
    from sqlalchemy import update as sa_update

    from mirror_memory.core.models import PurposeGrant, PurposeGrantEvent

    if not purpose or not str(purpose).strip():
        raise ValueError("grant_purpose requires a non-empty purpose")
    purpose = str(purpose).strip()
    if not user_id or not str(user_id).strip():
        raise ValueError("grant_purpose requires a non-empty user_id")
    operations_json = _operations_json(operations)
    if not issuer or not str(issuer).strip():
        raise ValueError("grant_purpose requires a non-empty issuer")

    existing = (
        session.query(PurposeGrant)
        .filter(PurposeGrant.user_id == user_id, PurposeGrant.purpose == purpose)
        .first()
    )
    if existing is None:
        if expected_version is not None:
            raise StaleGrantVersion(
                f"grant ({user_id!r}, {purpose!r}) does not exist; "
                "first issuance takes no expected_version"
            )
        grant = PurposeGrant(
            user_id=user_id,
            purpose=purpose,
            version=1,
            allowed_operations=operations_json,
            issuer=issuer,
            expires_at=expires_at,
        )
        session.add(grant)
        try:
            session.flush()  # surfaces the composite-PK collision...
            session.add(PurposeGrantEvent(
                user_id=user_id, purpose=purpose, action="granted",
                old_version=None, new_version=1, actor=issuer, reason=reason[:200],
            ))
            session.flush()
        except Exception:
            session.rollback()
            raise StaleGrantVersion(
                f"grant ({user_id!r}, {purpose!r}) was issued concurrently "
                "-- exactly one first issuance wins"
            ) from None
        return grant

    if expected_version is None:
        raise StaleGrantVersion(
            f"grant ({user_id!r}, {purpose!r}) exists at version "
            f"{existing.version}; pass expected_version to change it "
            "(the optimistic lock is what keeps concurrent issuers "
            "from clobbering each other)"
        )
    # flush first so the identity-map state does not mask a concurrent
    # commit inside the CAS UPDATE
    session.flush()
    result = session.execute(
        sa_update(PurposeGrant)
        .where(
            PurposeGrant.user_id == user_id,
            PurposeGrant.purpose == purpose,
            PurposeGrant.version == expected_version,
        )
        .values(
            version=PurposeGrant.version + 1,
            allowed_operations=operations_json,
            issuer=issuer,
            expires_at=expires_at,
            revoked_at=None,
        )
        .returning(PurposeGrant.version)
    )
    new_version = result.scalar_one_or_none()
    session.flush()
    if new_version is None:
        raise StaleGrantVersion(
            f"grant ({user_id!r}, {purpose!r}) no longer holds version "
            f"{expected_version}: a concurrent issuer moved it"
        )
    session.add(PurposeGrantEvent(
        user_id=user_id, purpose=purpose, action="updated",
        old_version=expected_version, new_version=int(new_version),
        actor=issuer, reason=reason[:200],
    ))
    session.flush()
    return session.query(PurposeGrant).filter(
        PurposeGrant.user_id == user_id, PurposeGrant.purpose == purpose
    ).first()


@_sanctioned_writer("consent")
def revoke_purpose(
    session,
    user_id: str,
    purpose: str,
    *,
    actor: str,
    reason: str = "",
    expected_version: int | None = None,
):
    """Revoke one purpose grant (immediate: the next ``authorize`` re-reads
    the row).  Idempotent -- revoking an already-revoked or non-existent
    grant changes nothing and appends no event.  The version is bumped so
    a worker holding an older version detects the change."""
    from sqlalchemy import update as sa_update

    from mirror_memory.core.models import PurposeGrant, PurposeGrantEvent

    if not actor or not str(actor).strip():
        raise ValueError("revoke_purpose requires a non-empty actor")
    if not purpose or not str(purpose).strip():
        raise ValueError("revoke_purpose requires a non-empty purpose")
    purpose = str(purpose).strip()

    existing = (
        session.query(PurposeGrant)
        .filter(PurposeGrant.user_id == user_id, PurposeGrant.purpose == purpose)
        .first()
    )
    if existing is None or existing.revoked_at is not None:
        return existing
    version = existing.version
    if expected_version is not None and expected_version != version:
        raise StaleGrantVersion(
            f"grant ({user_id!r}, {purpose!r}) holds version {version}, "
            f"not the expected {expected_version}"
        )
    result = session.execute(
        sa_update(PurposeGrant)
        .where(
            PurposeGrant.user_id == user_id,
            PurposeGrant.purpose == purpose,
            PurposeGrant.version == version,
        )
        .values(revoked_at=_now(), version=version + 1)
        .returning(PurposeGrant.version)
    )
    new_version = result.scalar_one_or_none()
    session.flush()
    if new_version is None:
        raise StaleGrantVersion(
            f"grant ({user_id!r}, {purpose!r}) moved during the revocation"
        )
    session.add(PurposeGrantEvent(
        user_id=user_id, purpose=purpose, action="revoked",
        old_version=version, new_version=int(new_version),
        actor=actor, reason=reason[:200],
    ))
    session.flush()
    return session.query(PurposeGrant).filter(
        PurposeGrant.user_id == user_id, PurposeGrant.purpose == purpose
    ).first()


def ensure_purpose_grant(session, user_id: str, purpose: str, *, issuer: str, operations: list[str] | None = None):
    """Idempotent issuance for hosts that run their own jobs against
    themselves (the benchmark harness): grant when missing, return the
    existing row otherwise."""
    from mirror_memory.core.models import PurposeGrant

    existing = (
        session.query(PurposeGrant)
        .filter(PurposeGrant.user_id == user_id, PurposeGrant.purpose == purpose)
        .first()
    )
    if existing is not None:
        return existing
    return grant_purpose(
        session, user_id, purpose,
        operations=list(operations or OPERATIONS), issuer=issuer,
        reason="ensure_purpose_grant (idempotent host-side issuance)",
    )


@_sanctioned_writer("consent")
def record_purpose_invocation(
    session,
    user_id: str,
    purpose: str,
    *,
    operation: str,
    decision: str,
    actor: str,
    reason: str = "",
) -> None:
    """Append an authorization-audit row for an operation that ran WITHOUT
    a grant decision -- the forget path (M3 batch-1 task 3): deletion is
    allowed regardless of authorization state, but every invocation is
    recorded with its purpose and the decision that admitted it."""
    from mirror_memory.core.models import PurposeGrantEvent

    if operation not in OPERATIONS:
        raise UnknownOperation(operation)
    session.add(PurposeGrantEvent(
        user_id=user_id,
        purpose=str(purpose or "").strip(),
        action="invoked",
        operation=operation,
        decision=decision,
        old_version=None,
        new_version=None,
        actor=actor,
        reason=reason[:200],
    ))
    session.flush()


# ---------------------------------------------------------------------------
# Entry-point inventory (M3 batch-1 task 3)
#
# Every public entry that can read or write memory must be registered here
# with the operation it exercises and where its purpose comes from.  The
# test suite discovers the actual surface (MemoryEngine public methods,
# server routes) and fails when an entry is missing from this registry --
# the same mechanism as the system_scope allowlist.  A new memory-touching
# entry without a registered authorization point is a leak by construction.
#
# ``engine`` documents which engine the authorization query runs on: the
# tenant (app) engine in every case -- never the maintenance engine, which
# authorize() refuses (a cross-tenant grant could satisfy the check).
# ---------------------------------------------------------------------------

ENTRY_REGISTRY: dict[str, dict] = {
    # ---- library API (MemoryEngine) ---------------------------------------
    "mirror_memory.api.MemoryEngine.observe": {
        "operation": "observe", "engine": "tenant(app)", "purpose_source": "parameter or engine default",
    },
    "mirror_memory.api.MemoryEngine.recall": {
        "operation": "recall", "engine": "tenant(app)", "purpose_source": "parameter or engine default",
    },
    "mirror_memory.api.MemoryEngine.panel": {
        "operation": "recall", "engine": "tenant(app)", "purpose_source": "parameter or engine default",
    },
    "mirror_memory.api.MemoryEngine.get_beliefs": {
        "operation": "explain", "engine": "tenant(app)", "purpose_source": "parameter or engine default",
    },
    "mirror_memory.api.MemoryEngine.get_verification_candidates": {
        "operation": "explain", "engine": "tenant(app)", "purpose_source": "parameter or engine default",
    },
    "mirror_memory.api.MemoryEngine.run_verification": {
        "operation": "correct", "engine": "tenant(app)", "purpose_source": "parameter or engine default",
    },
    "mirror_memory.api.MemoryEngine.belief_history": {
        "operation": "explain", "engine": "tenant(app)", "purpose_source": "parameter or engine default",
    },
    "mirror_memory.api.MemoryEngine.explain": {
        "operation": "explain", "engine": "tenant(app)", "purpose_source": "parameter or engine default",
    },
    "mirror_memory.api.MemoryEngine.correct": {
        "operation": "correct", "engine": "tenant(app)", "purpose_source": "parameter or engine default",
    },
    "mirror_memory.api.MemoryEngine.restore_belief": {
        "operation": "correct", "engine": "tenant(app)", "purpose_source": "parameter or engine default",
    },
    "mirror_memory.api.MemoryEngine.forget_belief": {
        "operation": "forget (allowed_regardless)", "engine": "tenant(app)",
        "purpose_source": "parameter or engine default; invocation audited",
    },
    "mirror_memory.api.MemoryEngine.forget_session": {
        "operation": "forget (allowed_regardless)", "engine": "tenant(app)",
        "purpose_source": "parameter or engine default; invocation audited",
    },
    "mirror_memory.api.MemoryEngine.delete_memories": {
        "operation": "forget (allowed_regardless)", "engine": "tenant(app)",
        "purpose_source": "parameter or engine default; invocation audited",
    },
    "mirror_memory.api.MemoryEngine.run_metabolism": {
        "operation": "extract", "engine": "tenant(app)", "purpose_source": "parameter or engine default",
    },
    # ---- HTTP server (routes call the engine entries above) ---------------
    "mirror_memory.server.observe": {
        "operation": "observe", "engine": "tenant(app)", "purpose_source": "X-MM-Purpose header, checked against the key's registered set (403 pre-SQL)",
    },
    "mirror_memory.server.recall": {
        "operation": "recall", "engine": "tenant(app)", "purpose_source": "X-MM-Purpose header, checked against the key's registered set (403 pre-SQL)",
    },
    "mirror_memory.server.panel": {
        "operation": "recall", "engine": "tenant(app)", "purpose_source": "X-MM-Purpose header, checked against the key's registered set (403 pre-SQL)",
    },
    "mirror_memory.server.beliefs": {
        "operation": "explain", "engine": "tenant(app)", "purpose_source": "X-MM-Purpose header, checked against the key's registered set (403 pre-SQL)",
    },
    "mirror_memory.server.belief_by_id": {
        "operation": "explain", "engine": "tenant(app)", "purpose_source": "X-MM-Purpose header, checked against the key's registered set (403 pre-SQL)",
    },
    "mirror_memory.server.stats": {
        "operation": "explain", "engine": "tenant(app)", "purpose_source": "X-MM-Purpose header, checked against the key's registered set (403 pre-SQL)",
    },
    "mirror_memory.server.delete_memories": {
        "operation": "forget (allowed_regardless)", "engine": "tenant(app)",
        "purpose_source": "X-MM-Purpose header, checked against the key's registered set (403 pre-SQL); invocation audited",
    },
    # ---- background worker ------------------------------------------------
    "mirror_memory.worker.evolution._process_single_job": {
        "operation": "extract", "engine": "tenant(app) for the job's tenant; maintenance engine only for the queue scan",
        "purpose_source": "the job's purpose column, recorded from the triggering observe at enqueue",
    },
    "mirror_memory.worker.evolution.process_pending": {
        "operation": "extract", "engine": "maintenance engine (scan) -> tenant(app) per job",
        "purpose_source": "each job's purpose column; the scan itself reads only the queue",
    },
    # ---- benchmark harness (host-side, explicit bench purpose) ------------
    "mirror_memory.bench.longrun.run_longrun": {
        "operation": "observe/recall/extract via the engine entries",
        "engine": "tenant(app)", "purpose_source": "default_purpose='benchmark', granted at setup",
    },
    "mirror_memory.bench.runner.run_cases": {
        "operation": "observe/recall via the engine entries",
        "engine": "tenant(app)", "purpose_source": "default_purpose='benchmark', granted at setup",
    },
}

#: Engine methods that are deliberately NOT purpose-gated, with the reason.
#: Anything not listed here and not in ENTRY_REGISTRY fails the discovery
#: test.
UNGATED_ENGINE_METHODS: dict[str, str] = {
    "set_enabled": "authorization administration (the feature switch itself)",
    "set_consent": "authorization administration (feature consent)",
    "grant_purpose": "authorization administration (issuance)",
    "revoke_purpose": "authorization administration (revocation)",
}
