"""MemoryEngine — the public API for mirror-memory.

Usage::

    from mirror_memory import MemoryEngine

    # One-liner with built-in LLM adapter:
    engine = MemoryEngine(
        config_path="config/",
        llm_api_key="sk-xxx",
        llm_model="deepseek-chat",
        llm_base_url="https://api.deepseek.com",
    )

    engine.observe(user_id="u1", session_id="s1", text="I love painting")
    context = engine.recall(user_id="u1", query="What are the user's hobbies?")
"""

from __future__ import annotations

import logging
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, NamedTuple

from mirror_memory.config.loader import load_config
from mirror_memory.config.schema import MemoryConfig
from mirror_memory.exceptions import ValidationError
from mirror_memory.models import BeliefInfo, DeleteResult, PanelData

logger = logging.getLogger(__name__)


#: ENTRY_REGISTRY keys of the memory-reading library entries (the audit
#: rows name them; the server relabels with its own route key).
ENTRY_RECALL = "mirror_memory.api.MemoryEngine.recall"
ENTRY_PANEL = "mirror_memory.api.MemoryEngine.panel"
ENTRY_GET_BELIEFS = "mirror_memory.api.MemoryEngine.get_beliefs"
ENTRY_VERIFICATION_CANDIDATES = "mirror_memory.api.MemoryEngine.get_verification_candidates"
ENTRY_BELIEF_HISTORY = "mirror_memory.api.MemoryEngine.belief_history"
ENTRY_EXPLAIN = "mirror_memory.api.MemoryEngine.explain"
ENTRY_EXPORT_PAGE = "mirror_memory.api.MemoryEngine.export_page"
ENTRY_EXPORT = "mirror_memory.api.MemoryEngine.export"


from mirror_memory.core import provenance
from mirror_memory.core.deletion import DeletionEntry, DeletionLog, DeletionLogUnavailable
from mirror_memory.core.export import ExportPage  # re-exported for hosts


class Authorized(NamedTuple):
    """What an entry knows after its authorization passed: the resolved
    purpose and the grant version that allowed it (both go into the
    access audit row)."""

    purpose: str
    grant_version: int | None


class MemoryEngine:
    """Structured memory engine for AI agents.

    Wraps extraction, storage, retrieval, and rendering into a single API.
    All domain-specific content is loaded from configuration files.

    Parameters
    ----------
    config_path:
        Path to the configuration directory.
    config:
        Pre-loaded ``MemoryConfig``.  Takes precedence over *config_path*.
    database_url:
        SQLAlchemy database URL.
    llm_client:
        Pre-built LLM client (must expose ``generate()``).
        Takes precedence over ``llm_api_key``/``llm_model``/``llm_base_url``.
    llm_api_key:
        API key for the built-in OpenAI-compatible LLM adapter.
    llm_model:
        Model identifier (e.g. ``"gpt-4o-mini"``, ``"deepseek-chat"``).
    llm_base_url:
        API base URL for non-OpenAI providers.
    """

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        config: MemoryConfig | None = None,
        database_url: str = "sqlite:///mirror_memory.db",
        llm_client: Any | None = None,
        llm_api_key: str | None = None,
        llm_model: str = "gpt-4o-mini",
        llm_base_url: str | None = None,
        tenant_id: str = "default",
        app_id: str = "default",
        single_tenant_mode: bool | None = None,
        engine: Any | None = None,
        schema_ready: bool = False,
        default_purpose: str | None = None,
        deletion_log: "DeletionLog | None" = None,
    ) -> None:
        if config is not None:
            self._config = config
        elif config_path is not None:
            self._config = load_config(config_path)
        else:
            raise ValueError("Provide either config_path or config")

        self._database_url = database_url
        self._llm_client = llm_client or self._build_llm(llm_api_key, llm_model, llm_base_url)
        # M1 tenancy: the engine entry point pins an explicit tenant; every
        # session it opens is created inside that scope and therefore bound
        # to it at creation time (see core.tenancy).
        self._tenant = (tenant_id, app_id)
        # M2 batch-2 task 2: a pre-built SQLAlchemy engine may be shared
        # between tenant wrappers (the server on PostgreSQL shares ONE pool
        # across tenants).  A wrapper never disposes a pool it does not own,
        # and ``schema_ready`` skips the schema/posture checks the owner of
        # the shared engine already ran.
        self._shared_engine = engine
        self._owns_engine = engine is None
        self._schema_ready = bool(schema_ready and engine is not None)
        # M3: an explicit engine-level default purpose (an end-side
        # single-tenant host sets it once).  None means NO default: callers
        # must pass a purpose; there is no implicit one.
        self._default_purpose = default_purpose
        # Task 6: end-side single-tenant mode (None -> env MM_SINGLE_TENANT).
        import os

        self._single_tenant = (
            single_tenant_mode
            if single_tenant_mode is not None
            else os.environ.get("MM_SINGLE_TENANT", "0") == "1"
        )
        self._initialized = False
        self._pipeline = None  # lazy-initialized
        # M4: the independent deletion log.  An explicit instance wins;
        # otherwise the config decides (backend file|postgres).  None =>
        # deletions are refused (fail-closed: an unlogged deletion cannot
        # be replayed after a restore).
        self._deletion_log = deletion_log if deletion_log is not None else self._build_deletion_log()
        self._db_id: str | None = None

    def _build_deletion_log(self) -> "DeletionLog | None":
        """The deletion log from configuration (M4 task 1).  backend
        "file" (default): an append-only JSONL file per database life
        under ``deletion_log.directory`` -- fsynced, hash-chained, NEVER
        the main database file.  backend "postgres": the log table in a
        SEPARATE database at ``deletion_log.url`` (outside any pg_dump of
        the main DB), where the app role holds INSERT/SELECT only."""
        from mirror_memory.core.deletion import FileDeletionLog, PgDeletionLog

        settings = self._config.deletion_log
        if settings.backend == "none":
            return None
        if settings.backend == "postgres":
            url = settings.url
            if not url:
                raise ValueError("deletion_log.backend=postgres requires deletion_log.url")
            return PgDeletionLog(url)
        directory = settings.directory
        if not directory:
            if self._database_url.startswith("sqlite:///"):
                db_path = self._database_url[len("sqlite:///"):]
                directory = str((Path(db_path).parent / "deletions").resolve())
            else:
                import hashlib as _hashlib

                digest = _hashlib.sha1(self._database_url.encode()).hexdigest()[:12]
                directory = str(Path(tempfile.gettempdir()) / f"mm-deletions-{digest}")
        return FileDeletionLog(directory)

    @staticmethod
    def _build_llm(api_key: str | None, model: str, base_url: str | None) -> Any:
        """Build the built-in OpenAI-compatible LLM adapter."""
        if api_key is None:
            return None
        from mirror_memory.llm import OpenAILLM

        return OpenAILLM(api_key=api_key, model=model, base_url=base_url)

    def _ensure_db(self) -> None:
        """Create tables on first use (SQLite) or verify the owner-created
        schema and the row-level-security posture (PostgreSQL)."""
        if self._initialized:
            return
        from mirror_memory.core.migrate import check_schema
        from mirror_memory.core.models import Base
        from mirror_memory.core.pg_engine import build_engine
        from mirror_memory.core.rls import is_postgres, prepare_runtime_engine, rls_enabled

        if self._shared_engine is not None:
            self._engine = self._shared_engine
            if self._schema_ready:
                # The owner of the shared engine ran the schema / posture /
                # single-tenant checks once; a per-tenant wrapper only binds.
                from mirror_memory.core import schema_migration

                self._db_id, self._applied_seq = schema_migration.read_watermark(
                    self._engine
                ) or (None, 0)
                self._initialized = True
                return
        else:
            self._engine = build_engine(self._database_url)
        if is_postgres(self._engine) and rls_enabled():
            # The app role never runs DDL: the owner-side upgrade_database
            # created the schema, and the engine refuses to serve unless
            # every tenant table is RLS-protected and this login is subject
            # to the policies (core.rls).
            prepare_runtime_engine(self._engine)
        else:
            Base.metadata.create_all(self._engine)
        # M4: the deletion watermark (db_id + applied seq) is created
        # OUTSIDE any transaction -- it must exist before the first
        # deletion (which only READS it; creating it here also avoids a
        # second-connection DDL inside a SQLite write transaction).  A
        # fresh database gets a fresh db_id; a RESTORED one brings its
        # watermark row back, which is how a restore is detected.
        from mirror_memory.core import schema_migration

        self._db_id, self._applied_seq = schema_migration.ensure_watermark(self._engine)
        # M4: startup check against the deletion log.  Equal -> normal; the
        # main DB behind its log (a restored backup) -> replay now, in this
        # process, before any entry serves; the log behind the main DB ->
        # refuse.  Another instance holding the replay lock raises
        # ReplayPending: this engine fails to start and the caller retries.
        from sqlalchemy.orm import Session as _Session

        from mirror_memory.core import replay

        replay.reconcile(lambda: _Session(self._engine), self._deletion_log,
                         tenant=self._tenant)
        # A database written by an older engine lacks the metabolism columns
        # the ORM now expects; fail with instructions instead of a mid-query
        # "no such column" error.
        check_schema(self._engine)
        if self._single_tenant:
            self._verify_single_tenant_startup()
            from mirror_memory.core.tenancy import set_single_tenant_mode

            set_single_tenant_mode(True)
        self._initialized = True

    def _verify_single_tenant_startup(self) -> None:
        """Fail-closed startup checks for end-side single-tenant mode.

        Refuses when (a) the server tenant registry / key file is also
        configured (mutually exclusive), (b) the database already holds
        more than one tenant, or (c) the stored tenant differs from the
        configured one.
        """
        import os

        from sqlalchemy import text as _text

        from mirror_memory.core.tenancy import system_scope

        if os.environ.get("MM_TENANT_KEYS_FILE"):
            raise RuntimeError(
                "single-tenant mode is mutually exclusive with the server "
                "tenant middleware (MM_TENANT_KEYS_FILE is set)"
            )
        try:
            from mirror_memory import server as _server

            if _server._tenant_key_registry:
                raise RuntimeError(
                    "single-tenant mode is mutually exclusive with the "
                    "server tenant middleware (tenant keys registered)"
                )
        except ImportError:
            pass
        with system_scope(reason="single-tenant startup: distinct-tenant check"):
            # PostgreSQL with row-level security: the app role sees only its
            # own tenant's rows, which would make this check vacuous -- the
            # probe runs on the maintenance engine (core.rls).
            from mirror_memory.core.rls import system_engine_for

            with system_engine_for(self._engine).connect() as conn:
                rows = conn.execute(
                    _text(
                        "SELECT DISTINCT tenant_id, app_id FROM mm_beliefs "
                        "LIMIT 2"
                    )
                ).all()
        stored = {(r[0], r[1]) for r in rows}
        if len(stored) > 1:
            raise RuntimeError(
                f"single-tenant mode refused: database holds multiple "
                f"tenants {sorted(stored)}"
            )
        if stored and stored != {self._tenant}:
            raise RuntimeError(
                f"single-tenant mode refused: database tenant "
                f"{sorted(stored)} != configured {self._tenant}"
            )

    @contextmanager
    def _session(self):
        """Open a session bound to this engine's tenant.

        The session is created inside the tenant scope, so tenancy binds it
        at creation time (core.tenancy session binding); the scope also
        covers the work done with the session.
        """
        from sqlalchemy.orm import Session

        from mirror_memory.core.tenancy import tenant_scope

        with tenant_scope(*self._tenant):
            session = Session(self._engine)
            try:
                yield session
            finally:
                session.close()

    def _get_pipeline(self):
        """Lazy-initialize the extraction pipeline (singleton per engine)."""
        if self._pipeline is None:
            from mirror_memory.extraction.pipeline import ExtractionPipeline

            self._pipeline = ExtractionPipeline(config=self._config, llm_client=self._llm_client)
        return self._pipeline

    # -- purpose authorization (M3 batch-1 tasks 2/3) ------------------------

    def _purpose_or_default(self, purpose: str | None) -> str:
        """The effective purpose: the call's argument or the engine-level
        default.  Neither present -> PurposeDenied(missing_purpose); there
        is no implicit default."""
        from mirror_memory.core import metrics
        from mirror_memory.core.authorization import Decision, PurposeDenied

        resolved = purpose if purpose is not None else self._default_purpose
        if not resolved or not str(resolved).strip():
            # decided in Python before any SQL (batch-1 invariant): the
            # count is in-process, never a query
            metrics.inc(metrics.MISSING_PURPOSE)
            raise PurposeDenied(Decision(False, "missing_purpose", None, ""), "unresolved")
        return str(resolved).strip()

    def _authorize(
        self, session, user_id: str, purpose: str | None, operation: str,
        *, audit_entry: str | None = None,
    ) -> "Authorized":
        """Authorize *operation* in the CURRENT transaction, before any
        business statement; raises PurposeDenied on refusal.

        With *audit_entry* (the ENTRY_REGISTRY key of a memory-reading
        entry) a refusal is written to the access audit -- decision
        ``denied`` with the reason -- and COMMITTED before the exception
        propagates: nothing else is pending this early in the transaction,
        and a probing caller must leave a trace.  The one refusal that is
        NOT audited is ``missing_purpose``: it is decided in Python before
        any SQL (batch-1 invariant) and has nothing to attribute -- no
        purpose, possibly no real user.  The audit INSERT on the allowed
        path is the entry's own job (:meth:`_audit`)."""
        from mirror_memory.core import access_audit, authorization, provenance

        resolved = self._purpose_or_default(purpose)
        try:
            decision = authorization.authorize_or_raise(session, user_id, resolved, operation)
        except authorization.PurposeDenied as exc:
            if audit_entry is not None:
                access_audit.record_access(
                    session, user_id=user_id, purpose=resolved,
                    operation=operation, entry=audit_entry, decision=access_audit.DENIED,
                    reason=exc.decision.reason, grant_version=exc.decision.grant_version,
                    max_ids=self._config.access_audit.max_ids,
                )
                session.commit()
            raise
        # everything this transaction writes is collected under `resolved`
        provenance.bind(session, resolved)
        return Authorized(resolved, decision.grant_version)

    def _audit(
        self, session, authorized: "Authorized", *, user_id: str, operation: str,
        entry: str, memory_ids: list[int],
    ) -> None:
        """Append the allowed-access audit row for a read entry.  Same
        transaction as the read; the caller commits -- and a failure of
        either the INSERT or that commit propagates, so the entry returns
        nothing (fail-closed: memory never leaves unrecorded)."""
        from mirror_memory.core import access_audit

        access_audit.record_access(
            session, user_id=user_id, purpose=authorized.purpose, operation=operation,
            entry=entry, decision=access_audit.ALLOWED,
            grant_version=authorized.grant_version, memory_ids=memory_ids,
            max_ids=self._config.access_audit.max_ids,
        )

    def _append_deletion_log(self, session, user_id: str, *, scope: str,
                             scope_key: str, fingerprints: list[dict],
                             generation: int, purpose: str | None) -> None:
        """Write the deletion-log entry for a deletion that is about to
        commit.  The log write happens BEFORE the main-DB commit (the
        entry is part of the transaction's preconditions): a log failure
        raises and rolls the deletion back; the reverse window -- log
        written, commit fails -- is safe because replay matches
        fingerprints that will never be found (core.deletion docstring)."""
        from mirror_memory.core.deletion import DeletionEntry

        if self._deletion_log is None:
            raise DeletionLogUnavailable(
                "no deletion log is configured: deletions are refused "
                "(an unlogged deletion cannot be replayed after a restore)"
            )
        if not self._db_id:
            raise DeletionLogUnavailable("the deletion watermark is missing: run the schema setup first")
        entry = self._deletion_log.append(DeletionEntry(
            db_id=self._db_id, tenant_id=self._tenant[0], app_id=self._tenant[1],
            user_id=user_id, scope=scope, scope_key=scope_key,
            fingerprints=fingerprints, generation=generation,
            purpose=self._purpose_or_default(purpose),
        ))
        # the entry is APPLIED by this very transaction: advance the
        # watermark here, so a committed deletion never needs a replay
        # (only a restore does)
        from mirror_memory.core import schema_migration

        wtable = schema_migration._watermark_table(schema_migration.MetaData())
        session.execute(
            wtable.update().where(wtable.c.id == 1).values(applied_seq=entry.seq))
        self._applied_seq = entry.seq

    def _current_deletion_generation(self, session, user_id: str) -> int:
        from sqlalchemy import func as _func, select as _select

        from mirror_memory.core.models import DeletionGeneration

        return int(session.scalar(
            _select(_func.max(DeletionGeneration.generation))
            .where(DeletionGeneration.user_id == user_id)
        ) or 0)

    def _record_forget_invocation(self, session, user_id: str, purpose: str | None) -> str:
        """The forget paths run regardless of authorization state (a revoked
        user can still have their data erased) but must carry a purpose, and
        every invocation is written to the authorization audit trail."""
        from mirror_memory.core import authorization, provenance

        resolved = self._purpose_or_default(purpose)
        authorization.record_purpose_invocation(
            session, user_id, resolved,
            operation="forget", decision="allowed_regardless", actor="engine",
        )
        provenance.bind(session, resolved)
        return resolved

    def observe(
        self,
        *,
        user_id: str,
        session_id: str,
        text: str,
        turn_count: int = 1,
        purpose: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Process a single conversation turn — extract and store beliefs.

        Fail-open: extraction errors never block the caller.

        Returns
        -------
        int
            The number of claims extracted this turn.

        Raises
        ------
        ValidationError
            If user_id, session_id, or text is empty, or text exceeds
            the maximum length.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")
        if not session_id or not session_id.strip():
            raise ValidationError("session_id must be a non-empty string")
        if not text or not text.strip():
            raise ValidationError("text must be a non-empty string")
        if len(text) > 50000:
            raise ValidationError("text exceeds maximum length of 50000 characters")

        self._ensure_db()
        pipeline = self._get_pipeline()
        started = time.monotonic()
        with self._session() as session:
            self._authorize(session, user_id, purpose, "observe")
            claims = pipeline.observe(
                session,
                user_id=user_id,
                session_id=session_id,
                text=text,
                turn_count=turn_count,
                **kwargs,
            )
            session.commit()
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "observe: user=%s session=%s claims=%d latency=%dms",
            user_id[:8], session_id[:16], len(claims), elapsed_ms,
        )
        return len(claims)

    def recall(
        self,
        *,
        user_id: str,
        query: str = "",
        language: str = "en",
        tail_load: int = 0,
        trace_hook: Any | None = None,
        now: Any | None = None,
        purpose: str | None = None,
    ) -> str | None:
        """Retrieve a rendered memory block for prompt injection.

        Parameters
        ----------
        user_id:
            The user identifier.
        query:
            The current user message, used for query-aware scoring.
        language:
            ``"zh"`` or ``"en"``.
        tail_load:
            Character count of content already in the prompt.
        trace_hook:
            Optional observability callable ``trace_hook(stage, **fields)``
            receiving the retrieve/context stages.  Purely additive -- it
            never changes what is rendered.
        now:
            Reference time for freshness scoring, time labels, and the
            access telemetry this call records.  Defaults to the current
            UTC time; benchmarks and tests pass a simulated clock so a
            recall can be positioned on the same timeline as the
            metabolism cycle that follows it.

        Returns
        -------
        str or None
            The rendered memory block, or ``None`` if no memory is
            available.

        Raises
        ------
        ValidationError
            If user_id is empty.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")

        self._ensure_db()
        from mirror_memory.render.renderer import render_memory_block_with_ids

        started = time.monotonic()
        with self._session() as session:
            authorized = self._authorize(session, user_id, purpose, "recall", audit_entry=ENTRY_RECALL)
            rule = provenance.visibility(session, user_id, authorized.purpose)
            result, rendered_ids = render_memory_block_with_ids(
                session,
                user_id,
                config=self._config,
                language=language,
                user_message=query,
                tail_load=tail_load,
                trace_hook=trace_hook,
                now=now,
                visibility=rule,
            )
            # The access audit row (M3 batch-2 task 2) rides the same
            # transaction as the read, together with the access telemetry
            # (which beliefs reached the prompt -- the reheat signal).  The
            # commit is mandatory: if the audit cannot be written the block
            # is NOT returned (fail-closed).
            self._audit(session, authorized, user_id=user_id, operation="recall",
                        entry=ENTRY_RECALL, memory_ids=rendered_ids)
            session.commit()
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "recall: user=%s chars=%d latency=%dms",
            user_id[:8], len(result or ""), elapsed_ms,
        )
        return result

    def run_metabolism(
        self, *, user_id: str, dry_run: bool = False, now: Any | None = None, purpose: str | None = None,
    ) -> dict:
        """Run one memory-metabolism cycle for a user.

        Scans the user's beliefs, scores heat, applies protection, and
        proposes storage-tier transitions (hot → warm → dormant → archived,
        plus reheat and evidence compaction) through the Publisher — the
        cycle itself never writes directly.  Designed to be scheduled daily
        per user (e.g. 02:00) by the host application.

        Parameters
        ----------
        user_id:
            The user identifier.
        dry_run:
            Compute every decision but publish nothing.
        now:
            Reference time for the cycle.  Defaults to the current UTC
            time; benchmarks and tests pass a simulated clock so retention
            windows (``cool_after_days`` / ``archive_after_days``) can be
            exercised without waiting.

        Returns
        -------
        dict
            The cycle report: scanned / protected / proposals / committed /
            refused counts, tier distribution before and after, and every
            transition (or refusal) with its reason.

        Raises
        ------
        ValidationError
            If user_id is empty.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")

        self._ensure_db()
        from mirror_memory.metabolism.planner import run_metabolism as run_cycle

        with self._session() as session:
            self._authorize(session, user_id, purpose, "extract")
            report = run_cycle(session, user_id, self._config, dry_run=dry_run, now=now)
            session.commit()
            return report.as_dict()

    def panel(
        self,
        *,
        user_id: str,
        language: str = "en",
        purpose: str | None = None,
    ) -> PanelData:
        """Build a structured panel of the user's memory for frontend display.

        Returns
        -------
        PanelData
            Structured panel with memory status, sections, and avatar info.

        Raises
        ------
        ValidationError
            If user_id is empty or language is invalid.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")
        if language not in ("zh", "en"):
            raise ValidationError(f"language must be 'zh' or 'en', got '{language}'")

        self._ensure_db()
        from mirror_memory.render.panel import build_panel

        with self._session() as session:
            authorized = self._authorize(session, user_id, purpose, "recall", audit_entry=ENTRY_PANEL)
            rule = provenance.visibility(session, user_id, authorized.purpose)
            raw = build_panel(session, user_id, config=self._config, language=language, visibility=rule)
            self._audit(session, authorized, user_id=user_id, operation="recall",
                        entry=ENTRY_PANEL, memory_ids=list(raw.get("belief_ids", [])))
            session.commit()
            return PanelData(
                memory=raw.get("memory", {}),
                sections=raw.get("sections", []),
                avatar=raw.get("avatar", {}),
            )

    def get_beliefs(
        self,
        *,
        user_id: str,
        dimensions: list[str] | None = None,
        limit: int = 50,
        purpose: str | None = None,
    ) -> list[BeliefInfo]:
        """List active beliefs for a user.

        Returns
        -------
        list[BeliefInfo]
            Active beliefs sorted by last evidence time.

        Raises
        ------
        ValidationError
            If user_id is empty or limit is out of range.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")
        if limit < 1 or limit > 200:
            raise ValidationError("limit must be between 1 and 200")

        self._ensure_db()
        from mirror_memory.core.repository import list_active_beliefs

        with self._session() as session:
            authorized = self._authorize(session, user_id, purpose, "explain", audit_entry=ENTRY_GET_BELIEFS)
            rule = provenance.visibility(session, user_id, authorized.purpose)
            beliefs = list_active_beliefs(
                session, user_id, dimensions=dimensions, limit=limit, visibility=rule,
            )
            self._audit(session, authorized, user_id=user_id, operation="explain",
                        entry=ENTRY_GET_BELIEFS, memory_ids=[b.id for b in beliefs])
            session.commit()
            return [
                BeliefInfo(
                    dimension=b.dimension,
                    key=b.key,
                    claim_text=b.claim_text,
                    confidence=b.confidence,
                    layer=b.layer,
                    source=b.source,
                    last_evidence_at=b.last_evidence_at,
                    subject=getattr(b, "subject", "user"),
                    predicate=getattr(b, "predicate", ""),
                    object=getattr(b, "object", ""),
                    cardinality=getattr(b, "cardinality", "multi"),
                    memory_tier=getattr(b, "memory_tier", "hot"),
                    retention_class=getattr(b, "retention_class", ""),
                )
                for b in beliefs
            ]

    def get_verification_candidates(
        self, *, user_id: str, limit: int = 1, session_id: str = "", purpose: str | None = None,
    ) -> list[dict]:
        """Get beliefs eligible for verification questioning.

        Selects L4 beliefs (confidence >= question threshold, cross-session
        evidence) sorted by ``question_value_tiers * confirm_rate_weight``,
        and records a ``question_injected`` intervention event for the top
        candidate so that :meth:`run_verification` can judge the user's reply
        on a later turn.  A new question is never injected while a previous
        one is still unanswered.

        Parameters
        ----------
        user_id:
            The user identifier.
        limit:
            Maximum number of candidates to return.
        session_id:
            Optional session the question is surfaced in (recorded on the
            injection event for the pending-verification window).

        Returns
        -------
        list[dict]
            Candidate dicts with ``belief_id``, ``dimension``, ``key``,
            ``claim_text``, ``confidence``, ``question`` and ``injected``.
            Empty when the loop is disabled (no LLM client or empty
            ``verification`` prompt) or on any error -- fail-open.

        Raises
        ------
        ValidationError
            If user_id is empty or limit is out of range.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")
        if limit < 1 or limit > 200:
            raise ValidationError("limit must be between 1 and 200")
        if self._llm_client is None or not (self._config.prompts.verification or "").strip():
            logger.info("verification: disabled (no LLM client or empty verification prompt)")
            return []

        self._ensure_db()
        from mirror_memory.extraction.verification import (
            build_verification_question,
            get_question_candidates,
            record_question_injection,
        )

        with self._session() as session:
            authorized = self._authorize(
                session, user_id, purpose, "explain", audit_entry=ENTRY_VERIFICATION_CANDIDATES,
            )
            rule = provenance.visibility(session, user_id, authorized.purpose)
            results: list[dict] = []
            try:
                beliefs = get_question_candidates(
                    session, user_id, self._config, limit=limit, visibility=rule,
                )
                injected = False
                for belief in beliefs:
                    recorded = False
                    if not injected:
                        event = record_question_injection(
                            session, user_id, session_id, belief, self._config
                        )
                        if event is not None:
                            recorded = True
                            injected = True
                    results.append(
                        {
                            "belief_id": int(belief.id),
                            "dimension": belief.dimension,
                            "key": belief.key,
                            "claim_text": belief.claim_text,
                            "confidence": float(belief.confidence),
                            "question": build_verification_question(belief, self._config),
                            "injected": recorded,
                        }
                    )
            except Exception:
                # the candidate selection is fail-open (an empty list, never
                # an error for the caller) -- the audit below is not
                logger.warning("get_verification_candidates failed; returning []", exc_info=True)
                session.rollback()
                results = []
            self._audit(session, authorized, user_id=user_id, operation="explain",
                        entry=ENTRY_VERIFICATION_CANDIDATES,
                        memory_ids=[r["belief_id"] for r in results])
            session.commit()
            return results

    def run_verification(
        self, *, user_id: str, session_id: str, user_text: str, purpose: str | None = None,
    ) -> bool:
        """Run verification judgment for this turn. Returns True if turn consumed.

        Judges the user's reply against the pending verification question
        (if any) and drives the belief lifecycle: confirm -> L4 -> L2,
        deny -> rejected, unclear -> event only.  Fail-open: errors never
        propagate; the caller should proceed with normal extraction when
        ``False`` is returned.

        Raises
        ------
        ValidationError
            If user_id or session_id is empty.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")
        if not session_id or not session_id.strip():
            raise ValidationError("session_id must be a non-empty string")
        if not user_text or not user_text.strip():
            logger.debug("run_verification: empty user_text -- nothing to judge")
            return False

        self._ensure_db()
        from mirror_memory.extraction.verification import run_verification_judgment

        with self._session() as session:
            self._authorize(session, user_id, purpose, "correct")
            try:
                consumed = run_verification_judgment(
                    session,
                    user_id,
                    session_id,
                    user_text,
                    self._config,
                    self._llm_client,
                )
                session.commit()
                return consumed
            except Exception:
                logger.warning("run_verification failed; fail-open", exc_info=True)
                session.rollback()
                return False

    def set_enabled(self, *, user_id: str, enabled: bool) -> None:
        """Enable or disable memory for a user.

        Returns
        -------
        None

        Raises
        ------
        ValidationError
            If user_id is empty.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")

        self._ensure_db()
        from mirror_memory.core.repository import set_memory_enabled

        with self._session() as session:
            set_memory_enabled(session, user_id, enabled)
            session.commit()

    def grant_purpose(
        self,
        *,
        user_id: str,
        purpose: str,
        operations: list[str],
        issuer: str,
        expires_at: Any | None = None,
        expected_version: int | None = None,
        reason: str = "",
    ) -> dict:
        """Issue or update one purpose authorization (M3 batch 1 task 1).

        First issuance takes no ``expected_version``; updating an existing
        grant requires it (the optimistic lock -- a stale issuer is
        refused, concurrent issuers cannot clobber each other).  Every
        accepted change appends a row to ``mm_purpose_grant_events``.
        """
        self._ensure_db()
        from mirror_memory.core import authorization

        with self._session() as session:
            grant = authorization.grant_purpose(
                session, user_id, purpose,
                operations=operations, issuer=issuer, expires_at=expires_at,
                expected_version=expected_version, reason=reason,
            )
            session.commit()
            return {
                "user_id": grant.user_id,
                "purpose": grant.purpose,
                "version": grant.version,
                "allowed_operations": grant.allowed_operations,
                "revoked": grant.revoked_at is not None,
            }

    def revoke_purpose(
        self,
        *,
        user_id: str,
        purpose: str,
        actor: str,
        reason: str = "",
        expected_version: int | None = None,
    ) -> dict:
        """Revoke one purpose authorization (immediate: the next
        ``authorize`` re-reads the row).  Idempotent."""
        self._ensure_db()
        from mirror_memory.core import authorization

        with self._session() as session:
            grant = authorization.revoke_purpose(
                session, user_id, purpose, actor=actor, reason=reason,
                expected_version=expected_version,
            )
            session.commit()
            if grant is None:
                return {"user_id": user_id, "purpose": purpose, "revoked": False, "existed": False}
            return {
                "user_id": grant.user_id,
                "purpose": grant.purpose,
                "version": grant.version,
                "revoked": grant.revoked_at is not None,
                "existed": True,
            }

    def set_consent(self, *, user_id: str, feature: str, granted: bool) -> None:
        """Set feature-level consent for a user.

        Parameters
        ----------
        user_id:
            The user identifier.
        feature:
            Feature name (e.g. ``"extraction"``, ``"recall"``, ``"verification"``).
        granted:
            Whether the feature is allowed.

        Raises
        ------
        ValidationError
            If user_id or feature is empty.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")
        if not feature or not feature.strip():
            raise ValidationError("feature must be a non-empty string")

        self._ensure_db()
        from mirror_memory.core.repository import set_feature_consent

        with self._session() as session:
            set_feature_consent(session, user_id, feature.strip(), granted)
            session.commit()

    def purge_access_audit(self, *, older_than: Any | None = None) -> int:
        """Maintenance: delete this tenant's access-audit rows that are out
        of retention (``config.access_audit.retention_days``, or an explicit
        *older_than* instant).  Returns the number removed.  Not purpose
        gated: it reads no memory and is the operator's retention duty, run
        per tenant by the host.  Never automatic."""
        from mirror_memory.core import access_audit

        self._ensure_db()
        cutoff = older_than if older_than is not None else access_audit.retention_cutoff(
            self._config.access_audit.retention_days,
        )
        with self._session() as session:
            removed = access_audit.purge_access_audit(session, older_than=cutoff)
            session.commit()
        logger.info("access audit: purged %d row(s) older than %s", removed, cutoff)
        return removed

    def delete_memories(self, *, user_id: str, purpose: str | None = None) -> DeleteResult:
        """Delete all memory data for a user.

        Returns
        -------
        DeleteResult
            Per-table deletion counts.

        Raises
        ------
        ValidationError
            If user_id is empty.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")

        self._ensure_db()
        from mirror_memory.core.repository import delete_user_memories

        with self._session() as session:
            self._record_forget_invocation(session, user_id, purpose)
            fingerprints: list[dict] = []
            counts = delete_user_memories(session, user_id, fingerprints=fingerprints)
            self._append_deletion_log(
                session, user_id, scope="user", scope_key="",
                fingerprints=fingerprints, generation=counts["generation"],
                purpose=purpose,
            )
            session.commit()
            counts.pop("generation", None)
            return DeleteResult(**counts)

    def forget_belief(self, *, user_id: str, belief_id: int, purpose: str | None = None) -> bool:
        """Targeted forget: delete a single belief and everything derived from it.

        The deletion is one transaction — belief, events, evidence links,
        digest, and tier-audit rows go together, orphaned evidence is
        pruned, and a content-free tombstone advances the user's deletion
        generation (the barrier that stops a stale worker from writing the
        data back).  Routed through the Publisher like every other state
        change; unlike collection, deletion is allowed even while memory
        is disabled.

        Returns ``True`` if the belief was found and deleted.

        Raises
        ------
        ValidationError
            If user_id is empty.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")

        self._ensure_db()
        from mirror_memory.core.proposal import (
            TRANSITION_FORGET,
            StateTransitionProposal,
        )
        from mirror_memory.core.publisher import Publisher
        from mirror_memory.core.repository import get_state_revision

        with self._session() as session:
            self._record_forget_invocation(session, user_id, purpose)
            collector: dict = {"fingerprints": [], "generation": None}
            decision = Publisher(session, self._config).publish(
                StateTransitionProposal(
                    transition=TRANSITION_FORGET,
                    user_id=user_id,
                    target_belief_id=belief_id,
                    expected_revision=get_state_revision(session, user_id),
                    payload={"deletion_collector": collector},
                )
            )
            if decision.committed and collector["generation"] is not None:
                # the deletion log is written BEFORE the main-DB commit
                # (fail-closed: a log failure rolls the deletion back)
                self._append_deletion_log(
                    session, user_id, scope="belief", scope_key=str(belief_id),
                    fingerprints=collector["fingerprints"],
                    generation=collector["generation"], purpose=purpose,
                )
            session.commit()
            return decision.committed

    def restore_belief(self, *, user_id: str, belief_id: int, purpose: str | None = None) -> bool:
        """Restore an archived belief into the default retrieval scan.

        Archiving is reversible by design: an archived belief has left the
        default scan (its conclusion is still true, its evidence is still
        summarised) but an explicit restore brings it back to warm through
        the Publisher — the same proposal path as every tier move, with
        the same invariants (legal edge, scope, revision).

        Returns ``True`` if the belief was archived and has been restored.

        Raises
        ------
        ValidationError
            If user_id is empty.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")

        self._ensure_db()
        from mirror_memory.core.models import Belief
        from mirror_memory.core.proposal import (
            TRANSITION_TIER_TRANSITION,
            StateTransitionProposal,
        )
        from mirror_memory.core.publisher import Publisher
        from mirror_memory.core.repository import get_state_revision

        with self._session() as session:
            self._authorize(session, user_id, purpose, "correct")
            belief = session.get(Belief, belief_id)
            if belief is None or belief.user_id != user_id:
                return False
            from mirror_memory.core import provenance

            rule = provenance.visibility(session, user_id, self._purpose_or_default(purpose))
            if not provenance.belief_is_visible(session, user_id, rule, belief_id):
                return False  # identical to not-found; nothing written
            from_tier = belief.memory_tier or "hot"
            decision = Publisher(session, self._config).publish(
                StateTransitionProposal(
                    transition=TRANSITION_TIER_TRANSITION,
                    user_id=user_id,
                    target_belief_id=belief_id,
                    expected_revision=get_state_revision(session, user_id),
                    payload={
                        "from_tier": from_tier,
                        "to_tier": "warm",
                        "reason": "restore",
                        "score": 0.0,
                    },
                )
            )
            session.commit()
            return decision.committed

    def forget_session(self, *, user_id: str, session_id: str, purpose: str | None = None) -> bool:
        """Targeted forget: delete a session summary by session_id.

        Returns ``True`` if the summary was found and deleted.

        Raises
        ------
        ValidationError
            If user_id or session_id is empty.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")
        if not session_id or not session_id.strip():
            raise ValidationError("session_id must be a non-empty string")

        self._ensure_db()
        from mirror_memory.core.repository import forget_session_summary

        with self._session() as session:
            self._record_forget_invocation(session, user_id, purpose)
            fingerprints: list[dict] = []
            result = forget_session_summary(session, user_id, session_id,
                                            fingerprints=fingerprints)
            if result:
                # the summary rows were deleted: log before commit.  Session
                # scope does not bump the generation (no belief content
                # changed) -- M4's log entry records the deletion for replay
                # with the user's CURRENT generation.
                self._append_deletion_log(
                    session, user_id, scope="session", scope_key=session_id,
                    fingerprints=fingerprints,
                    generation=self._current_deletion_generation(session, user_id),
                    purpose=purpose,
                )
            session.commit()
            return result

    def correct(
        self,
        *,
        user_id: str,
        belief_id: int,
        new_claim_text: str,
        correction_note: str = "",
        new_predicate: str | None = None,
        new_object: str | None = None,
        new_value: dict | None = None,
        purpose: str | None = None,
    ) -> dict | None:
        """User-initiated correction: supersede a belief with corrected text.

        Unlike a normal UPDATE (driven by IdentityResolver), this is an
        explicit user action: "this memory is wrong, it should be X."

        Returns the new corrected belief as a dict, or ``None`` if the
        target is invalid.

        Raises
        ------
        ValidationError
            If user_id or new_claim_text is empty.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")
        if not new_claim_text or not new_claim_text.strip():
            raise ValidationError("new_claim_text must be a non-empty string")

        self._ensure_db()
        from mirror_memory.core.proposal import (
            TRANSITION_CORRECT,
            StateTransitionProposal,
        )
        from mirror_memory.core.publisher import Publisher
        from mirror_memory.core.repository import get_state_revision

        with self._session() as session:
            authorized = self._authorize(session, user_id, purpose, "correct")
            rule = provenance.visibility(session, user_id, authorized.purpose)
            if not provenance.belief_is_visible(session, user_id, rule, belief_id):
                # identical to the not-found return; no events, no writes
                return None
            decision = Publisher(session, self._config).publish(
                StateTransitionProposal(
                    transition=TRANSITION_CORRECT,
                    user_id=user_id,
                    target_belief_id=belief_id,
                    expected_revision=get_state_revision(session, user_id),
                    actor_type="user",
                    payload={
                        "new_claim_text": new_claim_text,
                        "correction_note": correction_note,
                        "new_predicate": new_predicate,
                        "new_object": new_object,
                        "new_value": new_value,
                    },
                )
            )
            session.commit()
            if not decision.committed:
                return None
            corrected_id = (decision.detail or {}).get("belief_id")
            if corrected_id is None:
                return None
            from mirror_memory.core.models import Belief

            row = session.get(Belief, corrected_id)
            if row is None:
                return None
            return {
                "belief_id": row.id,
                "claim_text": row.claim_text,
                "status": row.status,
                "confidence": row.confidence,
            }

    def export_page(
        self,
        *,
        user_id: str,
        purpose: str | None = None,
        cursor: str | None = None,
        page_size: int = 500,
    ) -> "ExportPage":
        """One page of the user's export (M3 batch-2 task 3): beliefs, then
        evidence, then ledger events, each with its source purpose; schema
        ``mm-export/1``.  Pass the page's ``next_cursor`` back to get the
        next one; ``None`` means the export is complete.

        Every page is its own transaction and AUTHORIZES AGAIN for the
        ``export`` operation before reading -- a grant revoked mid-export
        refuses the next page -- and writes one access-audit row with the
        ids it returned.  Pages are not one consistent snapshot: each
        carries the memory state ``revision`` so a consumer can detect a
        change between pages.  Needs the ``export`` operation in the grant.
        """
        from mirror_memory.core.export import ExportPage, read_page, now_iso
        from mirror_memory.core.repository import get_state_revision

        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")
        self._ensure_db()
        with self._session() as session:
            authorized = self._authorize(session, user_id, purpose, "export", audit_entry=ENTRY_EXPORT_PAGE)
            rule = provenance.visibility(session, user_id, authorized.purpose)
            section, items, ids, next_cursor = read_page(
                session, user_id, cursor=cursor, page_size=page_size, visibility=rule,
            )
            revision = get_state_revision(session, user_id)
            self._audit(session, authorized, user_id=user_id, operation="export",
                        entry=ENTRY_EXPORT_PAGE, memory_ids=ids)
            session.commit()
        return ExportPage(
            schema_version="mm-export/1",
            user_id=user_id,
            section=section,
            items=items,
            next_cursor=next_cursor,
            revision=revision,
            exported_at=now_iso(),
            authorization={"purpose": authorized.purpose, "grant_version": authorized.grant_version},
        )

    def export(
        self, *, user_id: str, purpose: str | None = None, page_size: int = 500,
    ) -> "Iterator[ExportPage]":
        """The whole export as a stream of pages (see :meth:`export_page`):
        memory bounded by *page_size*, re-authorized before every page, one
        audit row per page.  Stops when a page reports no next cursor."""
        from mirror_memory.core import access_audit

        cursor: str | None = None
        while True:
            with access_audit.entry_label(ENTRY_EXPORT):
                page = self.export_page(user_id=user_id, purpose=purpose, cursor=cursor, page_size=page_size)
            yield page
            if page.next_cursor is None:
                return
            cursor = page.next_cursor

    def belief_history(self, *, user_id: str, belief_id: int, purpose: str | None = None) -> list[dict]:
        """Return the version chain behind a belief, oldest first.

        The versioned ledger is the authoritative history: one identity per
        slot, one version per value that slot has held, with the truth
        interval and the belief interval recorded separately and the
        proposal that opened each interval attached.  A→B→A is three
        versions of one identity — the first stay is history that stays
        closed, not a row to reopen.

        Returns an empty list when the belief is not found.

        Raises
        ------
        ValidationError
            If user_id is empty.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")

        self._ensure_db()
        from mirror_memory.core.repository import belief_history

        with self._session() as session:
            authorized = self._authorize(session, user_id, purpose, "explain", audit_entry=ENTRY_BELIEF_HISTORY)
            rule = provenance.visibility(session, user_id, authorized.purpose)
            history = belief_history(session, user_id, belief_id, visibility=rule)
            self._audit(session, authorized, user_id=user_id, operation="explain",
                        entry=ENTRY_BELIEF_HISTORY, memory_ids=[belief_id] if history else [])
            session.commit()
            return history

    def explain(self, *, user_id: str, belief_id: int, purpose: str | None = None) -> dict | None:
        """Return the provenance chain for a belief.

        Returns a dict with belief state, event history, evidence sources,
        and supersession chain.  Returns ``None`` if the belief is not found.

        Raises
        ------
        ValidationError
            If user_id is empty.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")

        self._ensure_db()
        from mirror_memory.core.repository import explain_belief

        with self._session() as session:
            authorized = self._authorize(session, user_id, purpose, "explain", audit_entry=ENTRY_EXPLAIN)
            rule = provenance.visibility(session, user_id, authorized.purpose)
            explained = explain_belief(session, user_id, belief_id, visibility=rule)
            if explained is not None:
                from mirror_memory.core.export import source_purposes_for_beliefs

                # M3 batch-2 task 3: who was allowed to see this, and under
                # which purposes the evidence behind it was collected
                explained["authorization"] = {
                    "purpose": authorized.purpose,
                    "grant_version": authorized.grant_version,
                }
                explained["evidence_source_purposes"] = source_purposes_for_beliefs(
                    session, user_id, [belief_id],
                ).get(belief_id, [])
            self._audit(session, authorized, user_id=user_id, operation="explain",
                        entry=ENTRY_EXPLAIN, memory_ids=[belief_id] if explained is not None else [])
            session.commit()
            return explained
