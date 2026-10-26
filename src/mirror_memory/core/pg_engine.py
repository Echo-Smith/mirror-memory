"""PostgreSQL engine construction: pool sizing and per-connection timeouts
(M2 batch-2 task 2).

Every PostgreSQL engine the library opens -- the server's shared engine,
a host's MemoryEngine, the maintenance engine -- goes through
:func:`build_engine`, so the pool bounds and the three server-side
timeouts are one configuration surface.  SQLite engines are untouched:
the end-side shape has no pool to size and no server to time out.

Pool (``MM_PG_POOL_SIZE`` / ``MM_PG_MAX_OVERFLOW`` / ``MM_PG_POOL_PRE_PING``
/ ``MM_PG_POOL_RECYCLE``; defaults 5 / 10 / on / 1800 s):

- 5 + 10 is SQLAlchemy's own default and caps one process at 15 server
  connections, leaving room for several processes plus the maintenance
  pools under PostgreSQL's default ``max_connections = 100``.
- pre-ping costs one trivial round trip per checkout and turns a
  connection the server (or a proxy idle timer) dropped into a
  transparent reconnect instead of a failed request.
- recycle at 30 minutes keeps connections younger than common NAT/proxy
  idle limits and lets server-side role or parameter changes reach the
  pool without a restart.

Timeouts, set through libpq ``options`` at connect time (no extra round
trip; ``MM_PG_STATEMENT_TIMEOUT_MS`` / ``MM_PG_LOCK_TIMEOUT_MS`` /
``MM_PG_IDLE_IN_TRANSACTION_TIMEOUT_MS``):

- ``statement_timeout`` 30 s: observe/recall statements run in
  milliseconds; a statement still running after 30 s is a runaway query
  or a lock wait that will not resolve, and with at most 15 pooled
  connections a few of those pin the whole pool.  Cancelling it returns
  the connection; the transaction rolls back cleanly.
- ``lock_timeout`` 5 s: row-lock contention on claim/publish resolves in
  milliseconds; 5 s bounds queueing behind a stuck writer.  Migrations
  set their own (``MM_MIGRATION_LOCK_TIMEOUT``).
- ``idle_in_transaction_session_timeout`` 60 s: an idle open transaction
  holds its locks and blocks VACUUM; the engine's sessions are
  request-scoped, so 60 s is far beyond any legitimate request yet short
  enough to free a leaked session before it hurts the cluster.
"""

from __future__ import annotations

import os

from sqlalchemy.engine import Engine, make_url

APPLICATION_NAME = "mirror-memory"

POOL_DEFAULTS: dict[str, object] = {
    "pool_size": 5,
    "max_overflow": 10,
    "pool_pre_ping": True,
    "pool_recycle": 1800,
}

TIMEOUT_DEFAULTS_MS: dict[str, int] = {
    "statement_timeout": 30_000,
    "lock_timeout": 5_000,
    "idle_in_transaction_session_timeout": 60_000,
}

_POOL_ENV = {
    "pool_size": "MM_PG_POOL_SIZE",
    "max_overflow": "MM_PG_MAX_OVERFLOW",
    "pool_pre_ping": "MM_PG_POOL_PRE_PING",
    "pool_recycle": "MM_PG_POOL_RECYCLE",
}

_TIMEOUT_ENV = {
    "statement_timeout": "MM_PG_STATEMENT_TIMEOUT_MS",
    "lock_timeout": "MM_PG_LOCK_TIMEOUT_MS",
    "idle_in_transaction_session_timeout": "MM_PG_IDLE_IN_TRANSACTION_TIMEOUT_MS",
}


class EngineConfigError(ValueError):
    """An engine setting from the environment is malformed."""


def _int_env(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise EngineConfigError(f"{name} must be an integer; got {raw!r}") from exc
    if value < minimum:
        raise EngineConfigError(f"{name} must be >= {minimum}; got {value}")
    return value


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if raw.lower() in ("1", "true", "yes", "on"):
        return True
    if raw.lower() in ("0", "false", "no", "off"):
        return False
    raise EngineConfigError(f"{name} must be a boolean (1/0, true/false); got {raw!r}")


def pool_settings() -> dict[str, object]:
    return {
        "pool_size": _int_env(_POOL_ENV["pool_size"], POOL_DEFAULTS["pool_size"], minimum=1),
        "max_overflow": _int_env(_POOL_ENV["max_overflow"], POOL_DEFAULTS["max_overflow"], minimum=0),
        "pool_pre_ping": _bool_env(_POOL_ENV["pool_pre_ping"], POOL_DEFAULTS["pool_pre_ping"]),
        "pool_recycle": _int_env(_POOL_ENV["pool_recycle"], POOL_DEFAULTS["pool_recycle"], minimum=-1),
    }


def timeout_settings_ms() -> dict[str, int]:
    """Milliseconds per setting; 0 disables that timeout (PostgreSQL semantics)."""
    return {
        key: _int_env(env, TIMEOUT_DEFAULTS_MS[key], minimum=0)
        for key, env in _TIMEOUT_ENV.items()
    }


def connect_options(timeouts: dict[str, int] | None = None) -> str:
    """The libpq ``options`` string carrying the three timeouts.

    Each value is an integer validated above; the setting names are
    module constants -- nothing from the environment reaches the string
    except the validated integers.
    """
    timeouts = timeouts if timeouts is not None else timeout_settings_ms()
    return " ".join(
        "-c %s=%d" % (name, int(timeouts[name])) for name in TIMEOUT_DEFAULTS_MS
    )


def is_postgres_url(url) -> bool:
    try:
        return make_url(url).get_backend_name() == "postgresql"
    except Exception:  # noqa: BLE001 -- not a URL
        return False


def build_engine(url, *, application_name: str = APPLICATION_NAME, **overrides) -> Engine:
    """Create an engine with the library's pool and timeout settings.

    *overrides* win over the environment (tests, the maintenance engine's
    small pool).  Non-PostgreSQL URLs get a plain ``create_engine`` so the
    SQLite shape is byte-for-byte what it was.
    """
    import sqlalchemy

    if not is_postgres_url(url):
        return sqlalchemy.create_engine(url, **overrides)
    kwargs = dict(pool_settings())
    kwargs.update(overrides)
    connect_args = dict(kwargs.pop("connect_args", {}) or {})
    connect_args.setdefault("application_name", application_name)
    connect_args.setdefault("options", connect_options())
    return sqlalchemy.create_engine(url, connect_args=connect_args, **kwargs)
