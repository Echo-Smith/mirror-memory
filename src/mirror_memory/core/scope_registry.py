"""Scope allowlist registry (M1 batch-2 task 1b).

Single source of truth shared by the runtime ``system_scope`` check, the
CI AST scan (tools/check_system_scope.py), and tests: the allowlist file
``tenancy_system_scope_allowlist.yaml`` at the repository root.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ScopeAllowlistViolation(AssertionError):
    """An engine-code system_scope call site is not registered."""


def _load_allowlist() -> dict[tuple[str, str], dict]:
    import yaml

    candidates = [
        Path(__file__).resolve().parents[3] / "tenancy_system_scope_allowlist.yaml",
        Path.cwd() / "tenancy_system_scope_allowlist.yaml",
    ]
    for path in candidates:
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
            return {(e["module"], e["function"]): e for e in data}
    raise FileNotFoundError(
        "tenancy_system_scope_allowlist.yaml not found (looked at repo root "
        "and CWD); every system_scope call site must be registered there"
    )


_registry: dict[tuple[str, str], dict] | None = None


def registry() -> dict[tuple[str, str], dict]:
    global _registry
    if _registry is None:
        _registry = _load_allowlist()
    return _registry


def reload_registry() -> dict[tuple[str, str], dict]:
    global _registry
    _registry = _load_allowlist()
    return _registry


def assert_scope_allowed(caller: tuple[str, str], *, strict: bool) -> None:
    """Raise (strict) or warn when *caller* (module, function) is not in
    the allowlist."""
    entries = registry()
    if caller in entries:
        return
    module, func = caller
    # a registered function whose module moved/renamed is still a drift
    message = (
        f"system_scope called from unregistered engine site "
        f"{module}.{func}; register it in tenancy_system_scope_allowlist.yaml "
        f"(known: {sorted(f'{m}.{f}' for m, f in entries)})"
    )
    if strict:
        raise ScopeAllowlistViolation(message)
    logger.warning("system_scope allowlist: %s", message)
