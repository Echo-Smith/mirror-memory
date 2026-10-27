"""CI/static check (M2 task 4): sanctioned write functions may only be
called from the modules registered in ``tenancy_writer_callers.yaml``.

A function decorated with ``@_sanctioned_writer`` opens a write-class
scope, which is the only way engine code can write a classified tenant
table.  Calling one is therefore a privileged action: the decorator is the
capability, and this scan is what keeps the set of capabilities' holders
visible.

Two directions fail, both fatal:

- an unregistered caller -- a write path nobody reviewed;
- a ghost entry -- a registered module that no longer calls the function.
  Ghosts hide drift: the list stops describing reality and starts
  describing history.

Usage: python tools/check_writer_callers.py [--list]
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "mirror_memory"
REGISTRY = REPO / "tenancy_writer_callers.yaml"

#: The decorator that turns a function into a capability holder.
DECORATOR = "_sanctioned_writer"


class RegistryError(RuntimeError):
    """The caller registry is missing or malformed."""


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(SRC.parent).with_suffix("").parts)


def sanctioned_writers() -> dict[str, str]:
    """Map decorated function name -> the file that defines it."""
    found: dict[str, str] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                name = getattr(target, "id", None) or getattr(target, "attr", None)
                if name == DECORATOR:
                    found[node.name] = _module_name(path)
    return found


def call_sites(writers: set[str]) -> dict[str, set[str]]:
    """Map writer name -> modules that call it.

    A call is matched by the attribute or bare name used at the call site,
    so ``repository.record_evidence(...)`` and a locally imported
    ``record_evidence(...)`` both count.  The definition module itself is
    included when it calls the function internally.
    """
    sites: dict[str, set[str]] = {name: set() for name in writers}
    for path in sorted(SRC.rglob("*.py")):
        module = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name in writers:
                sites[name].add(module)
    return sites


def load_registry() -> dict[str, set[str]]:
    if not REGISTRY.exists():
        raise RegistryError(
            "tenancy_writer_callers.yaml not found at the repository root"
        )
    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8")) or {}
    writers = data.get("writers") or {}
    registry: dict[str, set[str]] = {}
    for name, spec in writers.items():
        if not isinstance(spec, dict):
            raise RegistryError("writer entry " + repr(name) + " must be a mapping")
        callers = spec.get("callers") or []
        if not callers:
            raise RegistryError(
                "writer entry " + repr(name) + " lists no callers; either the "
                "function is dead or the entry is unfinished"
            )
        registry[name] = set(callers)
    return registry


def audit() -> tuple[list[str], list[str], list[str], dict[str, set[str]]]:
    """Return (unregistered callers, ghost entries, unlisted writers, sites)."""
    writers = set(sanctioned_writers())
    registry = load_registry()
    sites = call_sites(writers)

    unregistered: list[str] = []
    ghosts: list[str] = []
    unlisted: list[str] = []

    for name in sorted(writers):
        allowed = registry.get(name)
        if allowed is None:
            unlisted.append(name)
            continue
        for module in sorted(sites.get(name, set()) - allowed):
            unregistered.append(module + "." + name)

    for name in sorted(registry):
        if name not in writers:
            ghosts.append(name + " (function is no longer a sanctioned writer)")
            continue
        for module in sorted(registry[name] - sites.get(name, set())):
            ghosts.append(module + "." + name)

    return unregistered, ghosts, unlisted, sites


def main(argv: list[str]) -> int:
    unregistered, ghosts, unlisted, sites = audit()

    if "--list" in argv:
        for name in sorted(sites):
            callers = ", ".join(sorted(sites[name])) or "(none)"
            print(name + ": " + callers)
        return 0

    ok = True
    for entry in unregistered:
        print("UNREGISTERED CALLER: " + entry + " is not in tenancy_writer_callers.yaml")
        ok = False
    for entry in ghosts:
        print("GHOST ENTRY: " + entry + " no longer exists")
        ok = False
    for name in unlisted:
        print("UNLISTED WRITER: " + name + " has no registry entry")
        ok = False

    if ok:
        total = sum(len(c) for c in sites.values())
        print(
            "writer callers OK: " + str(len(sites)) + " sanctioned writer(s), "
            + str(total) + " registered call site(s)"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
