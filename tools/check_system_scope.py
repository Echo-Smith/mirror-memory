"""CI/static check (M1 batch-2 task 1b): every system_scope call site in
engine code must be registered in tenancy_system_scope_allowlist.yaml,
and every allowlist entry must still have a matching call site (no ghost
entries).  Exits non-zero on any drift.

Usage: python tools/check_system_scope.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "mirror_memory"


def find_call_sites() -> set[tuple[str, str]]:
    sites: set[tuple[str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC.parent).with_suffix("")
        module = ".".join(rel.parts)
        if module == "mirror_memory.core.tenancy":
            continue  # definition site
        tree = ast.parse(path.read_text(encoding="utf-8"))
        stack: list[str] = ["<module>"]
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if stack and stack[-1] != node.name:
                    pass
        # walk with parents (simple recursive visitor)
        def visit(node: ast.AST, func: str) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit(child, child.name)
                else:
                    if isinstance(child, ast.Call):
                        f = child.func
                        name = None
                        if isinstance(f, ast.Name) and f.id == "system_scope":
                            name = "system_scope"
                        elif (
                            isinstance(f, ast.Attribute)
                            and f.attr == "system_scope"
                        ):
                            name = "system_scope"
                        if name:
                            sites.add((module, func))
                    visit(child, func)

        visit(tree, "<module>")
    return sites


def main() -> int:
    sys.path.insert(0, str(REPO / "src"))
    from mirror_memory.core.scope_registry import registry

    allowed = set(registry().keys())
    found = find_call_sites()
    unregistered = found - allowed
    ghosts = allowed - found
    ok = True
    for module, func in sorted(unregistered):
        print(f"UNREGISTERED: {module}.{func} calls system_scope")
        ok = False
    for module, func in sorted(ghosts):
        print(f"GHOST ENTRY: allowlist {module}.{func} has no call site")
        ok = False
    if ok:
        print(f"system_scope allowlist OK: {len(found)} call site(s) all registered")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
