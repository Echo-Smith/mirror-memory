"""M4 batch-2 patch 2, task 3: the src tree carries no debug output
code.

Two guards over ``src/mirror_memory``:

- no trace-marker debug switches (``MESH_TRACE`` / ``export_trace`` --
  the M4 sync-debugging leftovers) anywhere in src;
- no ``open(...)`` / ``Path(...)``, ``os.mkdir``-style LITERAL write into
  a fixed ``/tmp``-style path (a literal ``"/tmp/..."`` string as the
  path argument of a writing call).  Product code that derives a
  functional path from a variable (the replay lock file, tempfile
  usage, config-driven directories) is untouched -- the guard matches
  literals, so it cannot flag those.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "mirror_memory"

TRACE_MARKERS = ("MESH_TRACE", "export_trace")

_TMP_PREFIXES = ("/tmp/", "/var/tmp/", "/dev/shm")
_WRITE_FLAGS = ("w", "a", "x", "+")


def _py_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _literal_tmp_path(node: ast.AST) -> str | None:
    """The literal ``/tmp...`` string a call names, if any (plain str
    constants and f-strings whose FIRST part is such a constant)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value if node.value.startswith(_TMP_PREFIXES) else None
    if isinstance(node, ast.JoinedStr) and node.values:
        first = node.values[0]
        if (isinstance(first, ast.Constant) and isinstance(first.value, str)
                and first.value.startswith(_TMP_PREFIXES)):
            return first.value
    return None


def _call_args(call: ast.Call) -> list[ast.AST]:
    """The call's arguments PLUS the receiver chain's arguments -- a
    ``Path("/tmp/x").write_text(...)`` names the path in the receiver."""
    args: list[ast.AST] = list(call.args)
    for kw in call.keywords:
        args.append(kw.value)
    receiver = call.func
    while isinstance(receiver, ast.Attribute):
        receiver = receiver.value
    if isinstance(receiver, ast.Call):
        args.extend(receiver.args)
        for kw in receiver.keywords:
            args.append(kw.value)
    return args


def _writes_literal_tmp(call: ast.Call) -> bool:
    """A call whose literal path argument is a fixed /tmp-style path AND
    which plausibly writes (an ``open`` with a writing mode, or a
    Path/mkdir/open-style call with that literal)."""
    name = ast.dump(call.func)
    path = None
    for arg in _call_args(call):
        path = path or _literal_tmp_path(arg)
    if path is None:
        return False
    if "open" in name:
        # mode: positional 2nd arg or the ``mode`` keyword
        for arg in call.args[1:]:
            lit = _literal_tmp_path(arg)
            if lit is None and isinstance(arg, ast.Constant) \
                    and isinstance(arg.value, str) \
                    and any(f in arg.value for f in _WRITE_FLAGS):
                return True
        for kw in call.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant) \
                    and isinstance(kw.value.value, str) \
                    and any(f in kw.value.value for f in _WRITE_FLAGS):
                return True
        return False  # open() without a writing literal mode: reads only
    # Path(...)/mkdir/write_text-ish calls naming a fixed /tmp path
    return True


def test_src_has_no_trace_marker_debug_code():
    """The MESH_TRACE / export_trace debug switches are gone from src."""
    for path in _py_files():
        text = path.read_text(encoding="utf-8")
        for marker in TRACE_MARKERS:
            assert marker not in text, (path, marker)


def test_src_writes_no_literal_tmp_debug_files():
    """No src code writes a LITERAL /tmp-style path (debug dumps)."""
    offenders: list[tuple[str, int, str]] = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _writes_literal_tmp(node):
                offenders.append((str(path), node.lineno, ast.dump(node)[:80]))
    assert offenders == [], offenders
