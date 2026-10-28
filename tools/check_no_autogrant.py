"""CI/static check (M3 batch-1 patch task 1): no auto-grant hooks in src.

The engine must never read an environment variable that can cause an
authorization to be issued.  An env-var auto-grant shipped briefly in M3
batch 1 (MM_TEST_AUTOGRANT) and was removed as an authorization hole:
any code path that turns "no grant" into "grant" from configuration
rather than an explicit grant_purpose call defeats the fail-closed model
and is exactly the v2 fail-open shape the merge plan rejects.

Two directions, both fatal:

- any ``os.environ`` / ``os.getenv`` / ``os.environ.get`` key in src
  matching the denylist (case-insensitive ``autogrant``);
- any ``autogrant`` identifier or string literal anywhere in src (the
  hook's name is banned outright -- a renamed revival should not sneak
  past the env-key scan).

Usage: python tools/check_no_autogrant.py [--src DIR]     # CI + pytest
       python tools/check_no_autogrant.py --selftest      # counterexample
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SRC = REPO / "src"

#: Environment keys that must never be read by engine code.
DENYLIST = re.compile(r"autogrant", re.IGNORECASE)
#: The hook's name is banned as an identifier or literal anywhere in src.
BANNED_TOKEN = re.compile(r"autogrant", re.IGNORECASE)


def env_keys_read(tree: ast.AST) -> list[str]:
    """Environment-key string literals read via os.environ[...],
    os.environ.get(...), os.getenv(...)."""
    keys: list[str] = []

    def is_os_attr(node: ast.AST, name: str) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == name
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
        )

    for node in ast.walk(tree):
        # os.environ["KEY"] / os.environ.get("KEY", ...)
        if isinstance(node, ast.Subscript) and is_os_attr(node.value, "environ"):
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.append(key.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if is_os_attr(node.func.value, "environ") and node.func.attr == "get":
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    keys.append(node.args[0].value)
            if is_os_attr(node.func, "getenv"):
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    keys.append(node.args[0].value)
    return keys


def violations(src_dir: Path) -> list[str]:
    found: list[str] = []
    for path in sorted(src_dir.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(src_dir.parent)
        hit = BANNED_TOKEN.search(text)
        if hit:
            found.append(f"{rel}: contains banned token {hit.group(0)!r}")
            continue  # one line per file is enough
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            found.append(f"{rel}: unparseable ({exc})")
            continue
        for key in env_keys_read(tree):
            if DENYLIST.search(key):
                found.append(f"{rel}: engine reads forbidden env var {key!r}")
    return found


def selftest() -> int:
    """Counterexample: a src-like tree that reads MM_TEST_AUTOGRANT must
    fail; a clean tree must pass."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        bad = root / "src" / "hole.py"
        bad.write_text(
            "import os\n\n\ndef maybe_grant() -> bool:\n"
            '    return os.environ.get("MM_TEST_AUTOGRANT", "0") == "1"\n',
            encoding="utf-8",
        )
        if not violations(root / "src"):
            print("SELFTEST FAIL: the counterexample was not caught")
            return 1
        good = root / "src" / "hole.py"
        good.write_text(
            'import os\n\n\nTIMEOUT = os.environ.get("MM_PG_STATEMENT_TIMEOUT_MS", "30000")\n',
            encoding="utf-8",
        )
        if violations(root / "src"):
            print("SELFTEST FAIL: a clean env read was flagged:", violations(root / "src"))
            return 1
    print("selftest ok: counterexample caught, clean env read passes")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if "--selftest" in argv:
        return selftest()
    src = Path(argv[argv.index("--src") + 1]) if "--src" in argv else DEFAULT_SRC
    found = violations(src)
    if found:
        print("AUTOGRANT HOOKS IN SRC (authorization hole -- remove them):")
        for line in found:
            print(f"  {line}")
        return 1
    print(f"no-autogrant check OK: no grant-issuing env hooks in {src.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
