"""M3 batch-1 patch task 1: src must stay free of auto-grant hooks.

An env-var auto-grant shipped briefly and was removed as an authorization
hole (a config flip would turn "no grant" into "grant", the fail-open
shape the merge plan rejects).  These tests run the static checker
against the live src tree and against a counterexample, so the CI gate
cannot silently rot.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run_checker(*args):
    return subprocess.run(
        [sys.executable, str(REPO / "tools" / "check_no_autogrant.py"), *args],
        capture_output=True, text=True,
    )


def test_live_src_is_clean():
    result = _run_checker()
    assert result.returncode == 0, result.stdout + result.stderr


def test_counterexample_env_read_is_caught(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "hole.py").write_text(
        'import os\n\n\ndef maybe_grant() -> bool:\n'
        '    return os.environ.get("MM_TEST_AUTOGRANT", "0") == "1"\n',
        encoding="utf-8",
    )
    result = _run_checker("--src", str(src))
    assert result.returncode == 1
    assert "AUTOGRANT" in result.stdout.upper()  # banned token or env key, either is a catch


def test_counterexample_identifier_is_caught(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "renamed.py").write_text(
        "def _autogrant(session, user_id, purpose):\n    return None\n",
        encoding="utf-8",
    )
    result = _run_checker("--src", str(src))
    assert result.returncode == 1
    assert "autogrant" in result.stdout.lower()


def test_benign_env_read_is_not_flagged(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "settings.py").write_text(
        'import os\n\nPOOL = os.environ.get("MM_PG_POOL_SIZE", "5")\n',
        encoding="utf-8",
    )
    result = _run_checker("--src", str(src))
    assert result.returncode == 0, result.stdout
