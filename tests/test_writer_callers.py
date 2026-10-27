"""M2 task 4 -- sanctioned write functions may only be called by their
registered callers.

The ``@_sanctioned_writer`` decorator opens a write-class scope, and that
scope is the only way engine code can write a classified tenant table.  A
caller is therefore a capability holder; ``tenancy_writer_callers.yaml`` is
the list of holders and ``tools/check_writer_callers.py`` is the gate.

The counterexample is the point of the test: a module that calls
``record_evidence`` without being registered must fail the scan.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "check_writer_callers.py"
REGISTRY = REPO / "tenancy_writer_callers.yaml"


def _load_tool():
    spec = importlib.util.spec_from_file_location("check_writer_callers", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["check_writer_callers"] = module
    spec.loader.exec_module(module)
    return module


def test_registry_file_exists_and_parses():
    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    writers = data["writers"]
    assert writers, "the registry must list at least one writer"
    for name, spec in writers.items():
        assert spec.get("reason", "").strip(), name + " has no reason"
        assert spec.get("callers"), name + " lists no callers"


def test_every_sanctioned_writer_is_registered():
    tool = _load_tool()
    writers = set(tool.sanctioned_writers())
    registry = set(tool.load_registry())
    assert writers - registry == set(), "unregistered sanctioned writers"
    assert registry - writers == set(), "ghost registry entries"


def test_scan_is_clean_on_the_current_tree():
    tool = _load_tool()
    unregistered, ghosts, unlisted, sites = tool.audit()
    assert unregistered == [], unregistered
    assert ghosts == [], ghosts
    assert unlisted == [], unlisted
    assert sites, "the scan found no call sites at all -- check the walker"


def test_counterexample_temp_module_calling_record_evidence_fails(tmp_path, monkeypatch):
    """A new module that calls a sanctioned writer, unregistered, must fail.

    This is the counterexample the task requires: the check cannot be
    satisfied by adding code that writes memory content from somewhere new.
    """
    tool = _load_tool()

    # The writer set stays the real one; only the walked tree changes, so the
    # rogue module appears as a call site the registry does not know.
    writers = set(tool.sanctioned_writers())
    rogue_root = tmp_path / "src"
    rogue = rogue_root / "mirror_memory" / "rogue" / "sneaky.py"
    rogue.parent.mkdir(parents=True)
    rogue.write_text(
        "from mirror_memory.core.repository import record_evidence\n"
        "\n"
        "\n"
        "def inject(session, user_id, ref):\n"
        "    return record_evidence(session, user_id=user_id, ref=ref, content='x')\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(tool, "SRC", rogue_root / "mirror_memory")
    sites = tool.call_sites(writers)

    assert "mirror_memory.rogue.sneaky" in sites["record_evidence"]
    allowed = tool.load_registry()["record_evidence"]
    unregistered = sorted(sites["record_evidence"] - allowed)
    assert unregistered == ["mirror_memory.rogue.sneaky"]


def test_counterexample_ghost_entry_fails(tmp_path, monkeypatch):
    """A registry entry whose caller stopped calling must also fail: a list
    that overstates reality stops being a description of the code."""
    tool = _load_tool()

    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    registry["writers"]["record_evidence"]["callers"].append(
        "mirror_memory.core.publisher"
    )
    patched = tmp_path / "tenancy_writer_callers.yaml"
    patched.write_text(yaml.safe_dump(registry), encoding="utf-8")
    monkeypatch.setattr(tool, "REGISTRY", patched)

    unregistered, ghosts, _unlisted, _sites = tool.audit()
    assert any("mirror_memory.core.publisher.record_evidence" in g for g in ghosts)
    assert unregistered == []


def test_counterexample_unlisted_writer_fails(tmp_path, monkeypatch):
    """A new ``_sanctioned_writer`` function with no registry entry is an
    unreviewed capability and must fail the scan."""
    tool = _load_tool()

    extra = tmp_path / "mirror_memory"
    extra.mkdir()
    (extra / "newpath.py").write_text(
        "def _sanctioned_writer(cls):\n"
        "    def deco(fn):\n"
        "        return fn\n"
        "    return deco\n"
        "\n"
        "\n"
        "@_sanctioned_writer('engine')\n"
        "def write_something(session):\n"
        "    return None\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(tool, "SRC", extra)
    unregistered, _ghosts, unlisted, _sites = tool.audit()
    assert "write_something" in unlisted


def test_missing_registry_is_a_hard_error(tmp_path, monkeypatch):
    tool = _load_tool()
    monkeypatch.setattr(tool, "REGISTRY", tmp_path / "absent.yaml")
    with pytest.raises(tool.RegistryError):
        tool.load_registry()
