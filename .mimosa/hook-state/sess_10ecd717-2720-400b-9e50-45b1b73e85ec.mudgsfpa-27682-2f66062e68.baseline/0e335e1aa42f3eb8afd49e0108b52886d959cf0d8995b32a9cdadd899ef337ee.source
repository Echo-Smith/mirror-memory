"""Run manifest — makes every benchmark score attributable.

The plan's Phase 0 acceptance criterion is "每个分数都能追溯到具体 run、case
和配置".  Without a manifest, a score is a number with no provenance: two runs
that differ only in model version, prompt text, or config flag are
indistinguishable afterwards.

The manifest captures everything needed to tell them apart, and is written
next to the report so the two travel together.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


def _hash_file(path: str | Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return "missing"


def _hash_text(text: str | None) -> str:
    if not text:
        return "empty"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class RunManifest:
    """Everything needed to reproduce or attribute one benchmark run.

    ``dirty`` matters as much as ``commit``: an uncommitted tree means the
    commit hash alone does not identify the code that produced the score.
    """

    run_id: str
    created_at: str
    commit: str
    dirty: bool
    python: str
    platform: str
    benchmark: str
    dataset_path: str = ""
    dataset_sha256: str = ""
    cases: int = 0
    categories: list[str] = field(default_factory=list)
    answerer_model: str = ""
    extractor_model: str = ""
    base_url: str = ""
    thinking: str = ""
    prompt_hashes: dict[str, str] = field(default_factory=dict)
    config_hashes: dict[str, str] = field(default_factory=dict)
    generation_params: dict[str, Any] = field(default_factory=dict)
    engine_version: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        benchmark: str,
        dataset_path: str | Path | None = None,
        cases: int = 0,
        categories: list[str] | None = None,
        answerer_model: str = "",
        extractor_model: str = "",
        base_url: str = "",
        thinking: str = "",
        config_dir: str | Path = "config/",
        generation_params: dict[str, Any] | None = None,
        notes: str = "",
    ) -> RunManifest:
        import mirror_memory

        base = Path(config_dir)
        prompt_hashes = {
            p.stem: _hash_file(p)
            for p in sorted((base / "prompts").glob("*.txt"))
        }
        config_hashes = {
            p.name: _hash_file(p)
            for p in sorted(base.glob("*.yaml"))
        }

        return cls(
            run_id=run_id,
            created_at=datetime.now(UTC).isoformat(),
            commit=_git_commit(),
            dirty=_git_dirty(),
            python=sys.version.split()[0],
            platform=platform.platform(),
            benchmark=benchmark,
            dataset_path=str(dataset_path or ""),
            dataset_sha256=_hash_file(dataset_path) if dataset_path else "",
            cases=cases,
            categories=sorted(categories or []),
            answerer_model=answerer_model,
            extractor_model=extractor_model,
            base_url=base_url,
            thinking=thinking,
            prompt_hashes=prompt_hashes,
            config_hashes=config_hashes,
            generation_params=generation_params or {},
            engine_version=getattr(mirror_memory, "__version__", ""),
            notes=notes,
        )


def write_manifest(manifest: RunManifest, path: str | Path) -> Path:
    """Write the manifest as JSON next to the report."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def manifest_path_for(report_path: str | Path) -> Path:
    """The manifest path that belongs to a given report path."""
    report = Path(report_path)
    return report.with_name(f"{report.stem}.manifest.json")
