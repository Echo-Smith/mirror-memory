#!/usr/bin/env python3
"""M4 batch-2 patch 2, task 4: the observe() microbenchmark.

Measures ``MemoryEngine.observe()`` end to end on SQLite -- extraction
included, so every round that produces a belief also performs the M4
global_id write (the uuid minted at creation and its unique index) --
``--observes`` times, and reports wall-clock milliseconds with at least
3 significant digits (total and per observe).

The engine is built exactly like the tests build it (config/ + the
deterministic empty-pattern LLM stub + memory enabled for the user), so
the measured path is the product path, not a harness shortcut.  The
script asserts beliefs were actually written -- a run with zero global_id
writes is invalid, not fast.

Usage (repo root; ``src`` may be at either measured revision):
    python tools/perf_observe_micro.py --observes 2000
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path


class _FakeLLM:
    def generate(self, **kw):
        return '{"patterns": [], "support_policy": {}}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observes", type=int, default=2000)
    parser.add_argument("--user", default="micro")
    parser.add_argument("--out", default=None,
                        help="append the result line to this file")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="mm-micro-"))
    try:
        from mirror_memory.api import MemoryEngine
        from mirror_memory.config.loader import load_config
        from mirror_memory.core.repository import set_memory_enabled
        from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope

        cfg = load_config("config/")
        cfg.llm_client = _FakeLLM()
        cfg.worker.job_lease_seconds = 60
        cfg.session_summary_enabled = True
        cfg.deletion_log.directory = str(tmp / "deletions")
        engine = MemoryEngine(
            config=cfg, database_url=f"sqlite:///{tmp / 'micro.db'}",
            tenant_id="default", app_id="default", llm_client=_FakeLLM(),
            single_tenant_mode=True)
        engine._ensure_db()
        with tenant_scope("default", "default"), \
                fixture_write_scope("microbenchmark: enable memory"):
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker

            from mirror_memory.core import authorization

            s = sessionmaker(bind=create_engine(
                f"sqlite:///{tmp / 'micro.db'}"))()
            set_memory_enabled(s, args.user, True)
            # the M3 purpose layer is fail-closed: observe('api') needs a
            # real grant (setup, not measured)
            authorization.grant_purpose(
                s, args.user, "api",
                operations=list(authorization.OPERATIONS),
                issuer="perf-microbenchmark",
                reason="observe microbenchmark grant")
            s.commit()
            s.close()

        # one DISTINCT preference text per observe: the deterministic
        # extraction layer mints a belief for every observe, so each of
        # the measured calls performs the M4 global_id write this
        # microbenchmark is about
        texts = [f"I like microtopic{i} and coffee."
                 for i in range(args.observes)]

        t0 = time.perf_counter()
        for i in range(args.observes):
            engine.observe(user_id=args.user,
                           session_id=f"s{i % 10}",
                           text=texts[i],
                           purpose="api")
        elapsed = time.perf_counter() - t0

        from sqlalchemy import create_engine as _ce
        from sqlalchemy.orm import sessionmaker as _sm

        from mirror_memory.core.models import Belief
        from mirror_memory.core.tenancy import tenant_scope as _ts

        with _ts("default", "default"):
            s = _sm(bind=_ce(f"sqlite:///{tmp / 'micro.db'}"))()
            written = s.query(Belief).filter(
                Belief.user_id == args.user).count()
            with_global = s.query(Belief).filter(
                Belief.user_id == args.user,
                Belief.global_id != "").count()
            s.close()
        assert written > 0, "no beliefs written -- no global_id writes: " \
            "the run is invalid"
        assert with_global == written, \
            f"{written - with_global} beliefs without a global_id"

        total_ms = elapsed * 1000.0
        per_ms = total_ms / args.observes
        line = (f"observes={args.observes} beliefs={written} "
                f"total_ms={total_ms:.4g} ms_per_observe={per_ms:.4g}")
        print(line)
        if args.out:
            Path(args.out).write_text(line + "\n", encoding="utf-8")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
