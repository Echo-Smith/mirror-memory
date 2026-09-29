"""MetabolismBench — does the memory lifecycle preserve what matters?

StateBench scores whether the engine's *current state surface* is right.
MetabolismBench scores the long-run layer on top of it: after months of
simulated time, cooling, compaction, archiving, and deletion, is the memory
still correct where it must be, and gone where it must be?

Each case is a scripted life: conversation turns, then a sequence of
lifecycle operations against a simulated clock (``cycle`` advances time and
runs one metabolism cycle), then assertions on the rendered recall block
and on engine state.  Scoring is deterministic — the rendered block and the
database are the ground truth, no answer model is involved.  An LLM client
is needed only for *ingestion* (K2 semantic extraction), exactly as in
StateBench.

Two headline metrics, both reported per category and overall:

- **Semantic Preservation Rate** — of the conclusions answerable before a
  lifecycle operation (compaction / cooling / archiving), the fraction
  still answerable after it.
- **Storage Reduction Ratio** — evidence content characters after
  compaction divided by before; 1.0 means nothing was reclaimed.

Categories: current_survival, historical_preservation,
correction_protection, conflict_preservation, evidence_compaction,
reactivation, deletion, stale_worker.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from mirror_memory.core.utils import safe_json_list

# -- Case schema ---------------------------------------------------------------


@dataclass(frozen=True)
class MetabolismCase:
    """One scripted memory lifecycle."""

    case_id: str
    category: str
    turns: tuple[str, ...]
    steps: tuple[dict[str, Any], ...]
    note: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MetabolismCase":
        return cls(
            case_id=str(raw["case_id"]),
            category=str(raw["category"]),
            turns=tuple(str(t) for t in raw.get("turns", [])),
            steps=tuple(raw.get("steps", [])),
            note=str(raw.get("note", "")),
        )


def load_metabolism_cases(
    path: str | Path | None = None,
    *,
    categories: tuple[str, ...] | None = None,
) -> list[MetabolismCase]:
    """Load the packaged (or a custom) MetabolismBench case set."""
    if path is None:
        payload = json.loads(
            (
                Path(__file__).parent / "data" / "metabolism_v1.json"
            ).read_text(encoding="utf-8")
        )
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [MetabolismCase.from_dict(item) for item in payload["cases"]]
    if categories:
        cases = [c for c in cases if c.category in categories]
    return cases


# -- Result schema -------------------------------------------------------------


@dataclass
class CaseResult:
    case_id: str
    category: str
    passed: bool = False
    # Every assertion that ran, with its verdict — a failure is only
    # actionable if you can see *which* expectation broke.
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    # Semantic preservation: answerable-before vs answerable-after probes.
    probes_before: int = 0
    probes_after: int = 0
    # Storage: evidence content characters before vs after compaction.
    chars_before: int = 0
    chars_after: int = 0
    # Theoretical reduction for the same fold (1 - kept_sample / total
    # links): the keep policy's ceiling.  Efficiency divides the achieved
    # reduction by it, so "how well did we compact" is separable from
    # "how much was there to compact".
    theoretical_reduction: float | None = None
    recall_latency_ms: list[float] = field(default_factory=list)
    error: str = ""

    @property
    def semantic_preservation(self) -> float | None:
        if self.probes_before == 0:
            return None
        return round(self.probes_after / self.probes_before, 4)

    @property
    def storage_reduction(self) -> float | None:
        if self.chars_before == 0:
            return None
        return round(1.0 - self.chars_after / self.chars_before, 4)

    @property
    def compaction_efficiency(self) -> float | None:
        """Achieved reduction as a fraction of the keep policy's ceiling."""
        if self.chars_before == 0 or not self.theoretical_reduction:
            return None
        return round(min(self.storage_reduction or 0.0, 1.0) / self.theoretical_reduction, 4)


@dataclass
class BenchReport:
    results: list[CaseResult] = field(default_factory=list)
    model: str = ""
    base_url: str = ""
    degraded: bool = False  # True when no LLM client was supplied

    # -- aggregates ----------------------------------------------------------

    def by_category(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for result in self.results:
            bucket = out.setdefault(
                result.category,
                {"cases": 0, "passed": 0, "semantic": [], "storage": [], "efficiency": []},
            )
            bucket["cases"] += 1
            bucket["passed"] += int(result.passed)
            if result.semantic_preservation is not None:
                bucket["semantic"].append(result.semantic_preservation)
            if result.storage_reduction is not None:
                bucket["storage"].append(result.storage_reduction)
            if result.compaction_efficiency is not None:
                bucket["efficiency"].append(result.compaction_efficiency)
        for bucket in out.values():
            bucket["pass_rate"] = (
                round(bucket["passed"] / bucket["cases"], 4) if bucket["cases"] else 0.0
            )
            bucket["semantic_preservation"] = (
                round(statistics.fmean(bucket["semantic"]), 4) if bucket["semantic"] else None
            )
            bucket["storage_reduction"] = (
                round(statistics.fmean(bucket["storage"]), 4) if bucket["storage"] else None
            )
            bucket["compaction_efficiency"] = (
                round(statistics.fmean(bucket["efficiency"]), 4) if bucket["efficiency"] else None
            )
            del bucket["semantic"]
            del bucket["storage"]
            del bucket["efficiency"]
        return out

    @property
    def overall_pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(1 for r in self.results if r.passed) / len(self.results), 4)

    @property
    def semantic_preservation(self) -> float | None:
        values = [r.semantic_preservation for r in self.results if r.semantic_preservation is not None]
        return round(statistics.fmean(values), 4) if values else None

    @property
    def storage_reduction(self) -> float | None:
        values = [r.storage_reduction for r in self.results if r.storage_reduction is not None]
        return round(statistics.fmean(values), 4) if values else None

    @property
    def compaction_efficiency(self) -> float | None:
        values = [r.compaction_efficiency for r in self.results if r.compaction_efficiency is not None]
        return round(statistics.fmean(values), 4) if values else None

    @property
    def recall_latency_ms(self) -> dict[str, float] | None:
        samples = [ms for r in self.results for ms in r.recall_latency_ms]
        if not samples:
            return None
        samples.sort()
        return {
            "p50": round(statistics.median(samples), 2),
            "p95": round(samples[max(0, int(len(samples) * 0.95) - 1)], 2),
            "max": round(samples[-1], 2),
            "n": len(samples),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "degraded_k1_only": self.degraded,
            "cases": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "overall_pass_rate": self.overall_pass_rate,
            "semantic_preservation": self.semantic_preservation,
            "storage_reduction": self.storage_reduction,
            "compaction_efficiency": self.compaction_efficiency,
            "recall_latency_ms": self.recall_latency_ms,
            "by_category": self.by_category(),
            "failures": [
                {
                    "case_id": r.case_id,
                    "category": r.category,
                    "error": r.error,
                    "checks": [
                        {"check": name, "ok": ok, "detail": detail}
                        for name, ok, detail in r.checks
                        if not ok
                    ],
                }
                for r in self.results
                if not r.passed
            ],
        }


# -- Runner --------------------------------------------------------------------


def _evidence_chars(session, user_id: str) -> int:
    from mirror_memory.core.models import Evidence

    rows = session.scalars(
        __import__("sqlalchemy").select(Evidence).where(Evidence.user_id == user_id)
    )
    return sum(len(row.content or "") for row in rows)


def _find_beliefs(session, user_id: str, needle: str) -> list:
    """Beliefs whose key or claim_text contains *needle* (case-insensitive)."""
    from sqlalchemy import select

    from mirror_memory.core.models import Belief

    needle = needle.lower()
    rows = session.scalars(select(Belief).where(Belief.user_id == user_id))
    return [
        b
        for b in rows
        if needle in (b.key or "").lower() or needle in (b.claim_text or "").lower()
    ]


def _digest_for(session, belief_id: int):
    from sqlalchemy import select

    from mirror_memory.core.models import EvidenceDigest

    return session.scalar(select(EvidenceDigest).where(EvidenceDigest.belief_id == belief_id))


def _benchmark_config(config: Any, llm_client: Any | None) -> Any:
    """Force LLM extraction on every turn when a client is available.

    Mirrors StateBench's ``forced`` mode: the metabolism layer needs real
    beliefs to exist, and the production cost throttle (extract every N
    turns / after M keyword hits) would leave most scripted turns
    unextracted for reasons that have nothing to do with lifecycle
    behaviour.  Without a client the config passes through unchanged and
    the run is reported as K1-degraded.
    """
    if llm_client is None:
        return config
    extraction = config.extraction.model_copy(update={"llm_min_keyword_hits": 0})
    return config.model_copy(update={"extraction": extraction})


def run_case(
    case: MetabolismCase,
    config: Any,
    *,
    llm_client: Any | None = None,
) -> CaseResult:
    """Replay one case: ingest, operate, assert."""
    from mirror_memory.api import MemoryEngine

    result = CaseResult(case_id=case.case_id, category=case.category)
    user_id = f"mb-{case.case_id}"
    engine = MemoryEngine(
        config=_benchmark_config(config, llm_client),
        database_url="sqlite://",
        llm_client=llm_client,
    )
    clock = datetime.now(UTC)
    # Each observe is a distinct turn: the pipeline keys fallback evidence on
    # ``session:turn``, so re-using one index would collapse repeated
    # observations into a single evidence row.
    turn_index = 0

    def observe(text: str, session_id: str = "s1") -> None:
        nonlocal turn_index
        engine.observe(
            user_id=user_id, session_id=session_id, text=text, turn_count=turn_index
        )
        turn_index += 1

    def check(name: str, ok: bool, detail: str = "") -> bool:
        result.checks.append((name, bool(ok), detail))
        return bool(ok)

    def recall(query: str, *, at_clock: bool = True) -> str:
        started = time.monotonic()
        # Position the recall on the simulated timeline: access telemetry is
        # the reheat signal, and stamping it at real-now while the cycle
        # runs at the simulated clock would decay the signal away.
        block = engine.recall(
            user_id=user_id, query=query, now=clock if at_clock else None
        )
        result.recall_latency_ms.append((time.monotonic() - started) * 1000)
        return block or ""

    try:
        for turn in case.turns:
            observe(turn)

        for step in case.steps:
            op = step.get("op")

            if op == "cycle":
                clock = clock + timedelta(days=float(step.get("days", 30)))
                engine.run_metabolism(user_id=user_id, now=clock)

            elif op == "observe":
                observe(step["text"], session_id="s2")

            elif op == "recall":
                block = recall(step["query"])
                for expected in step.get("expect", []):
                    check(f"recall_contains:{expected}", expected.lower() in block.lower(),
                          f"query={step['query']!r} block={block[:160]!r}")
                for rejected in step.get("reject", []):
                    check(f"recall_excludes:{rejected}", rejected.lower() not in block.lower(),
                          f"query={step['query']!r} block={block[:160]!r}")

            elif op == "probe":
                # Semantic-preservation probe: record answerability before a
                # lifecycle op, to be compared after it.
                block = recall(step["query"])
                expected = step.get("expect", [])
                ok = all(e.lower() in block.lower() for e in expected) if expected else bool(block)
                result.probes_before += 1
                if ok:
                    result.probes_after += 1
                else:
                    # A probe that fails *before* the operation cannot be
                    # preserved by it; drop it from both sides.
                    result.probes_before -= 1
                    check(f"probe_before:{step['query'][:40]}", False,
                          f"block={block[:160]!r}")

            elif op == "probe_after":
                block = recall(step["query"])
                expected = step.get("expect", [])
                ok = all(e.lower() in block.lower() for e in expected) if expected else bool(block)
                result.probes_after += 1
                result.probes_before += 1
                check(f"probe_after:{step['query'][:40]}", ok, f"block={block[:160]!r}")

            elif op == "tier":
                with engine._session() as session:
                    beliefs = _find_beliefs(session, user_id, step["key"])
                    expected = step["expect"]
                    found = [b.memory_tier for b in beliefs]
                    check(f"tier:{step['key']}", expected in found,
                          f"expected={expected} found={found}")

            elif op == "digest":
                with engine._session() as session:
                    beliefs = _find_beliefs(session, user_id, step["key"])
                    digests = [_digest_for(session, b.id) for b in beliefs]
                    digests = [d for d in digests if d is not None]
                    if step.get("expect_exists", True):
                        check(f"digest_exists:{step['key']}", bool(digests),
                              f"beliefs={len(beliefs)}")
                        if digests and "support_count" in step:
                            check(
                                f"digest_support_count:{step['key']}",
                                digests[0].support_count == step["support_count"],
                                f"expected={step['support_count']} got={digests[0].support_count}",
                            )
                    else:
                        check(f"digest_absent:{step['key']}", not digests,
                              f"digests={len(digests)}")

            elif op == "correct":
                with engine._session() as session:
                    beliefs = _find_beliefs(session, user_id, step["key"])
                    if not check(f"correct_target:{step['key']}", bool(beliefs)):
                        continue
                    engine.correct(
                        user_id=user_id, belief_id=beliefs[0].id,
                        new_claim_text=step["new_text"],
                        new_object=step.get("new_object"),
                        new_predicate=step.get("new_predicate"),
                    )

            elif op == "inject_conflict":
                # A cross-source disagreement: the repository's documented
                # path for "two sources assert different things" (an app
                # ingesting a second source, as opposed to the user
                # retracting themselves).  The extractor never produces
                # this — a user retraction is a self-correction by design —
                # so the bench constructs it the way a host application
                # would.
                from mirror_memory.core.repository import (
                    CONFLICT_SOURCE_CONFLICT,
                    record_claim,
                )

                with engine._session() as session:
                    beliefs = _find_beliefs(session, user_id, step["key"])
                    if not check(f"conflict_target:{step['key']}", bool(beliefs)):
                        continue
                    target = beliefs[0]
                    record_claim(
                        session, user_id,
                        dimension=target.dimension,
                        key=target.key,
                        claim_text=step["claim_text"],
                        predicate=target.predicate,
                        object=target.object,
                        relation="contradicts",
                        conflict_kind=CONFLICT_SOURCE_CONFLICT,
                        confidence=0.5,
                    )
                    session.commit()

            elif op == "forget":
                # "Forget what I said about X" covers every belief the
                # extractor filed under that topic — the same text can
                # land on more than one key across turns.
                with engine._session() as session:
                    beliefs = _find_beliefs(session, user_id, step["key"])
                    if not check(f"forget_target:{step['key']}", bool(beliefs)):
                        continue
                    ids = [b.id for b in beliefs]
                for belief_id in ids:
                    engine.forget_belief(user_id=user_id, belief_id=belief_id)

            elif op == "restore":
                with engine._session() as session:
                    beliefs = _find_beliefs(session, user_id, step["key"])
                    if not check(f"restore_target:{step['key']}", bool(beliefs)):
                        continue
                    ids = [b.id for b in beliefs]
                for belief_id in ids:
                    engine.restore_belief(user_id=user_id, belief_id=belief_id)

            elif op == "storage_mark":
                with engine._session() as session:
                    result.chars_before = _evidence_chars(session, user_id)

            elif op == "gone":
                with engine._session() as session:
                    beliefs = _find_beliefs(session, user_id, step["key"])
                    check(f"gone:{step['key']}", not beliefs,
                          f"remaining={[b.key for b in beliefs]}")

            elif op == "tombstone":
                from sqlalchemy import select

                from mirror_memory.core.models import DeletionTombstone

                with engine._session() as session:
                    tombstones = list(
                        session.scalars(
                            select(DeletionTombstone).where(
                                DeletionTombstone.user_id == user_id
                            )
                        )
                    )
                    check("tombstone_exists", bool(tombstones))
                    if tombstones:
                        latest = tombstones[-1]
                        check("tombstone_content_free",
                              step["key"].lower() not in (latest.scope_hash or "").lower()
                              and step["key"].lower() not in (latest.counts_json or "").lower(),
                              f"scope_hash={latest.scope_hash}")

            elif op == "storage_check":
                with engine._session() as session:
                    result.chars_after = _evidence_chars(session, user_id)
                    if step.get("expect_reduction"):
                        check(
                            "storage_reduced",
                            result.chars_after < result.chars_before,
                            f"before={result.chars_before} after={result.chars_after}",
                        )
                    # The keep policy's ceiling for this fold: with N total
                    # support links and K kept representatives, no
                    # implementation can reclaim more than 1 - K/N.
                    beliefs = _find_beliefs(session, user_id, step.get("key", ""))
                    for belief in beliefs:
                        digest = _digest_for(session, belief.id)
                        if digest is None:
                            continue
                        kept = len(safe_json_list(digest.representative_ids))
                        total = digest.support_count
                        if total > kept > 0:
                            result.theoretical_reduction = round(1.0 - kept / total, 4)
                            break

            elif op == "stale_worker":
                # A worker computed against the pre-deletion revision; the
                # Publisher must refuse it after the belief is gone.
                from mirror_memory.core.proposal import (
                    TRANSITION_SUPPORT,
                    StateTransitionProposal,
                )
                from mirror_memory.core.publisher import Publisher
                from mirror_memory.core.repository import get_state_revision

                with engine._session() as session:
                    beliefs = _find_beliefs(session, user_id, step["key"])
                    if not check(f"stale_target:{step['key']}", bool(beliefs)):
                        continue
                    revision = get_state_revision(session, user_id)
                    proposal = StateTransitionProposal(
                        transition=TRANSITION_SUPPORT,
                        user_id=user_id,
                        target_belief_id=beliefs[0].id,
                        expected_revision=revision,
                        payload={"claim_text": "stale worker write"},
                    )
                    engine.forget_belief(user_id=user_id, belief_id=beliefs[0].id)
                    session.commit()
                with engine._session() as session:
                    decision = Publisher(session).publish(proposal)
                    check("stale_worker_refused", not decision.committed,
                          f"reason={decision.reason}")

            else:
                check(f"unknown_op:{op}", False, "unrecognised step operation")

        # A case passes when every check passed.
        result.passed = all(ok for _name, ok, _detail in result.checks) and bool(result.checks)
        if not result.checks:
            result.error = "no assertions ran"
    except Exception as exc:  # a crashed case is a failed case, with the reason
        result.error = f"{type(exc).__name__}: {exc}"
        result.passed = False
    return result


def _client_label(llm_client: Any | None) -> tuple[str, str]:
    """(model, base_url) for the report header; tolerates duck-typed clients."""
    if llm_client is None:
        return "", ""
    model = getattr(llm_client, "model", None) or getattr(llm_client, "_model", "") or ""
    base_url = getattr(llm_client, "base_url", None) or getattr(llm_client, "_base_url", "") or ""
    return str(model), str(base_url)


def run_metabolism_bench(
    config: Any,
    *,
    llm_client: Any | None = None,
    categories: tuple[str, ...] | None = None,
) -> BenchReport:
    """Run the whole MetabolismBench suite and return the report."""
    cases = load_metabolism_cases(categories=categories)
    model, base_url = _client_label(llm_client)
    report = BenchReport(model=model, base_url=base_url, degraded=llm_client is None)
    for case in cases:
        report.results.append(run_case(case, config, llm_client=llm_client))
    return report


# -- Gate ----------------------------------------------------------------------


@dataclass(frozen=True)
class MetabolismGate:
    """Release thresholds for the lifecycle layer."""

    # Every category that protects a user signal must be perfect: a
    # protection failure is a bug, not a tuning problem.
    protection_categories: tuple[str, ...] = (
        "correction_protection",
        "conflict_preservation",
        "deletion",
        "stale_worker",
    )
    min_pass_rate: float = 0.95
    min_semantic_preservation: float = 0.98
    # Efficiency against the keep policy's ceiling, not a flat byte ratio:
    # with 13 links and a 6-row sample the best possible reduction is
    # ~0.54, so a flat 0.80 would be unfalsifiable at the threshold.
    min_compaction_efficiency: float = 0.90

    def evaluate(self, report: BenchReport) -> "GateVerdict":
        failures: list[str] = []
        by_category = report.by_category()

        for category in self.protection_categories:
            bucket = by_category.get(category)
            if bucket is None:
                continue
            if bucket["pass_rate"] < 1.0:
                failures.append(
                    f"{category} pass rate {bucket['pass_rate']:.2f} < 1.00 "
                    f"(protection categories must be perfect)"
                )

        if report.overall_pass_rate < self.min_pass_rate:
            failures.append(
                f"overall pass rate {report.overall_pass_rate:.2f} < {self.min_pass_rate:.2f}"
            )
        semantic = report.semantic_preservation
        if semantic is not None and semantic < self.min_semantic_preservation:
            failures.append(
                f"semantic preservation {semantic:.3f} < {self.min_semantic_preservation:.2f}"
            )
        efficiency = report.compaction_efficiency
        if efficiency is not None and efficiency < self.min_compaction_efficiency:
            failures.append(
                f"compaction efficiency {efficiency:.3f} < {self.min_compaction_efficiency:.2f}"
            )
        return GateVerdict(passed=not failures, failures=failures, report=report)


@dataclass
class GateVerdict:
    passed: bool
    failures: list[str]
    report: BenchReport

    @property
    def summary(self) -> str:
        if self.passed:
            return (
                f"metabolism gate PASSED ({self.report.overall_pass_rate:.0%} cases, "
                f"semantic preservation {self.report.semantic_preservation}, "
                f"compaction efficiency {self.report.compaction_efficiency})"
            )
        return "metabolism gate FAILED: " + "; ".join(self.failures)


def check_metabolism_gate(report: BenchReport) -> GateVerdict:
    return MetabolismGate().evaluate(report)


# -- Rendering -----------------------------------------------------------------


def render_report(report: BenchReport) -> str:
    """Human-readable report, matching the StateBench report style."""
    lines: list[str] = []
    lines.append("MetabolismBench v1 — memory lifecycle under simulated time")
    lines.append(f"  extractor: {report.model or 'none (K1-only degraded)'}")
    if report.base_url:
        lines.append(f"  base_url:  {report.base_url}")
    lines.append("")

    lines.append(f"{'category':<26}{'cases':>6}{'pass':>7}{'semantic':>10}{'storage':>10}{'effic.':>8}")
    lines.append("-" * 67)
    for category, bucket in sorted(report.by_category().items()):
        semantic = bucket["semantic_preservation"]
        storage = bucket["storage_reduction"]
        efficiency = bucket["compaction_efficiency"]
        lines.append(
            f"{category:<26}{bucket['cases']:>6}{bucket['pass_rate']:>7.2f}"
            f"{(f'{semantic:.3f}' if semantic is not None else '-'):>10}"
            f"{(f'{storage:.3f}' if storage is not None else '-'):>10}"
            f"{(f'{efficiency:.3f}' if efficiency is not None else '-'):>8}"
        )
    lines.append("-" * 67)
    lines.append(
        f"{'OVERALL':<26}{len(report.results):>6}{report.overall_pass_rate:>7.2f}"
        f"{(f'{report.semantic_preservation:.3f}' if report.semantic_preservation is not None else '-'):>10}"
        f"{(f'{report.storage_reduction:.3f}' if report.storage_reduction is not None else '-'):>10}"
        f"{(f'{report.compaction_efficiency:.3f}' if report.compaction_efficiency is not None else '-'):>8}"
    )

    latency = report.recall_latency_ms
    if latency:
        lines.append("")
        lines.append(
            f"recall latency: p50={latency['p50']}ms p95={latency['p95']}ms "
            f"max={latency['max']}ms (n={latency['n']})"
        )

    verdict = check_metabolism_gate(report)
    lines.append("")
    lines.append(verdict.summary)

    failures = [r for r in report.results if not r.passed]
    if failures:
        lines.append("")
        lines.append("failures:")
        for result in failures:
            lines.append(f"  {result.case_id} [{result.category}]")
            if result.error:
                lines.append(f"    error: {result.error}")
            for name, _ok, detail in result.checks:
                if not _ok:
                    lines.append(f"    - {name}: {detail}")
    return "\n".join(lines)
