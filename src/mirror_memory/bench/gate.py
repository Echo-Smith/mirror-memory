"""Release gate — the numbers a claim of architectural advantage must clear.

The plan's P2-10 fixes thresholds so "the architecture is proven" is a
checkable statement rather than a narrative.  Until these clear, the only
honest external claim is a reproducible measurement, not an advantage.

First-stage thresholds (from the plan):

| metric | gate |
|---|---|
| forced state overall | >= 0.90 |
| forced state per category | >= 0.85 |
| production vs forced gap | <= 0.05 |
| recall overall | >= 0.85 |
| answer overall (fixed answerer) | >= 0.85 |
| reports missing a manifest | 0 |

``forced`` means every turn got an extraction attempt (the throttle's
cold-start and novelty escapes both firing); ``production`` is the engine as
configured.  The gap between them is what throttling costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GATE_STATE_OVERALL = 0.90
GATE_STATE_PER_CATEGORY = 0.85
GATE_PRODUCTION_GAP = 0.05
GATE_RECALL = 0.85
GATE_ANSWER = 0.85


@dataclass
class GateResult:
    """Outcome of checking one report against the release thresholds."""

    passed: bool
    checks: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        if not self.failures:
            return f"{status}: all release gates cleared"
        return f"{status}: " + "; ".join(self.failures)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "failures": self.failures,
        }


def _check(name: str, value: float, threshold: float, *, at_least: bool = True) -> dict:
    ok = value >= threshold if at_least else value <= threshold
    return {
        "check": name,
        "value": round(value, 4),
        "threshold": threshold,
        "comparison": ">=" if at_least else "<=",
        "passed": ok,
    }


def check_release_gate(
    report: Any,
    *,
    manifest_path: str | Path | None = None,
    forced: bool = True,
) -> GateResult:
    """Check a StateBench report against the release thresholds.

    Parameters
    ----------
    report:
        A ``StateBenchReport`` (needs ``score``, ``by_category`` and
        optionally ``tracks``).
    manifest_path:
        When given, the report must have a manifest next to it.
    forced:
        Whether this run was a forced run (every turn attempted).  The
        production-gap check only applies to forced runs, since a production
        run has nothing to compare against.
    """
    checks: list[dict] = []
    failures: list[str] = []

    overall = float(getattr(report, "score", 0.0))
    checks.append(_check("state_overall", overall, GATE_STATE_OVERALL))
    if overall < GATE_STATE_OVERALL:
        failures.append(f"state overall {overall:.3f} < {GATE_STATE_OVERALL}")

    for category, stats in (getattr(report, "by_category", None) or {}).items():
        value = float(stats.get("score", 0.0))
        checks.append(_check(f"state_category[{category}]", value, GATE_STATE_PER_CATEGORY))
        if value < GATE_STATE_PER_CATEGORY:
            failures.append(
                f"category {category} {value:.3f} < {GATE_STATE_PER_CATEGORY}"
            )

    tracks = getattr(report, "tracks", None) or {}
    if tracks:
        recall = float(tracks.get("recall", 0.0))
        checks.append(_check("recall_overall", recall, GATE_RECALL))
        if recall < GATE_RECALL:
            failures.append(f"recall {recall:.3f} < {GATE_RECALL}")

        answer = float(tracks.get("answer", 0.0))
        checks.append(_check("answer_overall", answer, GATE_ANSWER))
        if answer < GATE_ANSWER:
            failures.append(f"answer {answer:.3f} < {GATE_ANSWER}")

    if manifest_path is not None:
        path = Path(manifest_path)
        present = path.exists()
        checks.append({
            "check": "manifest_present",
            "value": 1.0 if present else 0.0,
            "threshold": 1.0,
            "comparison": ">=",
            "passed": present,
        })
        if not present:
            failures.append(f"missing manifest: {path}")

    return GateResult(passed=not failures, checks=checks, failures=failures)


def compare_forced_production(forced: Any, production: Any) -> dict:
    """Gap between a forced run and a production run of the same cases.

    A large gap means throttling is costing real recall; a small one means
    the escapes are doing their job.  Returns the per-category and overall
    deltas plus the gate verdict on the gap itself.
    """
    forced_score = float(getattr(forced, "score", 0.0))
    production_score = float(getattr(production, "score", 0.0))
    gap = forced_score - production_score

    per_category = {}
    forced_cats = getattr(forced, "by_category", None) or {}
    production_cats = getattr(production, "by_category", None) or {}
    for category in sorted(set(forced_cats) | set(production_cats)):
        f = float((forced_cats.get(category) or {}).get("score", 0.0))
        p = float((production_cats.get(category) or {}).get("score", 0.0))
        per_category[category] = round(f - p, 4)

    return {
        "forced": round(forced_score, 4),
        "production": round(production_score, 4),
        "gap": round(gap, 4),
        "gate": gap <= GATE_PRODUCTION_GAP,
        "threshold": GATE_PRODUCTION_GAP,
        "per_category_gap": per_category,
    }
