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
from typing import Any, Mapping

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


def _field(report: Any, name: str, default: Any = None) -> Any:
    return report.get(name, default) if isinstance(report, Mapping) else getattr(report, name, default)


def _track(report: Any, name: str) -> Mapping[str, Any] | None:
    value = (_field(report, "tracks") or {}).get(name)
    if isinstance(value, Mapping):
        return value
    if value is None:
        return None
    return {"score": value}


def _v11_report(report: Any) -> bool:
    return isinstance((_field(report, "tracks") or {}).get("state"), Mapping)


def _score(stats: Mapping[str, Any] | None) -> float | None:
    value = stats.get("score") if stats is not None else None
    return float(value) if value is not None else None


def _unevaluated(name: str, threshold: float, *, at_least: bool = True) -> dict:
    return {
        "check": name,
        "value": None,
        "threshold": threshold,
        "comparison": ">=" if at_least else "<=",
        "passed": False,
        "evaluated": False,
    }


def check_release_gate(
    report: Any,
    *,
    manifest_path: str | Path | None = None,
    production_report: Any | None = None,
    forced: bool = True,
) -> GateResult:
    """Check a StateBench report against the release thresholds.

    Parameters
    ----------
    report:
        A StateBench v1.1 report, JSON payload, or legacy v1 report.
    manifest_path:
        When given, the report must have a manifest next to it.
    production_report:
        The matching production run for a forced v1.1 report.  Both runs
        must use the same cases and extractor model.
    forced:
        Require a forced v1.1 run and its production comparison.  Legacy v1
        reports retain their original standalone gate behavior.
    """
    checks: list[dict] = []
    failures: list[str] = []

    v11 = _v11_report(report)
    state = _track(report, "state") if v11 else {
        "score": _field(report, "score"),
        "by_category": _field(report, "by_category") or {},
    }

    def require(name: str, value: float | None, threshold: float, label: str) -> None:
        if value is None:
            checks.append(_unevaluated(name, threshold))
            failures.append(f"{label} not evaluated")
            return
        checks.append(_check(name, value, threshold))
        if value < threshold:
            failures.append(f"{label} {value:.3f} < {threshold}")

    require("state_overall", _score(state), GATE_STATE_OVERALL, "state overall")
    for category, stats in (state.get("by_category") or {}).items():
        require(
            f"state_category[{category}]", _score(stats),
            GATE_STATE_PER_CATEGORY, f"category {category}",
        )

    for name, threshold in (("recall", GATE_RECALL), ("answer", GATE_ANSWER)):
        track = _track(report, name)
        if v11 or track is not None:
            require(f"{name}_overall", _score(track), threshold, name)

    if v11 and forced:
        from mirror_memory.bench.statebench_v1_1 import load_statebench_v1_1

        expected_cases = load_statebench_v1_1()
        expected_ids = [case.case_id for case in expected_cases]
        actual_ids = [
            item["case_id"] if isinstance(item, Mapping) else item.case_id
            for item in (_field(report, "cases") or [])
        ]
        coverage_ok = actual_ids == expected_ids
        checks.append({
            "check": "full_case_coverage",
            "value": len(actual_ids),
            "threshold": len(expected_ids),
            "comparison": "exact case IDs and order",
            "passed": coverage_ok,
        })
        if not coverage_ok:
            failures.append(
                f"full v1.1 case coverage required ({len(actual_ids)}/{len(expected_ids)})"
            )
        expected_categories = {case.category for case in expected_cases}
        actual_categories = set(state.get("by_category") or {})
        category_coverage_ok = actual_categories == expected_categories
        checks.append({
            "check": "full_category_coverage",
            "value": len(actual_categories),
            "threshold": len(expected_categories),
            "passed": category_coverage_ok,
        })
        if not category_coverage_ok:
            failures.append("all v1.1 state categories must be evaluated")
        for track_name in ("state", "recall", "answer"):
            track = _track(report, track_name)
            evaluated = track.get("total") if track is not None else None
            track_coverage_ok = evaluated == len(expected_ids)
            checks.append({
                "check": f"{track_name}_case_coverage",
                "value": evaluated,
                "threshold": len(expected_ids),
                "passed": track_coverage_ok,
            })
            if not track_coverage_ok:
                failures.append(f"{track_name} evaluated {evaluated or 0}/{len(expected_ids)} cases")
        mode = _field(report, "extraction_mode")
        mode_ok = mode == "forced"
        checks.append({"check": "forced_mode", "value": mode, "passed": mode_ok})
        if not mode_ok:
            failures.append(f"forced run required (got {mode or 'unknown'})")
        # Extraction health: a forced run with failed, empty, unparsed or
        # schema-rejected K2 calls is measuring the fallback, not the model.
        # The runner refuses a fully-failed run; the gate refuses any run
        # whose extraction was not clean, because a partially degraded run
        # still reports model-labelled numbers.
        health = _field(report, "extraction_health") or {}
        attempts = health.get("k2_attempts")
        expected_turns = health.get("dataset_turns")
        turns_ok = attempts is not None and attempts == expected_turns
        checks.append({
            "check": "k2_attempted_every_turn",
            "value": attempts,
            "threshold": expected_turns,
            "comparison": "==",
            "passed": turns_ok,
        })
        if not turns_ok:
            failures.append(
                f"K2 attempted {attempts if attempts is not None else 'unknown'} "
                f"of {expected_turns if expected_turns is not None else 'unknown'} turns"
            )
        for key, label in (
            ("k2_failures", "failed K2 calls"),
            ("k2_empty_responses", "empty K2 responses"),
            ("k2_parse_failures", "unparsed K2 responses"),
            ("k2_schema_rejections", "schema-rejected K2 responses"),
        ):
            count = health.get(key) or 0
            clean = count == 0
            checks.append({
                "check": key,
                "value": count,
                "threshold": 0,
                "comparison": "==",
                "passed": clean,
            })
            if not clean:
                failures.append(f"{count} {label}")
        if production_report is None:
            checks.append(_unevaluated("production_gap", GATE_PRODUCTION_GAP, at_least=False))
            failures.append("production gap not evaluated")
        else:
            try:
                gap = compare_forced_production(report, production_report)
            except ValueError as exc:
                failures.append(f"production comparison invalid: {exc}")
                checks.append(_unevaluated("production_gap", GATE_PRODUCTION_GAP, at_least=False))
            else:
                checks.append(_check(
                    "production_gap", gap["gap"], GATE_PRODUCTION_GAP, at_least=False,
                ))
                if not gap["gate"]:
                    failures.append(
                        f"production gap {gap['gap']:.3f} > {GATE_PRODUCTION_GAP}"
                    )

    if v11 and manifest_path is None:
        checks.append(_unevaluated("manifest_present", 1.0))
        failures.append("manifest not evaluated")
    elif manifest_path is not None:
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
    forced_v11 = _v11_report(forced)
    production_v11 = _v11_report(production)
    if forced_v11 != production_v11:
        raise ValueError("report versions differ")
    if forced_v11:
        if _field(forced, "extraction_mode") != "forced":
            raise ValueError("first report must use forced extraction")
        if _field(production, "extraction_mode") != "production":
            raise ValueError("second report must use production extraction")
        if _field(forced, "dataset_sha256") != _field(production, "dataset_sha256"):
            raise ValueError("dataset hashes differ")
        forced_ids = [item.case_id if not isinstance(item, Mapping) else item["case_id"]
                      for item in _field(forced, "cases")]
        production_ids = [item.case_id if not isinstance(item, Mapping) else item["case_id"]
                          for item in _field(production, "cases")]
        if forced_ids != production_ids:
            raise ValueError("case sets or order differ")
        if _field(forced, "extractor_model") != _field(production, "extractor_model"):
            raise ValueError("extractor models differ")
        forced_state = _track(forced, "state")
        production_state = _track(production, "state")
        forced_score = _score(forced_state)
        production_score = _score(production_state)
        if forced_score is None or production_score is None:
            raise ValueError("state track was not evaluated")
        forced_cats = forced_state.get("by_category") or {}
        production_cats = production_state.get("by_category") or {}
    else:
        forced_score = float(_field(forced, "score", 0.0))
        production_score = float(_field(production, "score", 0.0))
        forced_cats = _field(forced, "by_category") or {}
        production_cats = _field(production, "by_category") or {}
    gap = forced_score - production_score

    per_category = {}
    for category in sorted(set(forced_cats) | set(production_cats)):
        f = _score(forced_cats.get(category))
        p = _score(production_cats.get(category))
        if f is None or p is None:
            raise ValueError(f"category {category} is missing from one report")
        per_category[category] = round(f - p, 4)

    return {
        "forced": round(forced_score, 4),
        "production": round(production_score, 4),
        "gap": round(gap, 4),
        "gate": gap <= GATE_PRODUCTION_GAP,
        "threshold": GATE_PRODUCTION_GAP,
        "per_category_gap": per_category,
    }
