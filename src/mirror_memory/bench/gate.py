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

Track semantics (four-state)
----------------------------

Every track is evaluated independently and ends in exactly one of four
states:

``pass``
    The track ran and cleared its threshold.
``fail``
    The track ran and did not clear its threshold.
``not_run``
    The track was intentionally not attempted because a prerequisite was
    absent (e.g. no ``MM_LLM_API_KEY``).  ``not_run`` is **never** a pass.
``error``
    The track was attempted but execution raised.  ``error`` is **never** a
    pass and is **never** relabelled as ``not_run``; it stays visible as an
    infrastructure failure so it cannot hide behind a missing-key story.

Isolation rule: a completed track's result must not be overwritten because a
*different* track failed.  The per-track evaluation below never touches a
track it did not itself evaluate.

Forced vs production
--------------------

``forced`` — the threshold-bearing track — is the only track allowed to
demand an every-turn K2 attempt (full coverage + clean extraction health).
``production`` is the engine as configured: it reports its *true* coverage and
failures as-is.  A low production score is measured against the same fixed
thresholds (it never relaxes them) and against the forced/production gap.  A
production throttling does not lower the forced thresholds and a production
`not_run`/`error` does not silently pass, but it also does not overwrite the
forced result.
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

# The four track states.  A track is exactly one of these.
TRACK_PASS = "pass"
TRACK_FAIL = "fail"
TRACK_NOT_RUN = "not_run"
TRACK_ERROR = "error"
TRACK_STATES = frozenset({TRACK_PASS, TRACK_FAIL, TRACK_NOT_RUN, TRACK_ERROR})


@dataclass
class TrackVerdict:
    """Independent verdict for one track.

    ``state`` is one of the four ``TRACK_*`` values.  ``reason`` explains a
    non-pass; ``error`` keeps the raised exception payload so an execution
    failure is never mistaken for a clean ``not_run``.
    """

    track: str
    state: str
    score: float | None = None
    threshold: float | None = None
    reason: str = ""
    error: str = ""

    @property
    def passed(self) -> bool:
        return self.state == TRACK_PASS

    def to_dict(self) -> dict:
        return {
            "track": self.track,
            "state": self.state,
            "passed": self.passed,
            "score": self.score,
            "threshold": self.threshold,
            "reason": self.reason,
            "error": self.error,
        }


@dataclass
class GateResult:
    """Outcome of checking one report against the release thresholds."""

    passed: bool
    checks: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    tracks: dict[str, TrackVerdict] = field(default_factory=dict)

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
            "tracks": {name: verdict.to_dict() for name, verdict in self.tracks.items()},
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


def _track_state_of(stats: Mapping[str, Any] | None, threshold: float | None = None) -> str:
    """Classify a raw track payload into one four-state value.

    Precedence is important: ``error`` wins over ``not_run`` so an execution
    failure is never downgraded to a benign "did not run", and neither is ever
    ``pass``.

    When ``threshold`` is given and the track has a numeric score, the pass/fail
    decision is made against it (a below-threshold track is ``fail`` even
    though it ran).  Without a threshold, presence of a score is treated as a
    run and an explicitly ``passed: false`` payload is ``fail``.
    """
    if stats is None:
        return TRACK_NOT_RUN
    if stats.get("error"):
        return TRACK_ERROR
    if stats.get("not_run"):
        return TRACK_NOT_RUN
    score = _score(stats)
    if score is None:
        return TRACK_NOT_RUN
    if stats.get("passed") is False:
        return TRACK_FAIL
    if threshold is not None and score < threshold:
        return TRACK_FAIL
    return TRACK_PASS


def evaluate_tracks(report: Any, *, thresholds: Mapping[str, float] | None = None) -> dict[str, TrackVerdict]:
    """Evaluate each track independently into pass/fail/not_run/error.

    This is the track-isolation primitive: every track is decided from its own
    payload only.  A missing/empty/errored track never overwrites another.
    """
    thresholds = thresholds or {
        "state": GATE_STATE_OVERALL,
        "recall": GATE_RECALL,
        "answer": GATE_ANSWER,
    }
    v11 = _v11_report(report)
    verdicts: dict[str, TrackVerdict] = {}
    raw_tracks = _field(report, "tracks") or {}
    for name, threshold in thresholds.items():
        if not v11 and name == "state":
            stats = {"score": _field(report, "score")}
        else:
            # A legacy v1 report that simply does not declare a track (e.g. no
            # ``answer`` key at all) has no such track -- do not manufacture a
            # ``not_run`` verdict for a track the report never mentions.
            if not v11 and name not in raw_tracks:
                continue
            stats = _track(report, name)
        state = _track_state_of(stats, threshold)
        score = _score(stats) if stats is not None else None
        reason = ""
        if state == TRACK_ERROR:
            reason = str(stats.get("error") or "track execution failed")
        elif state == TRACK_NOT_RUN:
            reason = "track not run (prerequisite absent)"
        elif state == TRACK_FAIL:
            if score is not None and score < threshold:
                reason = f"{name} {score:.3f} < {threshold}"
            else:
                reason = f"{name} marked failed"
        verdicts[name] = TrackVerdict(
            track=name,
            state=state,
            score=score,
            threshold=threshold,
            reason=reason,
            error=str(stats.get("error") or "") if stats is not None else "",
        )
    return verdicts


def evaluate_production_track(production_report: Any) -> TrackVerdict:
    """Four-state verdict for a *production* run's execution integrity.

    Production is measured as configured, so a throttled run (some turns do not
    fire K2) is the EXPECTED truthful state and is not an integrity error.  But
    a real LLM call failure -- K2 failures, or zero calls when turns exist and a
    client was supplied -- means the measured numbers came from the K1 fallback,
    not the model.  That must be surfaced as ``error`` (never ``pass``, never a
    silent ``not_run``) so it cannot be read as a clean production measurement.

    A clean production run is ``pass`` (it ran truthfully).  Its actual scores
    are compared to forced through the gap check, not this verdict.
    """
    health = _field(production_report, "extraction_health") or {}
    attempts = health.get("k2_attempts")
    failures = health.get("k2_failures") or 0
    expected_turns = health.get("dataset_turns")
    if failures:
        return TrackVerdict(
            track="production", state=TRACK_ERROR,
            reason=f"{failures} production K2 call(s) failed -- measurement would be K1 fallback",
            error=f"{failures} failed K2 calls",
        )
    if attempts == 0 and (expected_turns or 0) > 0:
        return TrackVerdict(
            track="production", state=TRACK_ERROR,
            reason="production made zero K2 calls for a non-empty run -- measurement would be K1 fallback",
            error="zero K2 calls",
        )
    return TrackVerdict(
        track="production", state=TRACK_PASS,
        score=_score(_track(production_report, "state")),
        reason="",
    )


def check_release_gate(
    report: Any,
    *,
    manifest_path: str | Path | None = None,
    production_report: Any | None = None,
    forced: bool = True,
    track_verdicts: Mapping[str, TrackVerdict] | None = None,
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
    track_verdicts:
        Optional pre-computed four-state verdicts (from ``evaluate_tracks`` or
        the CI adapter).  When supplied, per-track pass/fail/not_run/error is
        taken from here — this is how a ``not_run`` or ``error`` track stays
        distinct from a measured ``fail``.  The legacy numerical checks below
        still run against the report payload so the thresholds are unchanged.
    """
    checks: list[dict] = []
    failures: list[str] = []
    tracks: dict[str, TrackVerdict] = dict(track_verdicts or {})

    v11 = _v11_report(report)
    state = _track(report, "state") if v11 else {
        "score": _field(report, "score"),
        "by_category": _field(report, "by_category") or {},
    }

    # Per-track four-state evaluation for the three scored tracks.  Dedicated
    # verdicts supplied by the caller win; otherwise derive them here.
    derived = evaluate_tracks(report)
    for name, verdict in derived.items():
        tracks.setdefault(name, verdict)

    # --- track state gating -------------------------------------------------
    # An `error` track is a hard failure (infrastructure), surfaced loudly.
    # A `not_run` track is a failure (fail-closed) but labelled as such.
    # `fail` is decided by the numerical thresholds below.
    for name in ("state", "recall", "answer"):
        verdict = tracks.get(name)
        if verdict is None:
            continue
        if verdict.state == TRACK_ERROR:
            failures.append(f"{name} track error: {verdict.error or verdict.reason}")
        elif verdict.state == TRACK_NOT_RUN and not (v11 is False and name == "answer"):
            # Legacy standalone v1 reports have no answer track at all; that is
            # not a track failure for the legacy path.
            failures.append(f"{name} not evaluated")

    def require(name: str, value: float | None, threshold: float, label: str) -> None:
        if value is None:
            checks.append(_unevaluated(name, threshold))
            # Only add the failure once (the state loop above may already have).
            if f"{label} not evaluated" not in failures:
                failures.append(f"{label} not evaluated")
            return
        checks.append(_check(name, value, threshold))
        if value < threshold:
            failures.append(f"{label} {value:.3f} < {threshold}")

    # Skip the numerical requirement for tracks explicitly marked not_run or
    # error so a missing key is reported as `not_run`, not as `0.000 < 0.90`.
    def track_blocked(name: str) -> bool:
        verdict = tracks.get(name)
        return verdict is not None and verdict.state in {TRACK_NOT_RUN, TRACK_ERROR}

    if not track_blocked("state"):
        require("state_overall", _score(state), GATE_STATE_OVERALL, "state overall")
        for category, stats in (state.get("by_category") or {}).items():
            require(
                f"state_category[{category}]", _score(stats),
                GATE_STATE_PER_CATEGORY, f"category {category}",
            )

    for name, threshold in (("recall", GATE_RECALL), ("answer", GATE_ANSWER)):
        track = _track(report, name)
        if track_blocked(name):
            continue
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
                verdict = tracks.get(track_name)
                # Coverage is only a failure if the track actually ran.  A
                # not_run / error track is already reported distinctly above;
                # do not double-report it as a coverage failure.
                if verdict is None or verdict.state not in {TRACK_NOT_RUN, TRACK_ERROR}:
                    failures.append(
                        f"{track_name} evaluated {evaluated or 0}/{len(expected_ids)} cases"
                    )
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
        #
        # These every-turn-K2 requirements apply to the FORCED track ONLY.
        # A production run is measured as configured and never has to prove an
        # every-turn attempt here.
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

        # Production comparison — evaluated independently of the forced track's
        # own pass/fail.  A production error/not_run must be visible but must
        # not erase the forced track's already-computed result.
        if production_report is None:
            checks.append(_unevaluated("production_gap", GATE_PRODUCTION_GAP, at_least=False))
            failures.append("production gap not evaluated")
        else:
            # Record the production track's own four-state verdict (execution
            # integrity).  A real call failure or a zero-call run is `error`,
            # never a pass; it is surfaced here and does NOT touch the forced
            # verdict that was computed above.
            prod_verdict = evaluate_production_track(production_report)
            tracks.setdefault("production", prod_verdict)
            if prod_verdict.state == TRACK_ERROR:
                failures.append(f"production track error: {prod_verdict.error or prod_verdict.reason}")
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

    return GateResult(passed=not failures, checks=checks, failures=failures, tracks=tracks)


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


__all__ = [
    "GATE_ANSWER",
    "GATE_PRODUCTION_GAP",
    "GATE_RECALL",
    "GATE_STATE_OVERALL",
    "GATE_STATE_PER_CATEGORY",
    "GateResult",
    "TRACK_ERROR",
    "TRACK_FAIL",
    "TRACK_NOT_RUN",
    "TRACK_PASS",
    "TRACK_STATES",
    "TrackVerdict",
    "check_release_gate",
    "compare_forced_production",
    "evaluate_production_track",
    "evaluate_tracks",
]
