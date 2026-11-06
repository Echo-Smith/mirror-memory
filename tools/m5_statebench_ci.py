"""CI adapter for the StateBench release gate (report-only).

GitHub Actions cannot let the release gate turn CI red (see the m5 plan and
docs/statebench.md).  This tool runs the StateBench v1.1 tracks, records the
provenance manifest, evaluates ``bench.gate.check_release_gate`` with four-state
per-track semantics, and writes distinct, non-overwriting artifacts:

* **raw per-case results** — the full report JSON (per-case track outcomes);
* **manifest** — commit, dataset SHA-256, config/prompt hashes, model;
* **gate verdict** — the pass/fail verdict + per-check + per-track states;
* **summary** — a Markdown block whose FIRST line is ``PASS: ...`` / ``FAIL: ...``.

Each artifact is written to its OWN path.  They never overwrite one another so
a later write cannot erase an earlier one.

Track states (four-state)
-------------------------

``pass`` / ``fail`` / ``not_run`` / ``error`` — evaluated per track and never
merged:

* ``not_run`` — a track whose prerequisite (e.g. ``MM_LLM_API_KEY``) is absent.
  Never a pass.
* ``error`` — a track whose execution raised.  Never a pass and never
  relabelled as ``not_run``; surfaced loudly in the summary.
* A completed track's result is never overwritten because a *different* track
  errored.

Forced vs production
--------------------

When the key is present this tool runs BOTH a forced and a production track
pair.  Forced requires an every-turn K2 attempt and is checked against the
full thresholds.  Production is reported as configured: its true coverage and
failures are counted and compared to forced via the gap check.  A production
throttle never lowers the forced thresholds, and a production ``not_run`` /
``error`` is visible but does not erase the forced verdict.

Dataset version
---------------

The report carries ``dataset_version`` (from the v1.1 dataset header, currently
``1.2.0`` for ``statebench_v1_1.json``).  Some legacy docs call the file
"v1.1"; the authoritative in-file version is emitted verbatim in the summary so
v1 (``statebench_v1.json``) and v1.1 results are never confused.

Usage::

    # deterministic track only; LLM tracks marked not_run when no key
    python tools/m5_statebench_ci.py --output artifacts/statebench-gate.json

    # allow LLM tracks when the secret is present
    MM_LLM_API_KEY=... python tools/m5_statebench_ci.py \
        --allow-llm --output artifacts/statebench-gate.json

``--allow-llm`` only *permits* the LLM tracks; they run only when
``MM_LLM_API_KEY`` is set.  This tool always runs the deterministic track.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

from mirror_memory.bench.gate import (
    TRACK_ERROR,
    TRACK_FAIL,
    TRACK_NOT_RUN,
    TRACK_PASS,
    GateResult,
    TrackVerdict,
    check_release_gate,
    evaluate_tracks,
)
from mirror_memory.bench.statebench_v1_1_runner import (
    StateBenchV11Report,
    run_statebench_v1_1,
    write_statebench_v1_1_report,
)

TRACK_STATE = "state"
TRACK_RECALL = "recall"
TRACK_ANSWER = "answer"
TRACK_FORCED = "forced"
TRACK_PRODUCTION = "production"
SCORED_TRACKS = (TRACK_STATE, TRACK_RECALL, TRACK_ANSWER)
# Every track that can appear in the verdict / summary, in display order.
ALL_TRACKS = (TRACK_STATE, TRACK_RECALL, TRACK_ANSWER, TRACK_FORCED, TRACK_PRODUCTION)

# Failure phase tags so a client-construction failure is never conflated with a
# track-run failure.
PHASE_CLIENT_INIT = "client_init"
PHASE_RUN = "run"


def _error_verdict(
    track: str, exc: BaseException, *, phase: str,
) -> TrackVerdict:
    """Build an ``error`` verdict that names the failure phase.

    ``phase='client_init'`` means the extractor/answerer client could not be
    constructed; ``phase='run'`` means the track started and then raised.  Both
    are ``error`` (never pass, never not_run), but the reason must tell them
    apart so an operator knows whether to fix credentials/config or the run.
    """
    detail = f"{type(exc).__name__}: {exc}"
    if phase == PHASE_CLIENT_INIT:
        reason = f"client construction/init failed: {detail}"
    else:
        reason = f"track run failed: {detail}"
    return TrackVerdict(track=track, state=TRACK_ERROR, reason=reason, error=detail)


def _not_run_track(reason: str = "prerequisite absent") -> dict[str, Any]:
    """A track that intentionally did not run -- never a pass."""
    return {"total": None, "passed": None, "score": None, "not_run": True, "reason": reason}


def _error_track(exc: BaseException) -> dict[str, Any]:
    """A track whose execution raised -- never a pass, never ``not_run``."""
    return {
        "total": None,
        "passed": None,
        "score": None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def build_gate_report(
    report: dict[str, Any],
    *,
    llm_enabled: bool,
    answer_not_run_reason: str = "prerequisite absent",
) -> dict[str, Any]:
    """Return a gate-facing report with explicit four-state track markers.

    A deterministic run always fills ``state`` and ``recall``.  ``answer``
    requires an LLM answerer; when absent it is replaced with a ``not_run``
    marker (``score`` is ``None``) so the gate treats it as unevaluated
    (fail-closed), not a pass.
    """
    gate_report = json.loads(json.dumps(report))  # deep copy
    tracks = gate_report.setdefault("tracks", {})
    if not llm_enabled:
        tracks[TRACK_ANSWER] = _not_run_track(answer_not_run_reason)
    return gate_report


def _track_status(track: dict[str, Any] | None) -> str:
    """Map a raw track payload to a display state string.

    ``error`` takes precedence over ``not_run`` so an execution failure is
    never displayed as a benign "did not run".
    """
    if track is None:
        return TRACK_NOT_RUN
    if track.get("error"):
        return TRACK_ERROR
    if track.get("not_run"):
        return TRACK_NOT_RUN
    score = track.get("score")
    if score is None:
        return TRACK_NOT_RUN
    return "evaluated"


def _verdict_for_checks(result: GateResult, name: str) -> TrackVerdict | None:
    return result.tracks.get(name)


def _render_checks(checks: list[dict[str, Any]]) -> list[str]:
    lines = ["", "| check | value | threshold | pass |", "|---|---|---|---|"]
    for c in checks:
        value = c.get("value")
        if value is None:
            value_s = "n/a" if c.get("evaluated") is False else "not_run"
        else:
            value_s = str(value)
        threshold = c.get("threshold", "")
        ok = c.get("passed")
        pass_s = "pass" if ok is True else ("not_run" if c.get("evaluated") is False else "FAIL")
        lines.append(f"| {c.get('check')} | {value_s} | {threshold} | {pass_s} |")
    return lines


def render_summary(
    *,
    passed: bool,
    failures: list[str],
    checks: list[dict[str, Any]],
    tracks: dict[str, Any],
    extraction_mode: str,
    dataset_sha256: str,
    dataset_version: str,
    llm_enabled: bool,
    report_path: str,
    manifest_path: str,
    verdict_path: str,
    verdict_tracks: dict[str, TrackVerdict] | None = None,
) -> str:
    """Markdown summary.  First line is ``PASS: ...`` or ``FAIL: ...``."""
    verdict_tracks = verdict_tracks or {}
    errors = [name for name, v in verdict_tracks.items() if v.state == TRACK_ERROR]

    if passed:
        head = "PASS: all release gates cleared"
    else:
        # First line must carry the failing items; surface errors first because
        # they are infrastructure failures, not mere threshold misses.
        detail = "; ".join(failures) if failures else "gate did not pass"
        if errors:
            detail = f"track error(s) [{', '.join(errors)}]; " + detail
        head = f"FAIL: {detail}"

    lines = [head, ""]
    lines.append(f"- dataset version: `{dataset_version}`")
    lines.append(f"- extraction mode: `{extraction_mode}`")
    lines.append(f"- LLM tracks enabled: `{llm_enabled}`")
    lines.append(f"- dataset sha256: `{dataset_sha256}`")
    lines.append(f"- raw report: `{report_path}`")
    lines.append(f"- manifest: `{manifest_path}`")
    lines.append(f"- gate verdict: `{verdict_path}`")
    lines.append("")
    lines.append("### Tracks (four-state)")
    lines.append("")
    lines.append("| track | score | state |")
    lines.append("|---|---|---|")
    # Render every track that has a verdict or a raw payload, so a `not_run`
    # forced AND production (no key) are BOTH visible -- neither stands in for
    # the other.
    rendered: list[str] = []
    for name in ALL_TRACKS:
        if name in verdict_tracks:
            rendered.append(name)
        elif name in tracks:
            rendered.append(name)
    for name in rendered:
        verdict = verdict_tracks.get(name)
        if verdict is not None:
            state_label = verdict.state
            score_label = "n/a" if verdict.score is None else f"{verdict.score:.3f}"
        else:
            status = _track_status(tracks.get(name))
            state_label = status if status in (TRACK_NOT_RUN, TRACK_ERROR) else "evaluated"
            score_label = status
        lines.append(f"| {name} | {score_label} | {state_label} |")
    lines.extend(_render_checks(checks))
    lines.append("")
    if errors:
        lines.append(
            "> **Infrastructure error**: the track(s) above raised during "
            "execution. This is NOT a `not_run` and NOT a pass; investigate "
            "before reading any score."
        )
        lines.append("")
    lines.append(
        "> A `not_run` track is never a pass; an `error` track is an "
        "infrastructure failure, not a `not_run` and not a pass. The workflow "
        "is report-only; a FAIL gate here does not turn CI red."
    )
    return "\n".join(lines)


def _paths(output: Path) -> dict[str, Path]:
    """Distinct, non-overwriting artifact paths derived from one output stem.

    Forced and production reports are saved SEPARATELY, each with its own
    manifest, plus the gate verdict and the summary.  No path overlaps another,
    so a failed track can never erase a completed one's files.
    """
    stem = output.stem
    parent = output.parent
    return {
        "report": output,
        "manifest": output.with_name(f"{stem}.manifest.json"),
        "forced_report": parent / f"{stem}.forced.json",
        "forced_manifest": parent / f"{stem}.forced.manifest.json",
        "production_report": parent / f"{stem}.production.json",
        "production_manifest": parent / f"{stem}.production.manifest.json",
        "verdict": output.with_name(f"{stem}.verdict.json"),
        "summary": output.with_name(f"{stem}.summary.md"),
    }


def _run_track(
    name: str,
    runner: Callable[[], StateBenchV11Report],
) -> tuple[StateBenchV11Report | None, dict[str, Any] | None, TrackVerdict | None]:
    """Run one track in isolation.  Never raises: errors become ``error``.

    Returns ``(report, raw_track_payload, verdict)``.  A successful run yields
    the report; a raised exception yields an ``error`` payload and NO report,
    and never touches any other track.
    """
    try:
        report = runner()
    except Exception as exc:  # noqa: BLE001 - isolate per-track failures
        payload = _error_track(exc)
        return None, payload, TrackVerdict(
            track=name, state=TRACK_ERROR, reason=f"{type(exc).__name__}: {exc}",
            error=f"{type(exc).__name__}: {exc}",
        )
    return report, None, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="StateBench release-gate CI adapter")
    parser.add_argument("--output", required=True, help="Path for the raw per-case report JSON")
    parser.add_argument(
        "--allow-llm",
        action="store_true",
        help="Permit LLM tracks when MM_LLM_API_KEY is set (they still need the key)",
    )
    parser.add_argument(
        "--mode",
        choices=("deterministic",),
        default="deterministic",
        help="Deterministic track is always run by this adapter",
    )
    parser.add_argument("--config-path", default="config/")
    parser.add_argument("--database-url", default="sqlite://")
    parser.add_argument("--models", default=os.getenv("MM_LLM_MODEL", "deepseek-chat"))
    parser.add_argument("--summary-out", default="", help="Override the Markdown summary path")
    parser.add_argument("--verdict-out", default="", help="Override the gate-verdict JSON path")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit to the first N cases (subset, for fast tests; full dataset when omitted)",
    )
    args = parser.parse_args(argv)

    output = Path(args.output)
    paths = _paths(output)
    if args.summary_out:
        paths["summary"] = Path(args.summary_out)
    if args.verdict_out:
        paths["verdict"] = Path(args.verdict_out)

    api_key = os.getenv("MM_LLM_API_KEY")
    llm_enabled = bool(args.allow_llm and api_key)

    # --- deterministic track (always) --------------------------------------
    base_report = run_statebench_v1_1(
        config_path=args.config_path,
        database_url=args.database_url,
        extraction_mode="deterministic",
        llm_client=None,
        answerer=None,
        answerer_model="",
        max_cases=args.max_cases,
    )
    report = base_report
    report_dict = base_report.to_dict()

    # --- LLM tracks (isolated; degrade to not_run/error, never raise) ------
    # Keep the report OBJECTS (not just dicts) so each can be written to its
    # own path with its own manifest.
    forced_report_obj: StateBenchV11Report | None = None
    production_report_obj: StateBenchV11Report | None = None
    production_report: dict[str, Any] | None = None
    answer_not_run_reason = "prerequisite absent (MM_LLM_API_KEY absent)"
    # Four-state verdicts for the LLM-owned tracks, set explicitly so a missing
    # key yields TWO separate `not_run` tracks (forced AND production), and a
    # client-construction failure is distinct from a track-run failure.
    forced_verdict: TrackVerdict | None = None
    production_verdict: TrackVerdict | None = None
    answer_verdict: TrackVerdict | None = None
    init_phase = PHASE_RUN  # default; set to client_init if construction fails

    if not llm_enabled:
        # A. No key: forced AND production are EACH explicitly not_run.  They
        # are separate tracks -- one must never stand in for the other.
        no_llm = "not_run: MM_LLM_API_KEY absent (LLM tracks disabled)"
        forced_verdict = TrackVerdict(track=TRACK_FORCED, state=TRACK_NOT_RUN, reason=no_llm)
        production_verdict = TrackVerdict(track=TRACK_PRODUCTION, state=TRACK_NOT_RUN, reason=no_llm)
        answer_verdict = TrackVerdict(track=TRACK_ANSWER, state=TRACK_NOT_RUN, reason=no_llm)
        answer_not_run_reason = no_llm
    else:
        try:
            from mirror_memory.bench.run import _build_answerer, _build_extractor

            extra_body = {"thinking": {"type": "disabled"}}
            base_url = os.getenv("MM_LLM_BASE_URL")
            extractor = _build_extractor(args.models, api_key, base_url, extra_body)
            answerer = _build_answerer(args.models, api_key, base_url, extra_body)
        except Exception as exc:  # noqa: BLE001 - client construction failure
            # B. Construction failure: the client itself could not be built, so
            # every LLM-owned track that needs it is `error` (client-init), not
            # not_run and not pass.
            init_phase = PHASE_CLIENT_INIT
            forced_verdict = _error_verdict(TRACK_FORCED, exc, phase=PHASE_CLIENT_INIT)
            production_verdict = _error_verdict(TRACK_PRODUCTION, exc, phase=PHASE_CLIENT_INIT)
            answer_verdict = _error_verdict(TRACK_ANSWER, exc, phase=PHASE_CLIENT_INIT)
            answer_not_run_reason = forced_verdict.error
        else:
            # Run forced then production.  Each is isolated: a run failure in
            # one never discards the other's completed result.
            forced_report_obj, forced_payload, _ = _run_track(
                "forced",
                lambda: run_statebench_v1_1(
                    config_path=args.config_path,
                    database_url=args.database_url,
                    extraction_mode="forced",
                    llm_client=extractor,
                    answerer=answerer,
                    answerer_model=args.models,
                    max_cases=args.max_cases,
                ),
            )
            production_report_obj, production_payload, _ = _run_track(
                "production",
                lambda: run_statebench_v1_1(
                    config_path=args.config_path,
                    database_url=args.database_url,
                    extraction_mode="production",
                    llm_client=extractor,
                    answerer=answerer,
                    answerer_model=args.models,
                    max_cases=args.max_cases,
                ),
            )
            # B. Track-run failures carry a RUN-phase reason.
            if forced_payload is not None:
                forced_verdict = TrackVerdict(
                    track=TRACK_FORCED, state=TRACK_ERROR,
                    reason=f"track run failed: {forced_payload.get('error', '')}",
                    error=forced_payload.get("error", ""),
                )
            if production_payload is not None:
                production_verdict = TrackVerdict(
                    track=TRACK_PRODUCTION, state=TRACK_ERROR,
                    reason=f"track run failed: {production_payload.get('error', '')}",
                    error=production_payload.get("error", ""),
                )
            if forced_report_obj is not None:
                report = forced_report_obj
                report_dict = forced_report_obj.to_dict()
            if production_report_obj is not None:
                production_report = production_report_obj.to_dict()

    # --- write raw report + manifest (deterministic / gate subject) --------
    out, manifest = write_statebench_v1_1_report(
        report,
        paths["report"],
        config_dir=args.config_path,
        base_url=os.getenv("MM_LLM_BASE_URL", "") if llm_enabled else "",
        thinking="default",
        generation_params={"temperature": 0.0, "max_tokens": 800},
    )
    manifest = paths["manifest"] if Path(manifest) == paths["manifest"] else Path(manifest)

    # --- write forced and production SEPARATELY (own report + own manifest) -
    # C. A successful track's per-case report + its own manifest are written
    # even when the OTHER track errored / not_run.  Forced and production each
    # have their own file pair; neither overwrites the other.
    forced_report_path: str = ""
    forced_manifest_path: str = ""
    production_report_path: str = ""
    production_manifest_path: str = ""
    if forced_report_obj is not None:
        fout, fman = write_statebench_v1_1_report(
            forced_report_obj,
            paths["forced_report"],
            config_dir=args.config_path,
            base_url=os.getenv("MM_LLM_BASE_URL", "") if llm_enabled else "",
            thinking="default",
            generation_params={"temperature": 0.0, "max_tokens": 800},
        )
        forced_report_path, forced_manifest_path = str(fout), str(fman)
    if production_report_obj is not None:
        pout, pman = write_statebench_v1_1_report(
            production_report_obj,
            paths["production_report"],
            config_dir=args.config_path,
            base_url=os.getenv("MM_LLM_BASE_URL", "") if llm_enabled else "",
            thinking="default",
            generation_params={"temperature": 0.0, "max_tokens": 800},
        )
        production_report_path, production_manifest_path = str(pout), str(pman)

    # --- build gate-facing report and four-state verdicts -------------------
    gate_report = build_gate_report(
        report_dict,
        llm_enabled=llm_enabled and init_phase != PHASE_CLIENT_INIT and answer_verdict is None,
        answer_not_run_reason=answer_not_run_reason,
    )
    derived = evaluate_tracks(gate_report)
    # Explicit forced / production verdicts override the derived ones.  When a
    # track *ran* (even throttled), its real verdict is what the gate computes
    # from the report; explicit verdicts are only for not_run / error cases.
    if forced_verdict is not None:
        derived[TRACK_FORCED] = forced_verdict
    if production_verdict is not None:
        derived[TRACK_PRODUCTION] = production_verdict
    if answer_verdict is not None:
        derived[TRACK_ANSWER] = answer_verdict
    # When forced actually RAN (no explicit error/not_run verdict) surface its
    # own four-state so the forced track is visible in the verdict and summary,
    # not just implicit in the gate subject.  Its state score is measured against
    # the fixed overall threshold (forced is the threshold-bearing track).
    if forced_verdict is None and forced_report_obj is not None:
        from mirror_memory.bench.gate import GATE_STATE_OVERALL

        f_state_track = (forced_report_obj.to_dict().get("tracks") or {}).get("state") or {}
        f_score = f_state_track.get("score")
        f_state = (
            TRACK_PASS if (f_score is not None and f_score >= GATE_STATE_OVERALL)
            else TRACK_FAIL
        )
        forced_verdict = TrackVerdict(
            track=TRACK_FORCED, state=f_state, score=f_score,
            threshold=GATE_STATE_OVERALL,
            reason="" if f_state == TRACK_PASS
            else f"forced state {f_score:.3f} < {GATE_STATE_OVERALL}"
            if f_score is not None else "forced state not evaluated",
        )
        derived[TRACK_FORCED] = forced_verdict

    result = check_release_gate(
        gate_report,
        manifest_path=manifest,
        production_report=production_report,
        forced=True,
        track_verdicts=derived,
    )
    # Ensure forced / production always appear in the final verdict tracks, and
    # that an error / not_run on either is reflected in the gate failures so the
    # summary first line is honest.
    def _ensure_track(name: str, verdict: TrackVerdict | None) -> None:
        if verdict is None:
            return
        if name not in result.tracks:
            result.tracks[name] = verdict
        if verdict.state == TRACK_ERROR:
            msg = f"{name} track error: {verdict.reason}"
            if not any(msg in f for f in result.failures):
                result.failures.append(msg)
                result.passed = False
        elif verdict.state == TRACK_NOT_RUN:
            msg = f"{name} not run"
            if not any(msg in f for f in result.failures):
                result.failures.append(msg)
                result.passed = False

    _ensure_track(TRACK_FORCED, forced_verdict)
    _ensure_track(TRACK_PRODUCTION, production_verdict)
    _ensure_track(TRACK_ANSWER, answer_verdict)

    dataset_version = report_dict.get("dataset_version", "")

    # --- write gate verdict (its own path) ---------------------------------
    verdict_payload = {
        "passed": result.passed,
        "summary": result.summary,
        "failures": result.failures,
        "checks": result.checks,
        "tracks": {name: verdict.to_dict() for name, verdict in result.tracks.items()},
        "llm_tracks_enabled": llm_enabled,
        "dataset_version": dataset_version,
        "dataset_sha256": gate_report.get("dataset_sha256"),
        "extraction_mode": gate_report.get("extraction_mode"),
        "report": str(out),
        "manifest": str(manifest),
        "forced_report": forced_report_path,
        "forced_manifest": forced_manifest_path,
        "production_report": production_report_path,
        "production_manifest": production_manifest_path,
        "dataset_version_label": "statebench_v1_1 (in-file version "
        f"{dataset_version})",
    }
    paths["verdict"].parent.mkdir(parents=True, exist_ok=True)
    paths["verdict"].write_text(
        json.dumps(verdict_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- write summary (its own path) --------------------------------------
    # The summary renders its track table from result.tracks (which now always
    # includes forced / production as not_run or error where applicable), so a
    # no-key run shows BOTH forced and production as not_run.
    summary = render_summary(
        passed=result.passed,
        failures=result.failures,
        checks=result.checks,
        tracks=gate_report.get("tracks", {}),
        extraction_mode=gate_report.get("extraction_mode", "deterministic"),
        dataset_sha256=gate_report.get("dataset_sha256", ""),
        dataset_version=dataset_version,
        llm_enabled=llm_enabled,
        report_path=str(out),
        manifest_path=str(manifest),
        verdict_path=str(paths["verdict"]),
        verdict_tracks=result.tracks,
    )
    print(summary)
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(summary + "\n", encoding="utf-8")

    # Exit 0 always -- report-only.  Errors are in the artifacts, not the exit.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
