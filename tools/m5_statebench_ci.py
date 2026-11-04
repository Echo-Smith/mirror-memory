"""CI adapter for the StateBench release gate (report-only).

GitHub Actions cannot let the release gate turn CI red (see the m5 plan and
docs/statebench.md).  This tool runs the deterministic StateBench track, records
the provenance manifest, and evaluates ``bench.gate.check_release_gate`` while
representing LLM-only tracks (extraction / answer) as ``not_run`` when the
API-key secret is absent.

Contract:

* A ``not_run`` track is **never** counted as a pass.  It is passed to the gate
  as an unevaluated track (``score`` is ``None``) so ``check_release_gate``
  emits ``evaluated: false`` / ``passed: false`` for it, and the overall
  verdict is therefore not a pass.
* The process exit code is always 0.  The verdict is emitted in the JSON
  payload (``passed`` / ``summary``) and in a Markdown summary block whose
  FIRST line is ``PASS: ...`` or ``FAIL: ...``.  The CI workflow consumes that
  first line for ``$GITHUB_STEP_SUMMARY`` and keeps the job non-required /
  ``continue-on-error`` so the workflow conclusion never turns CI red.

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
import sys
from pathlib import Path
from typing import Any

from mirror_memory.bench.gate import check_release_gate
from mirror_memory.bench.statebench_v1_1_runner import (
    run_statebench_v1_1,
    write_statebench_v1_1_report,
)


LLM_TRACKS = ("extraction", "answer")

# The state track is always evaluated (deterministic).  recall is evaluated by
# the deterministic runner without a model.  The answer track requires an LLM
# answerer, and the forced/production extraction tracks require an extractor
# client.  We surface all three tracks explicitly so the summary states, per
# track, whether it ran and whether its gate stands.
TRACK_STATE = "state"
TRACK_RECALL = "recall"
TRACK_ANSWER = "answer"


def _not_run_track() -> dict[str, Any]:
    """A track that intentionally did not run -- never a pass."""
    return {"total": None, "passed": None, "score": None, "not_run": True}


def build_gate_report(report: dict[str, Any], *, llm_enabled: bool) -> dict[str, Any]:
    """Return a gate-facing report where missing LLM tracks are not_run.

    The deterministic track always fills ``state`` and ``recall``.  ``answer``
    requires an LLM answerer; when LLM tracks are disabled we replace it with a
    ``not_run`` marker whose ``score`` is ``None`` so the gate treats it as
    unevaluated (fail-closed), not a pass.
    """
    gate_report = json.loads(json.dumps(report))  # deep copy
    tracks = gate_report.setdefault("tracks", {})
    if not llm_enabled:
        # Explicitly mark the LLM-owned answer track as not_run.  The gate
        # reads `score`; None -> unevaluated -> not a pass.
        tracks[TRACK_ANSWER] = _not_run_track()
    return gate_report


def _track_status(track: dict[str, Any] | None) -> str:
    if track is None:
        return "not_run"
    if track.get("not_run"):
        return "not_run"
    score = track.get("score")
    return "not_run" if score is None else f"{score:.3f}"


def _render_checks(checks: list[dict[str, Any]]) -> list[str]:
    lines = ["", "| check | value | threshold | pass |", "|---|---|---|---|"]
    for c in checks:
        value = c.get("value")
        if value is None:
            # A None value with a failed check means the track did not run.
            value_s = "n/a" if c.get("evaluated") is False else "not_run"
        else:
            value_s = str(value)
        threshold = c.get("threshold", "")
        ok = c.get("passed")
        # not_run (unevaluated) is never a pass
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
    llm_enabled: bool,
    report_path: str,
    manifest_path: str,
) -> str:
    """Markdown summary.  First line is ``PASS: ...`` or ``FAIL: ...``."""
    if passed:
        head = "PASS: all release gates cleared"
    else:
        # First line must carry the failing items.
        detail = "; ".join(failures) if failures else "gate did not pass"
        head = f"FAIL: {detail}"

    lines = [head, ""]
    lines.append(f"- extraction mode: `{extraction_mode}`")
    lines.append(f"- LLM tracks enabled: `{llm_enabled}`")
    lines.append(f"- dataset sha256: `{dataset_sha256}`")
    lines.append(f"- report: `{report_path}`")
    lines.append(f"- manifest: `{manifest_path}`")
    lines.append("")
    lines.append("### Tracks")
    lines.append("")
    lines.append("| track | score | status |")
    lines.append("|---|---|---|")
    for name in (TRACK_STATE, TRACK_RECALL, TRACK_ANSWER):
        status = _track_status(tracks.get(name))
        lines.append(f"| {name} | {status} | {'not_run' if status == 'not_run' else 'evaluated'} |")
    lines.extend(_render_checks(checks))
    lines.append("")
    lines.append(
        "> A `not_run` track is never a pass. The workflow is report-only; "
        "a FAIL gate here does not turn CI red."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="StateBench release-gate CI adapter")
    parser.add_argument("--output", required=True, help="Path for the gate report JSON")
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
    parser.add_argument("--summary-out", default="", help="Write the Markdown summary to a file")
    args = parser.parse_args(argv)

    api_key = os.getenv("MM_LLM_API_KEY")
    # LLM tracks only run when permitted AND the key is actually present.
    llm_enabled = bool(args.allow_llm and api_key)
    llm_status = "not_run"  # never a pass by default

    report = run_statebench_v1_1(
        config_path=args.config_path,
        database_url=args.database_url,
        extraction_mode="deterministic",
        llm_client=None,
        answerer=None,
        answerer_model="",
    )
    report_dict = report.to_dict()

    production_report: dict[str, Any] | None = None
    if llm_enabled:
        # The LLM tracks need a key.  We run the forced + production pair so the
        # gate can evaluate extraction and answer, and the production gap.  Any
        # failure to run degrades to not_run (report-only); it never raises.
        try:
            from mirror_memory.bench.run import _build_answerer, _build_extractor

            extra_body = {"thinking": {"type": "disabled"}}
            base_url = os.getenv("MM_LLM_BASE_URL")
            extractor = _build_extractor(args.models, api_key, base_url, extra_body)
            answerer = _build_answerer(args.models, api_key, base_url, extra_body)
            forced_report = run_statebench_v1_1(
                config_path=args.config_path,
                database_url=args.database_url,
                extraction_mode="forced",
                llm_client=extractor,
                answerer=answerer,
                answerer_model=args.models,
            )
            production_report_obj = run_statebench_v1_1(
                config_path=args.config_path,
                database_url=args.database_url,
                extraction_mode="production",
                llm_client=extractor,
                answerer=answerer,
                answerer_model=args.models,
            )
            report = forced_report
            report_dict = forced_report.to_dict()
            production_report = production_report_obj.to_dict()
            llm_status = "evaluated"
        except Exception as exc:  # noqa: BLE001 - report-only, degrade to not_run
            print(f"LLM tracks requested but did not run: {exc}", file=sys.stderr)
            llm_status = "not_run"

    out, manifest = write_statebench_v1_1_report(
        report,
        args.output,
        config_dir=args.config_path,
        base_url=os.getenv("MM_LLM_BASE_URL", "") if llm_enabled else "",
        thinking="default",
        generation_params={"temperature": 0.0, "max_tokens": 800},
    )

    gate_report = build_gate_report(report_dict, llm_enabled=(llm_status == "evaluated"))
    result = check_release_gate(
        gate_report,
        manifest_path=manifest,
        production_report=production_report,
        forced=True,
    )

    # Report-only: never signal failure to the shell.
    payload = {
        "passed": result.passed,
        "summary": result.summary,
        "failures": result.failures,
        "checks": result.checks,
        "llm_tracks_enabled": llm_enabled,
        "tracks": {
            name: _track_status(gate_report.get("tracks", {}).get(name))
            for name in (TRACK_STATE, TRACK_RECALL, TRACK_ANSWER)
        },
        "extraction_mode": gate_report.get("extraction_mode"),
        "dataset_sha256": gate_report.get("dataset_sha256"),
        "report": str(out),
        "manifest": str(manifest),
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = render_summary(
        passed=result.passed,
        failures=result.failures,
        checks=result.checks,
        tracks=gate_report.get("tracks", {}),
        extraction_mode=gate_report.get("extraction_mode", "deterministic"),
        dataset_sha256=gate_report.get("dataset_sha256", ""),
        llm_enabled=llm_enabled,
        report_path=str(out),
        manifest_path=str(manifest),
    )
    print(summary)
    if args.summary_out:
        Path(args.summary_out).write_text(summary + "\n", encoding="utf-8")

    # Exit 0 always -- report-only.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
