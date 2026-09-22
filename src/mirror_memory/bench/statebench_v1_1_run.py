"""Command line entry point for StateBench v1.1."""

from __future__ import annotations

import argparse
import os
import sys

from mirror_memory.bench.run import _build_answerer, _build_extractor
from mirror_memory.bench.statebench_v1_1_runner import (
    render_statebench_v1_1_report,
    run_statebench_v1_1,
    write_statebench_v1_1_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run StateBench v1.1")
    parser.add_argument(
        "--mode",
        choices=("deterministic", "production", "forced"),
        default="deterministic",
    )
    parser.add_argument("--splits", default="")
    parser.add_argument("--categories", default="")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--config-path", default="config/")
    parser.add_argument("--database-url", default="sqlite://")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.getenv("MM_LLM_MODEL", "deepseek-chat"))
    parser.add_argument("--api-key", default=os.getenv("MM_LLM_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("MM_LLM_BASE_URL"))
    parser.add_argument("--with-answerer", action="store_true")
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args(argv)

    if args.mode != "deterministic" and not args.api_key:
        print("production and forced modes require --api-key", file=sys.stderr)
        return 2
    if args.with_answerer and not args.api_key:
        print("--with-answerer requires --api-key", file=sys.stderr)
        return 2

    extra_body = {"thinking": {"type": "disabled"}} if args.no_thinking else None
    extractor = None
    if args.mode != "deterministic":
        extractor = _build_extractor(
            args.model, args.api_key, args.base_url, extra_body
        )
    answerer = None
    if args.with_answerer:
        answerer = _build_answerer(
            args.model, args.api_key, args.base_url, extra_body
        )

    categories = {
        value.strip() for value in args.categories.split(",") if value.strip()
    } or None
    splits = {
        value.strip() for value in args.splits.split(",") if value.strip()
    } or None
    report = run_statebench_v1_1(
        categories=categories,
        splits=splits,
        config_path=args.config_path,
        database_url=args.database_url,
        extraction_mode=args.mode,
        llm_client=extractor,
        answerer=answerer,
        answerer_model=args.model if answerer else "",
        max_cases=args.max_cases,
    )
    print(render_statebench_v1_1_report(report))
    out, manifest = write_statebench_v1_1_report(
        report,
        args.output,
        config_dir=args.config_path,
        base_url=args.base_url or "",
        thinking="disabled" if extra_body else "default",
        generation_params={"temperature": 0.0, "max_tokens": 800},
    )
    print(f"report: {out}")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
