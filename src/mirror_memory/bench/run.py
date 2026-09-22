"""Benchmark CLI.

Runs the funnel over a dataset and prints the stage metrics plus the failure
attribution, so a score always comes with an explanation.

    # Full LoCoMo run (needs an answering model for the answer stage)
    python -m mirror_memory.bench.run \\
        --dataset /path/to/locomo10.json --categories 4 --output out.json

    # Retrieval-side only, no LLM at all: extraction / identity / store /
    # retrieve / context are all measurable without an answering model
    python -m mirror_memory.bench.run --dataset /path/to/locomo10.json --no-answerer

The answer stage is skipped without ``--answerer-model``; every other metric
is still reported, which is the point -- the funnel localises the loss before
any answering model is involved.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from mirror_memory.bench.locomo import category_counts, load_locomo_cases
from mirror_memory.bench.manifest import (
    RunManifest,
    manifest_path_for,
    write_manifest,
)
from mirror_memory.bench.runner import render_report, run_cases


def _build_answerer(model: str, api_key: str, base_url: str | None, extra_body=None):
    from mirror_memory.llm import OpenAILLM

    llm = OpenAILLM(
        api_key=api_key, model=model, base_url=base_url, extra_body=extra_body,
    )

    def answerer(question: str, context: str) -> str:
        prompt = (
            "Answer the question using only the memory context below. "
            "Reply with the answer alone.\n\n"
            f"[Memory]\n{context or '(none)'}\n\n[Question]\n{question}\n\nANSWER:"
        )
        raw = llm.generate(system_prompt="", payload_text=prompt)
        return raw.rsplit("ANSWER:", 1)[-1].strip()

    return answerer


def _build_extractor(model: str, api_key: str, base_url: str | None, extra_body=None):
    """The K2 extractor client, with the same provider settings as the answerer.

    The signature deliberately matches ``_build_answerer`` exactly: the two are
    built side by side from the same CLI flags, and a differing parameter order
    is a swap waiting to happen -- one that authenticates with the model name
    and turns every failure into a silent "no claims".
    """
    from mirror_memory.llm import OpenAILLM

    return OpenAILLM(
        api_key=api_key, model=model, base_url=base_url, extra_body=extra_body,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the mirror-memory benchmark funnel")
    parser.add_argument("--dataset", required=True, help="Path to the LoCoMo JSON dataset")
    parser.add_argument(
        "--categories",
        default="",
        help="Comma-separated LoCoMo category numbers (1 multi-hop, 2 temporal, "
             "3 open-domain, 4 single-hop, 5 adversarial). Default: all.",
    )
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--config-path", default="config/")
    parser.add_argument("--database-url", default="sqlite://")
    parser.add_argument("--language", default="en")
    parser.add_argument("--output", default=None, help="Write the full JSON report here")
    parser.add_argument(
        "--no-answerer",
        action="store_true",
        help="Skip the answer stage (no LLM call). All other metrics still run.",
    )
    parser.add_argument("--answerer-model", default="deepseek-chat")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Send thinking.type=disabled. Reasoning models otherwise spend their "
             "token budget on thinking and can return an empty completion, which "
             "silently yields zero extracted claims.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    extra_body = {"thinking": {"type": "disabled"}} if args.no_thinking else None

    categories: set[int] | None = None
    if args.categories.strip():
        categories = {int(c) for c in args.categories.split(",") if c.strip()}

    cases = load_locomo_cases(
        args.dataset, categories=categories, max_conversations=args.max_conversations
    )
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        print("No cases matched the given filters.", file=sys.stderr)
        return 1

    print(f"loaded {len(cases)} cases: {category_counts(cases)}")

    answerer = None
    if not args.no_answerer:
        import os

        api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(
                "No API key found; running without the answer stage. "
                "Pass --no-answerer to silence this, or set DEEPSEEK_API_KEY.",
                file=sys.stderr,
            )
        else:
            answerer = _build_answerer(
                args.answerer_model, api_key, args.base_url, extra_body
            )

    # The extractor runs on the same provider so K2 actually fires.
    config = None
    if not args.no_answerer:
        import os

        api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            from mirror_memory.config.loader import load_config

            config = load_config(args.config_path)
            config.llm_client = _build_extractor(
                args.answerer_model, api_key, args.base_url, extra_body
            )

    report = run_cases(
        cases,
        config=config,
        config_path=None if config is not None else args.config_path,
        database_url=args.database_url,
        answerer=answerer,
        language=args.language,
    )

    print()
    print(render_report(report))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nfull report written to {out}")

        # A score without provenance cannot be attributed later, so the
        # manifest travels next to the report.
        manifest = RunManifest.build(
            run_id=out.stem,
            benchmark="locomo",
            dataset_path=args.dataset,
            cases=len(cases),
            categories=sorted({c.category for c in cases}),
            answerer_model=args.answerer_model if answerer else "",
            extractor_model=args.answerer_model if config is not None else "",
            base_url=args.base_url,
            thinking="disabled" if extra_body else "default",
            config_dir=args.config_path,
            generation_params={"temperature": 0.0, "max_tokens": 800},
            notes=";".join(filter(None, [
                f"max_cases={args.max_cases}" if args.max_cases else "",
                f"max_conversations={args.max_conversations}" if args.max_conversations else "",
                "no_answerer" if args.no_answerer else "",
            ])),
        )
        mpath = write_manifest(manifest, manifest_path_for(out))
        print(f"run manifest written to {mpath}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
