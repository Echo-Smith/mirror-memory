"""LoCoMo adapter -- turn the dataset into benchmark cases with real evidence.

LoCoMo ships an explicit ``evidence`` field per question (dia-ids such as
``"D1:3"``), which is exactly what the funnel needs: the ground-truth fact is
the turn the answer was drawn from, so extraction recall is measured against
a known source rather than guessed from the answer string.

Category names follow the LoCoMo paper: 1 multi-hop, 2 temporal, 3 open-domain,
4 single-hop, 5 adversarial.
"""

from __future__ import annotations

import json
from pathlib import Path

from mirror_memory.bench.runner import BenchCase

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


def _session_keys(conversation: dict) -> list[str]:
    return sorted(
        (k for k in conversation if k.startswith("session_") and not k.endswith("_date_time")),
        key=lambda k: int(k.split("_")[1]),
    )


def _turn_index(conversation: dict) -> dict[str, str]:
    """Map every dia-id in the conversation to its text."""
    index: dict[str, str] = {}
    for key in _session_keys(conversation):
        for turn in conversation.get(key) or []:
            dia_id = turn.get("dia_id")
            text = (turn.get("text") or "").strip()
            if dia_id and text:
                index[dia_id] = text
    return index


def _conversation_turns(conversation: dict) -> list[str]:
    turns: list[str] = []
    for key in _session_keys(conversation):
        for turn in conversation.get(key) or []:
            text = (turn.get("text") or "").strip()
            if text:
                turns.append(text)
    return turns


def load_locomo_cases(
    path: str | Path,
    *,
    categories: set[int] | None = None,
    max_conversations: int | None = None,
) -> list[BenchCase]:
    """Load LoCoMo QA pairs as :class:`BenchCase` instances.

    Parameters
    ----------
    path:
        Path to the LoCoMo JSON file (a list of conversation entries).
    categories:
        Restrict to these LoCoMo category numbers.  ``None`` keeps all.
    max_conversations:
        Only read the first N conversations (useful for a smoke run).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("data") or raw.get("conversations") or []

    cases: list[BenchCase] = []
    for conv_idx, entry in enumerate(raw):
        if max_conversations is not None and conv_idx >= max_conversations:
            break
        conversation = entry.get("conversation") or {}
        turns = _conversation_turns(conversation)
        if not turns:
            continue
        dia_index = _turn_index(conversation)

        questions = entry.get("qa") or entry.get("qa_pairs") or []
        for qi, qa in enumerate(questions):
            category = qa.get("category")
            if categories is not None and category not in categories:
                continue
            answer = str(qa.get("answer", "")).strip()
            if not answer:
                continue

            # Evidence is the turn(s) the answer was drawn from.  Fall back to
            # the answer text itself so the case is still scoreable.
            evidence_turns = [
                dia_index[d] for d in (qa.get("evidence") or []) if d in dia_index
            ]
            evidence = " ".join(evidence_turns) or answer

            cases.append(
                BenchCase(
                    case_id=f"conv{conv_idx}_q{qi}",
                    category=CATEGORY_NAMES.get(category, f"cat{category}"),
                    turns=turns,
                    question=str(qa.get("question", "")).strip(),
                    answer=answer,
                    evidence=evidence,
                    # Every question from one conversation shares the ingest.
                    group=f"conv{conv_idx}",
                )
            )
    return cases


def category_counts(cases: list[BenchCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.category] = counts.get(case.category, 0) + 1
    return dict(sorted(counts.items()))
