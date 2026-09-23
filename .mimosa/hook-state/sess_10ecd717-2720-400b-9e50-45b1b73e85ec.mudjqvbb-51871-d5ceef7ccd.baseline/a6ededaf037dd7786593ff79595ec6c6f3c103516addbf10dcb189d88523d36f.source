"""Dynamic character-budget computation -- pure function, zero domain coupling.

Budget = clamp(BASE * (1.2 - 0.4 * pressure), FLOOR, CAP)

pressure = 0.7 * min(tail_load / NOMINAL_TAIL_LOAD, 1)
         + 0.3 * min(knowledge_proxy, 1)

- tail_load: character count of history/summary already occupying the prompt.
- knowledge_proxy: normalised [0, 1] signal of how much the current turn
  already hits known topics (e.g. topic_count / MAX_TOPICS).

Deterministic: same input -> same output.
"""

from __future__ import annotations

# -- Budget constants --------------------------------------------------------

NOMINAL_TAIL_LOAD = 1800   # "full" history+summary character count
BASE_MULTIPLIER_HIGH = 1.2 # pressure=0 amplification
BASE_MULTIPLIER_LOW = 0.8  # pressure=1 contraction
W_TAIL = 0.7               # weight of tail-load pressure
W_KNOWLEDGE = 0.3          # weight of knowledge-proxy pressure


def compute_profile_budget(
    tail_load: int,
    knowledge_proxy: float,
    *,
    base: int,
    floor: int,
    cap: int,
) -> int:
    """Return the dynamic character budget for the current render.

    Parameters
    ----------
    tail_load:
        Character count of content already occupying the prompt window
        (history, summaries, etc.).
    knowledge_proxy:
        Normalised [0, 1] signal of how much the current turn already
        covers known topics.  Typically ``topic_count / max_topics``.
    base:
        Nominal budget at zero pressure.
    floor:
        Minimum budget (never shrink below this).
    cap:
        Maximum budget (never grow above this).
    """
    tail_pressure = min(max(tail_load, 0) / NOMINAL_TAIL_LOAD, 1.0)
    proxy = min(max(knowledge_proxy, 0.0), 1.0)
    pressure = W_TAIL * tail_pressure + W_KNOWLEDGE * proxy
    raw = base * (BASE_MULTIPLIER_HIGH - (BASE_MULTIPLIER_HIGH - BASE_MULTIPLIER_LOW) * pressure)
    return int(max(floor, min(cap, round(raw))))
