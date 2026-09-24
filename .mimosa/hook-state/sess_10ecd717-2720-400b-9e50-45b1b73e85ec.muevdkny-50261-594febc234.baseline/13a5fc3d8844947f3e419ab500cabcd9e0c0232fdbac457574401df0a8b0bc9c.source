"""Memory module — Atom identity, canonicalization, and resolution.

This module provides the cognitive triple (Subject → Predicate → Object)
model and the IdentityResolver that decides what happens when a new
CandidateAtom arrives (CREATE / SUPPORT / UPDATE / CONTRADICT / NOOP).

Usage::

    from mirror_memory.memory import CandidateAtom, Resolution
    from mirror_memory.memory.identity import resolve_identity
    from mirror_memory.memory.canonicalize import canonicalize_atom
"""

from mirror_memory.memory.atom import (
    ACTION_CONTRADICT,
    ACTION_CREATE,
    ACTION_NOOP,
    ACTION_SUPPORT,
    ACTION_UPDATE,
    CARDINALITY_EVENT,
    CARDINALITY_MULTI,
    CARDINALITY_SINGLE,
    CandidateAtom,
    Resolution,
)

__all__ = [
    "CandidateAtom",
    "Resolution",
    "ACTION_CREATE",
    "ACTION_SUPPORT",
    "ACTION_UPDATE",
    "ACTION_CONTRADICT",
    "ACTION_NOOP",
    "CARDINALITY_SINGLE",
    "CARDINALITY_MULTI",
    "CARDINALITY_EVENT",
]
