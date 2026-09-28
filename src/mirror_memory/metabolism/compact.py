"""Evidence compaction — which observations survive, as pure rules.

Fifty "I like coffee" messages are one fact with fifty observations.  A
belief whose support evidence exceeds ``min_support_links`` gets the bulk
folded into an :class:`~mirror_memory.core.models.EvidenceDigest` while a
small representative sample stays live::

    keep:  earliest 1 ─┐
           latest 3  ─┼─ every correct / contradict / verify link always
           top-authority 2 ┘  (a user correction is never compacted away)
    fold:  every other live support row → retention_state="compacted",
           content cleared, counted in the digest

Already-folded rows stay folded: re-compaction only processes rows that
went live again (re-observed messages un-fold themselves), so the digest's
counts accumulate across runs while the hot store never re-bloats.

The functions here are pure — the planner proposes from them, and the
Publisher recomputes the same selection as a commit gate, so a stale or
buggy plan cannot fold the wrong rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import NamedTuple

# Compaction starts strictly above this many support links on one belief.
MIN_SUPPORT_LINKS = 12

# Authority ranking for the "top-authority" keep slot: the user's own words
# outrank everything; derived signals rank last.
AUTHORITY_RANK = {
    "user": 0,
    "system": 1,
    "assistant": 2,
    "derived": 3,
}

# Relations whose links are never foldable, regardless of policy: a user
# correction, a contradiction, or a verification is load-bearing by itself.
ALWAYS_KEEP_RELATIONS = frozenset({"correct", "contradict", "verify"})


class EvidenceGraphRow(NamedTuple):
    """One typed link joined with its evidence row (pure view of the graph)."""

    evidence_id: int
    relation: str
    authority: str
    source_type: str
    observed_at: datetime | None
    retention_state: str


@dataclass(frozen=True)
class KeepPolicy:
    """How many support rows survive compaction, per rule.

    The three slots may overlap ("the latest message is also top-authority")
    — the union is kept, so the sample is at most ``oldest + recent +
    highest_authority`` rows.
    """

    oldest: int = 1
    recent: int = 3
    highest_authority: int = 2

    def as_dict(self) -> dict[str, int]:
        return {
            "oldest": self.oldest,
            "recent": self.recent,
            "highest_authority": self.highest_authority,
        }


DEFAULT_KEEP_POLICY = KeepPolicy()


def clamp_policy(raw: dict | None) -> KeepPolicy:
    """Coerce a payload/config policy into sane bounds.

    Every slot keeps at least one row — a policy that kept nothing would
    fold the entire sample, and the sample is what makes the digest
    auditable.  Used by the Publisher on proposal payloads.
    """
    raw = raw or {}
    policy = KeepPolicy(
        oldest=int(raw.get("oldest", 1)),
        recent=int(raw.get("recent", 3)),
        highest_authority=int(raw.get("highest_authority", 2)),
    )
    return KeepPolicy(
        oldest=min(max(policy.oldest, 1), 10),
        recent=min(max(policy.recent, 1), 10),
        highest_authority=min(max(policy.highest_authority, 1), 10),
    )


@dataclass
class CompactionPlan:
    """One belief's compaction decision (computed, not committed)."""

    belief_id: int
    keep_ids: tuple[int, ...] = ()
    fold_ids: tuple[int, ...] = ()
    relation_counts: dict[str, int] = field(default_factory=dict)

    @property
    def folds_anything(self) -> bool:
        return bool(self.fold_ids)


def _naive(observed_at: datetime | None) -> datetime | None:
    """Normalise to naive UTC so sorting never mixes aware and naive."""
    if observed_at is None:
        return None
    if observed_at.tzinfo is None:
        return observed_at
    return observed_at.astimezone(UTC).replace(tzinfo=None)


def _time_key(row: EvidenceGraphRow) -> tuple[int, datetime, int]:
    """Observed-at ascending; undated rows sort oldest, id breaks ties."""
    naive = _naive(row.observed_at)
    if naive is None:
        return (0, datetime.min, row.evidence_id)
    return (1, naive, row.evidence_id)


def plan_compaction(
    graph_rows: list[EvidenceGraphRow],
    belief_id: int,
    *,
    policy: KeepPolicy | None = None,
    min_support: int = MIN_SUPPORT_LINKS,
) -> CompactionPlan | None:
    """Decide what to keep and what to fold for one belief.

    ``min_support``: compaction requires **more** than this many support
    links in total (folded history included — re-compaction does not
    re-trigger below the threshold).

    Returns ``None`` when the belief is below the threshold; a plan with
    empty ``fold_ids`` when everything live is already worth keeping.
    """
    keep_policy = policy or DEFAULT_KEEP_POLICY
    rows = list(graph_rows)
    support_rows = [r for r in rows if r.relation == "support"]
    if len(support_rows) <= min_support:
        return None

    counts: dict[str, int] = {}
    for r in rows:
        counts[r.relation] = counts.get(r.relation, 0) + 1

    # Protected relations are unconditionally kept; only live support rows
    # are selection candidates (folded rows stay folded).
    keep: set[int] = {
        r.evidence_id for r in rows if r.relation in ALWAYS_KEEP_RELATIONS
    }
    live_support = [r for r in support_rows if r.retention_state != "compacted"]

    by_time = sorted(live_support, key=_time_key)
    if keep_policy.oldest > 0:
        keep.update(r.evidence_id for r in by_time[: keep_policy.oldest])
    if keep_policy.recent > 0:
        keep.update(r.evidence_id for r in by_time[-keep_policy.recent :])
    by_authority = sorted(
        live_support,
        key=lambda r: (AUTHORITY_RANK.get(r.authority, 99), r.evidence_id),
    )
    if keep_policy.highest_authority > 0:
        keep.update(r.evidence_id for r in by_authority[: keep_policy.highest_authority])

    fold_ids = tuple(
        sorted(r.evidence_id for r in live_support if r.evidence_id not in keep)
    )
    return CompactionPlan(
        belief_id=belief_id,
        keep_ids=tuple(sorted(keep)),
        fold_ids=fold_ids,
        relation_counts=counts,
    )


def build_digest_fields(
    graph_rows: list[EvidenceGraphRow], keep_ids
) -> dict:
    """Aggregate stats for the digest row (pure).

    Counts come from the **full** link graph — representatives and folded
    history alike — so repeated compactions accumulate rather than
    overwrite, and the digest always states the belief's whole evidence
    history.
    """
    rows = list(graph_rows)
    counts: dict[str, int] = {}
    sources: dict[str, int] = {}
    authorities: dict[str, int] = {}
    observed: list[datetime] = []
    for r in rows:
        counts[r.relation] = counts.get(r.relation, 0) + 1
        if r.relation == "support":
            sources[r.source_type] = sources.get(r.source_type, 0) + 1
            authorities[r.authority] = authorities.get(r.authority, 0) + 1
        naive = _naive(r.observed_at)
        if naive is not None:
            observed.append(naive)

    support_total = counts.get("support", 0)
    first = min(observed) if observed else None
    last = max(observed) if observed else None

    top_source = max(sources, key=sources.get) if sources else ""
    summary = ""
    if support_total:
        span = f" since {first:%Y-%m}" if first is not None else ""
        summary = (
            f"{support_total} observation{'s' if support_total != 1 else ''}{span}, "
            f"kept sample: {len(list(keep_ids))}"
            + (f", source: {top_source}" if top_source else "")
        )

    return {
        "support_count": counts.get("support", 0),
        "contradict_count": counts.get("contradict", 0),
        "verify_count": counts.get("verify", 0),
        "correct_count": counts.get("correct", 0),
        "first_seen_at": first,
        "last_seen_at": last,
        "representative_ids": sorted(keep_ids),
        "summary": summary,
        "source_distribution": dict(sorted(sources.items())),
        "authority_distribution": dict(sorted(authorities.items())),
    }
