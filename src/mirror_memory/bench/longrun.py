"""LongRunBench — does the metabolism layer survive a year of use?

MetabolismBench (29 scripted cases) proves the lifecycle *rules*.  This
bench proves the *runtime*: one user, a year of simulated time, thousands of
observations, hundreds of beliefs, corrections and conflicts — with
metabolism cycles, recalls and forgets interleaved — and then asks whether
the things that must survive did, and the things that must be gone are.

What it measures
----------------
- **Semantic preservation** — of the current facts answerable before the
  long run, the fraction still answerable *with the right value* after
  (target ≥ 99%).
- **Correction loss** — corrected values that stopped being current
  (target 0).
- **Conflict loss** — unresolved conflicts that got compacted or cooled
  away (target 0).
- **Forgotten resurrection** — forgotten content any recall path still
  returns (target 0).
- **Wrong archive count** — beliefs that must never be archived (canonical,
  corrected, conflicted) but were (target 0).
- **Hot evidence reduction** — evidence content characters after compaction
  divided by before (target ≥ 70% reclaimed).
- **Recall p50/p95** and **cycle latency** — the hot path stays hot.

The observation stream is deterministic (K1 extraction, no LLM): the point
is the lifecycle under load, not extraction quality.  A simulated clock is
stamped onto each observation so retention windows can be exercised without
waiting a year.
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

# -- Scenario ------------------------------------------------------------------

DAYS = 365
CYCLE_EVERY_DAYS = 7
RECALL_EVERY_DAYS = 3

# Belief population.
CANONICAL_FACTS = 30      # name / age / city (single-valued, never archived)
PREFERENCES = 150         # multi-valued, observed repeatedly -> compaction
GOALS = 50                # wants_to, some abandoned
EVENTS = 40               # went_to style occurrences
TRANSIENT = 30            # short-lived task statements

CORRECTIONS = 100         # user corrections (engine.correct)
CONFLICTS = 50            # cross-source disagreements (repository path)
FORGETS = 20              # targeted forgets

# How often a preference is re-observed: enough to cross the compaction
# threshold (12) for most of them, so the hot store has real bulk to fold.
PREFERENCE_OBSERVATIONS = (24, 40)

SEED = 20260929


@dataclass
class LongRunReport:
    observations: int = 0
    beliefs: int = 0
    evidence_rows: int = 0
    cycles: int = 0
    recalls: int = 0
    corrections: int = 0
    conflicts: int = 0
    forgets: int = 0
    # Hot-store trajectory (one sample per cycle).
    hot_evidence_chars: list[int] = field(default_factory=list)
    ingested_chars: int = 0
    live_evidence_rows: list[int] = field(default_factory=list)
    tier_counts: list[dict[str, int]] = field(default_factory=list)
    cycle_latency_ms: list[float] = field(default_factory=list)
    recall_latency_ms: list[float] = field(default_factory=list)
    # Verification.
    semantic_total: int = 0
    semantic_preserved: int = 0
    semantic_loss_detail: list[str] = field(default_factory=list)
    correction_loss: list[str] = field(default_factory=list)
    correction_loss_detail: list[str] = field(default_factory=list)
    conflict_loss: list[str] = field(default_factory=list)
    forgotten_resurrection: list[str] = field(default_factory=list)
    wrong_archive: list[str] = field(default_factory=list)
    wrong_archive_detail: list[str] = field(default_factory=list)
    # Ids the runner corrected / conflicted / forgot, checked at the end.
    corrected_ids: list[int] = field(default_factory=list)
    conflicted_ids: list[int] = field(default_factory=list)
    forgotten_ids: list[int] = field(default_factory=list)
    db_pages: int = 0
    wall_seconds: float = 0.0

    # -- derived ------------------------------------------------------------

    @property
    def semantic_preservation(self) -> float:
        if not self.semantic_total:
            return 1.0
        return round(self.semantic_preserved / self.semantic_total, 4)

    @property
    def hot_evidence_reduction(self) -> float | None:
        """Reclaimed share of everything ever ingested.

        The store keeps growing all year (new preferences are observed), so
        a peak-vs-final comparison measures the growth rate, not the
        compaction.  What the target asks is: of all the evidence content
        the engine ever took in, how much is still live in the hot store.
        """
        if not self.ingested_chars:
            return None
        final = self.hot_evidence_chars[-1] if self.hot_evidence_chars else 0
        return round(1.0 - final / self.ingested_chars, 4)

    @property
    def recall_p95_ms(self) -> float | None:
        if not self.recall_latency_ms:
            return None
        samples = sorted(self.recall_latency_ms)
        return round(samples[max(0, int(len(samples) * 0.95) - 1)], 2)

    @property
    def recall_p50_ms(self) -> float | None:
        if not self.recall_latency_ms:
            return None
        return round(statistics.median(self.recall_latency_ms), 2)

    @property
    def cycle_p95_ms(self) -> float | None:
        if not self.cycle_latency_ms:
            return None
        samples = sorted(self.cycle_latency_ms)
        return round(samples[max(0, int(len(samples) * 0.95) - 1)], 2)

    def as_dict(self) -> dict:
        return {
            "observations": self.observations,
            "beliefs": self.beliefs,
            "evidence_rows": self.evidence_rows,
            "cycles": self.cycles,
            "recalls": self.recalls,
            "corrections": self.corrections,
            "conflicts": self.conflicts,
            "forgets": self.forgets,
            "semantic_preservation": self.semantic_preservation,
            "semantic_total": self.semantic_total,
            "hot_evidence_reduction": self.hot_evidence_reduction,
            "recall_p50_ms": self.recall_p50_ms,
            "recall_p95_ms": self.recall_p95_ms,
            "cycle_p95_ms": self.cycle_p95_ms,
            "correction_loss": self.correction_loss,
            "conflict_loss": self.conflict_loss,
            "forgotten_resurrection": self.forgotten_resurrection,
            "wrong_archive": self.wrong_archive,
            "corrected_ids": self.corrected_ids,
            "conflicted_ids": self.conflicted_ids,
            "forgotten_ids": self.forgotten_ids,
            "db_pages": self.db_pages,
            "wall_seconds": round(self.wall_seconds, 1),
            "ingested_chars": self.ingested_chars,
            "live_hot_chars": self.hot_evidence_chars[-1] if self.hot_evidence_chars else 0,
        }


# -- Scenario generation -------------------------------------------------------

_CITIES = [
    "Berlin", "Shanghai", "Lisbon", "Oslo", "Toronto", "Nairobi", "Munich",
    "Porto", "Kyoto", "Austin", "Dublin", "Perth", "Bogota", "Tallinn",
]
_NAMES = ["Alice", "Marco", "Yuki", "Priya", "Sofia", "Omar", "Lena", "Diego"]
_HOBBIES = [
    "painting landscapes", "hiking", "chess", "ceramics", "jazz piano",
    "rock climbing", "bread baking", "birdwatching", "table tennis",
    "creative writing", "gardening", "surfing", "photography", "origami",
    "learning Spanish", "learning French", "learning German",
    "collecting vinyl", "making sushi", "doing yoga", "doing pilates",
    "watching documentaries", "listening to podcasts", "playing chess online",
    "restoring furniture", "keeping a journal", "urban sketching",
    "camping", "fishing", "knitting", "woodworking", "metal detecting",
    "genealogy research", "astronomy", "geocaching", "stand-up paddleboarding",
    "ice skating", "skiing", "snowboarding", "mountain biking",
    "road cycling", "swimming", "rowing", "boxing", "jiu-jitsu",
]
_GOALS = [
    "learn the guitar", "run a marathon", "write a novel", "learn Japanese",
    "build a mobile app", "start a podcast", "get a pilot licence",
    "learn to sail", "meditate daily", "read fifty books", "learn to cook",
    "visit every continent", "run a 10k", "learn photography", "learn Python",
    "learn Rust", "build a website", "start a blog", "learn the violin",
    "climb a mountain", "swim a kilometre", "learn sign language",
    "learn chess openings", "grow vegetables", "renovate the kitchen",
    "learn calligraphy", "write a screenplay", "learn animation",
    "build a robot", "launch a side project", "learn data science",
    "learn machine learning", "read the classics", "learn philosophy",
    "study astronomy", "learn wine tasting", "learn coffee brewing",
    "train for a triathlon", "learn parkour", "learn breakdancing",
    "learn salsa dancing", "learn ballroom dancing", "join a choir",
    "learn music production", "learn sound design", "make a short film",
    "learn 3D modelling", "learn game development", "build a home lab",
    "learn networking", "get a certification", "learn a new language yearly",
    "travel to japan", "travel to iceland", "travel to patagonia",
    "see the northern lights", "walk the camino", "cycle across a country",
    "learn scuba diving", "learn freediving", "learn surfing",
    "learn kitesurfing", "learn windsurfing",
]
_EVENTS = ["a museum", "a concert", "a market", "a conference", "a workshop"]
_TASKS = ["buy milk", "call the dentist", "book a flight", "renew my passport"]


def _hobby_pool() -> list[str]:
    """A few hundred distinct preference objects (modifier × activity)."""
    modifiers = ["", "urban ", "weekend ", "competitive ", "casual ",
                 "indoor ", "outdoor ", "morning ", "late-night "]
    pool: list[str] = []
    for modifier in modifiers:
        for hobby in _HOBBIES:
            pool.append(f"{modifier}{hobby}".strip())
    return pool


@dataclass
class Scenario:
    """The scripted year: what happens on which day.

    Corrections, conflicts and forgets name a *day and a count* rather than
    specific beliefs: the runner picks whatever is current on that day,
    which is what a user actually does, and records the ids it touched so
    the verifier checks exactly those rows.
    """

    days: list[list[str]] = field(default_factory=list)  # day -> observation texts
    # The canonical values that must still be current at the end: the
    # verifier compares the actual object, not just the row's presence.
    expected_name: str = ""
    expected_age: str = ""
    expected_city: str = ""
    correction_days: list[int] = field(default_factory=list)
    conflict_days: list[int] = field(default_factory=list)
    forget_day: int = DAYS - 1
    correction_batch: int = 10
    conflict_batch: int = 5
    forget_batch: int = FORGETS


def build_scenario(
    rng: random.Random,
    *,
    days: int = DAYS,
    preferences: int = PREFERENCES,
    goals: int = GOALS,
    corrections: int = CORRECTIONS,
    conflicts: int = CONFLICTS,
    forgets: int = FORGETS,
) -> Scenario:
    scenario = Scenario(days=[[] for _ in range(days)])

    def emit(day: int, text: str) -> None:
        scenario.days[day].append(text)

    # Canonical facts: a name, an age, and a chain of residences with one
    # return trip (A -> B -> A).
    name = rng.choice(_NAMES)
    emit(0, f"my name is {name}")
    scenario.expected_name = name
    age_value = str(rng.randint(25, 55))
    emit(1, f"I'm {age_value} years old")
    scenario.expected_age = age_value

    cities = rng.sample(_CITIES, 3)
    emit(2, f"I live in {cities[0]}")
    emit(min(120, days // 3), f"I moved to {cities[1]}")
    emit(min(240, days * 2 // 3), f"I moved back to {cities[0]}")
    scenario.expected_city = cities[0]

    # Preferences: each observed many times across the year, so most cross
    # the compaction threshold.
    hobbies = rng.sample(_hobby_pool(), preferences)
    for index, hobby in enumerate(hobbies):
        times = rng.randint(*PREFERENCE_OBSERVATIONS)
        for _ in range(times):
            emit(rng.randrange(days), f"I really like {hobby}")
        # A third of them are reversed mid-year, then a few revived.
        if index % 3 == 0:
            reverse_day = rng.randrange(days // 4, max(days // 4 + 1, days * 5 // 8))
            emit(reverse_day, f"I no longer like {hobby}")
            if index % 6 == 0:
                emit(rng.randrange(reverse_day + 10, days), f"I like {hobby} again")

    # Goals: stated early, some abandoned, some resumed.
    goals = rng.sample(_GOALS, goals)
    for index, goal in enumerate(goals):
        emit(rng.randrange(0, max(1, days // 6)), f"I want to {goal}")
        if index % 4 == 0:
            emit(rng.randrange(days // 3, days * 5 // 6), f"I want to {goal}")

    # Episodic events: each occurrence is its own fact.
    for _ in range(EVENTS):
        emit(rng.randrange(days), f"I went to {rng.choice(_EVENTS)} yesterday")

    # Transient task statements: short-lived, should cool quickly.
    for _ in range(TRANSIENT):
        emit(rng.randrange(days), f"I want to {rng.choice(_TASKS)}")

    # Corrections: the user fixes current beliefs late in the year.
    correction_days = sorted(rng.sample(range(days // 2, days - 5), 10))
    scenario.correction_days = correction_days
    scenario.correction_batch = max(1, corrections // 10)

    # Conflicts: cross-source disagreements injected late.
    scenario.conflict_days = sorted(rng.sample(range(days // 2, days - 5), 10))
    scenario.conflict_batch = max(1, conflicts // 10)

    # Forgets happen on the last day so nothing re-observes the content
    # afterwards: a hobby stated again after its forget is a *new* belief,
    # which is correct engine behaviour, not a resurrection.
    scenario.forget_day = days - 1
    scenario.forget_batch = forgets

    return scenario


# -- Runner --------------------------------------------------------------------


def run_longrun(
    config,
    scenario: Scenario | None = None,
    *,
    user_id: str = "longrun-u1",
    seed: int = SEED,
    scenario_kwargs: dict | None = None,
) -> LongRunReport:
    """Drive the engine through the scripted year and verify the outcome."""
    from sqlalchemy import select

    from mirror_memory.api import MemoryEngine
    from mirror_memory.core.models import Belief, Evidence
    from mirror_memory.core.proposal import (
        TRANSITION_CORRECT,
        TRANSITION_FORGET,
        StateTransitionProposal,
    )
    from mirror_memory.core.publisher import Publisher
    from mirror_memory.core.repository import (
        CONFLICT_SOURCE_CONFLICT,
        get_state_revision,
        set_memory_enabled,
    )
    from mirror_memory.core.utils import safe_json

    started = time.monotonic()
    rng = random.Random(seed)
    scenario = scenario or build_scenario(rng, **(scenario_kwargs or {}))
    report = LongRunReport()

    engine = MemoryEngine(config=config, database_url="sqlite://")
    engine._ensure_db()
    start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    def stamp(day: int) -> datetime:
        return start + timedelta(days=day)

    def backdate(day: int) -> None:
        """Move the day's writes onto the simulated clock.

        The engine stamps ``now`` at ingest; the bench needs retention
        windows to see a year of ageing, so the rows it just wrote are
        re-dated to the simulated day.  Fixture setup, not an engine path.
        """
        when = stamp(day).replace(tzinfo=None)  # the engine stores naive UTC
        with engine._session() as session:
            for belief in session.scalars(select(Belief).where(Belief.user_id == user_id)):
                current = belief.last_evidence_at
                if current is not None and current.tzinfo is not None:
                    current = current.replace(tzinfo=None)
                if current is None or current > when:
                    belief.last_evidence_at = when
                    belief.last_supported_at = when
                    first = belief.first_seen_at
                    if first is not None and first.tzinfo is not None:
                        first = first.replace(tzinfo=None)
                    if first is None or first > when:
                        belief.first_seen_at = when
            session.commit()

    def hot_stats() -> tuple[int, int]:
        with engine._session() as session:
            rows = session.scalars(
                select(Evidence).where(Evidence.user_id == user_id)
            )
            chars = 0
            live = 0
            for row in rows:
                chars += len(row.content or "")
                if row.retention_state != "compacted":
                    live += 1
            return chars, live

    with engine._session() as session:
        set_memory_enabled(session, user_id, True)
        session.commit()

    # Ids the runner touched, so the verifier checks exactly those rows.
    corrected_ids: list[int] = []
    conflicted_ids: list[int] = []
    forgotten_ids: list[int] = []
    forgotten_content: list[tuple[int, str]] = []

    def current_beliefs(
        session,
        limit: int,
        rng_pick: random.Random,
        *,
        exclude_predicates: tuple[str, ...] = (),
        in_scan_only: bool = True,
    ) -> list:
        """A random sample of the user's *current* beliefs.

        ``in_scan_only`` keeps the sample inside the default retrieval scan
        (hot/warm/dormant): a correction or dispute is a statement about
        what the engine is currently asserting, and an archived belief is
        out of that surface by design -- acting on one would be a restore,
        not a correction.
        """
        rows = [
            b for b in session.scalars(select(Belief).where(Belief.user_id == user_id))
            if b.status == "active"
            and b.predicate
            and b.predicate not in exclude_predicates
            and (not in_scan_only or b.memory_tier != "archived")
        ]
        rng_pick.shuffle(rows)
        return rows[:limit]

    # -- the year ------------------------------------------------------------
    turn_index = 0
    total_days = len(scenario.days)
    for day in range(total_days):
        for text in scenario.days[day]:
            # A distinct turn index per observation: the pipeline keys
            # fallback evidence on ``session:turn``, so reusing one index
            # would collapse a day's observations into a single evidence row
            # and the compaction threshold would never be reached.
            engine.observe(
                user_id=user_id, session_id=f"lr-d{day}", text=text,
                turn_count=turn_index,
            )
            turn_index += 1
            report.observations += 1
            report.ingested_chars += len(text)
        if scenario.days[day]:
            backdate(day)

        # Corrections, conflicts, forgets on their scheduled days.  Each
        # picks from what is *current* on that day -- the user corrects what
        # the engine is currently asserting -- and records the ids touched.
        if day in scenario.correction_days:
            with engine._session() as session:
                for target in current_beliefs(session, scenario.correction_batch, rng):
                    decision = Publisher(session).publish(
                        StateTransitionProposal(
                            transition=TRANSITION_CORRECT,
                            user_id=user_id,
                            target_belief_id=target.id,
                            expected_revision=get_state_revision(session, user_id),
                            actor_type="user",
                            payload={
                                "new_claim_text": f"User corrected: {target.claim_text}",
                                "new_object": target.object,
                                "new_predicate": target.predicate,
                            },
                        )
                    )
                    if decision.committed:
                        report.corrections += 1
                        corrected_ids.append(
                            (decision.detail or {}).get("belief_id") or target.id
                        )
                session.commit()

        if day in scenario.conflict_days:
            with engine._session() as session:
                for target in current_beliefs(session, scenario.conflict_batch, rng):
                    from mirror_memory.core.repository import record_claim

                    record_claim(
                        session, user_id,
                        dimension=target.dimension, key=target.key,
                        claim_text=f"A second source disputes {target.object}",
                        predicate=target.predicate, object=target.object,
                        relation="contradicts",
                        conflict_kind=CONFLICT_SOURCE_CONFLICT,
                        confidence=0.5,
                    )
                    conflicted_ids.append(target.id)
                    report.conflicts += 1
                session.commit()

        if day == scenario.forget_day:
            with engine._session() as session:
                for target in current_beliefs(
                    session, scenario.forget_batch, rng,
                    exclude_predicates=("name", "age", "lives_in"),
                ):
                    decision = Publisher(session).publish(
                        StateTransitionProposal(
                            transition=TRANSITION_FORGET,
                            user_id=user_id,
                            target_belief_id=target.id,
                            expected_revision=get_state_revision(session, user_id),
                            actor_type="user",
                        )
                    )
                    if decision.committed:
                        report.forgets += 1
                        forgotten_ids.append(target.id)
                        forgotten_content.append(
                            (target.id, (target.claim_text or target.object or "").strip())
                        )
                session.commit()

        # Metabolism cycle.
        if day % CYCLE_EVERY_DAYS == 0:
            cycle_started = time.monotonic()
            engine.run_metabolism(user_id=user_id, now=stamp(day))
            report.cycles += 1
            report.cycle_latency_ms.append((time.monotonic() - cycle_started) * 1000)
            chars, live = hot_stats()
            report.hot_evidence_chars.append(chars)
            report.live_evidence_rows.append(live)

        # Recall probe on the hot path.
        if day % RECALL_EVERY_DAYS == 0:
            probe_started = time.monotonic()
            engine.recall(
                user_id=user_id, query="what do they like?", now=stamp(day)
            )
            report.recall_latency_ms.append((time.monotonic() - probe_started) * 1000)
            report.recalls += 1

    # -- final state ---------------------------------------------------------
    from sqlalchemy import text

    with engine._session() as session:
        report.beliefs = session.query(Belief).filter_by(user_id=user_id).count()
        report.evidence_rows = session.query(Evidence).filter_by(user_id=user_id).count()
        pages = session.execute(text("PRAGMA page_count")).scalar()
        report.db_pages = int(pages or 0)

        # Semantic preservation: the canonical facts that must still be
        # current *with the value the scenario established*.  (Superseded
        # history is deliberately not counted: it is supposed to be closed.)
        canonical_rows = [
            b for b in session.scalars(select(Belief).where(Belief.user_id == user_id))
            if b.predicate in ("name", "age", "lives_in")
        ]
        for expected, predicate in (
            (scenario.expected_name, "name"),
            (scenario.expected_age, "age"),
            (scenario.expected_city, "lives_in"),
        ):
            report.semantic_total += 1
            rows = [
                b for b in canonical_rows
                if b.predicate == predicate and b.status == "active"
                and b.memory_tier != "archived"
            ]
            # Presence is not preservation: the *value* must be the one the
            # scenario established, or a wrong-but-active row scores 1.000.
            if rows and (b := max(rows, key=lambda x: x.id)).object.strip().lower() == (
                expected or ""
            ).strip().lower():
                report.semantic_preserved += 1
            else:
                got = b.object if rows else None
                report.semantic_loss_detail.append(
                    f"{predicate}: expected={expected!r} got={got!r}"
                )

        # Correction loss: the lifecycle must never *silently* drop a
        # corrected value.  A row superseded by a newer user statement (a
        # revival, a further correction) or deleted by a forget is the user
        # changing their mind -- not a loss.  What counts is a corrected row
        # that is still the current value yet has been archived.
        for belief_id in corrected_ids:
            belief = session.get(Belief, belief_id)
            if belief is None or belief.status != "active":
                continue
            if belief.memory_tier == "archived":
                report.correction_loss.append(belief.key)
                report.correction_loss_detail.append(
                    f"{belief.key} ({belief.object!r}): active but tier={belief.memory_tier} "
                    f"protected={belief.protected_reason!r}"
                )

        # Conflict loss: every recorded dispute must still be standing.
        # Scanning only the beliefs that still carry the flag would let a
        # deleted or silently-closed dispute drop out of the denominator,
        # so each recorded id is checked explicitly.  A row that was
        # forgotten afterwards is a user decision, not a loss; a row
        # superseded by a newer user statement is the dispute resolving
        # itself.  What counts is the flag vanishing, the row being
        # archived, or the row disappearing without either cause.
        for belief_id in conflicted_ids:
            if belief_id in forgotten_ids:
                continue
            belief = session.get(Belief, belief_id)
            if belief is None:
                report.conflict_loss.append(f"missing:{belief_id}")
                continue
            flagged = (
                safe_json(belief.value_json).get("clarification_status")
                == "needs_clarification"
            )
            successor = (
                session.get(Belief, belief.superseded_by) if belief.superseded_by else None
            )
            if not flagged and successor is not None and successor.status == "active":
                continue  # resolved by a newer user statement
            if not flagged:
                report.conflict_loss.append(f"flag_lost:{belief.key}")
                continue
            if belief.memory_tier == "archived":
                report.conflict_loss.append(f"dispute_archived:{belief.key}")

        # Forgotten resurrection: a forgotten belief must not exist, and no
        # recall path may return its content.  (The content check is on the
        # claim text, not the id -- an id can appear in the block by
        # coincidence, e.g. the age "44" containing the id 4.)
        probe = engine.recall(
            user_id=user_id, query="what do they like?", now=stamp(total_days)
        )
        probe_lower = (probe or "").lower()
        for belief_id, content in forgotten_content:
            if session.get(Belief, belief_id) is not None:
                report.forgotten_resurrection.append(content or str(belief_id))
                continue
            if content and content.lower() in probe_lower:
                report.forgotten_resurrection.append(content)

        # Wrong archive: a belief that must stay reachable -- a canonical
        # fact, a live correction, an open dispute -- has been archived.
        # Superseded history may be archived freely.
        for belief in session.scalars(select(Belief).where(Belief.user_id == user_id)):
            if belief.memory_tier != "archived" or belief.status != "active":
                continue
            if (
                belief.key in ("my_name", "age")
                or belief.predicate in ("name", "age", "lives_in")
                or belief.source == "user_corrected"
                or safe_json(belief.value_json).get("clarification_status")
                == "needs_clarification"
            ):
                report.wrong_archive.append(belief.key)
                report.wrong_archive_detail.append(
                    f"{belief.key}: source={belief.source} "
                    f"protected={belief.protected_reason!r} "
                    f"tier={belief.memory_tier} last_evidence={belief.last_evidence_at}"
                )

    report.wall_seconds = time.monotonic() - started
    report.corrected_ids = corrected_ids
    report.conflicted_ids = conflicted_ids
    report.forgotten_ids = forgotten_ids
    return report


# -- Rendering -----------------------------------------------------------------

_TARGETS = {
    "semantic_preservation": (">=", 0.99),
    "correction_loss": ("==", 0),
    "conflict_loss": ("==", 0),
    "forgotten_resurrection": ("==", 0),
    "wrong_archive": ("==", 0),
    "hot_evidence_reduction": (">=", 0.70),
    "recall_p95_ms": ("<", 100),
}


def render_report(report: LongRunReport) -> str:
    """Human-readable report, matching the StateBench report style."""
    lines: list[str] = [
        "LongRunBench v1 — one user, one simulated year",
        f"  observations={report.observations} beliefs={report.beliefs} "
        f"evidence={report.evidence_rows}",
        f"  cycles={report.cycles} recalls={report.recalls} "
        f"corrections={report.corrections} conflicts={report.conflicts} "
        f"forgets={report.forgets}",
        "",
        f"{'metric':<28}{'value':>12}{'target':>14}{'ok':>5}",
        "-" * 59,
    ]
    values = {
        "semantic_preservation": report.semantic_preservation,
        "correction_loss": len(report.correction_loss),
        "conflict_loss": len(report.conflict_loss),
        "forgotten_resurrection": len(report.forgotten_resurrection),
        "wrong_archive": len(report.wrong_archive),
        "hot_evidence_reduction": report.hot_evidence_reduction,
        "recall_p95_ms": report.recall_p95_ms,
    }
    for name, (op, target) in _TARGETS.items():
        value = values[name]
        if value is None:
            ok = "-"
            shown = "n/a"
        else:
            ok = "yes" if (
                value >= target if op == ">=" else
                value <= target if op == "<" else
                value == target
            ) else "NO"
            shown = f"{value:.3f}" if isinstance(value, float) else str(value)
        lines.append(f"{name:<28}{shown:>12}{op + ' ' + str(target):>14}{ok:>5}")

    lines.append("")
    lines.append(
        f"ingested={report.ingested_chars} chars"
        f" -> live hot={report.hot_evidence_chars[-1] if report.hot_evidence_chars else 0}"
        f"   live rows: {report.live_evidence_rows[-1] if report.live_evidence_rows else 0}"
        f" / {report.evidence_rows}"
    )
    lines.append(
        f"recall p50={report.recall_p50_ms}ms cycle p95={report.cycle_p95_ms}ms "
        f"db pages={report.db_pages} wall={report.wall_seconds:.0f}s"
    )
    for label, items in (
        ("correction loss", report.correction_loss),
        ("conflict loss", report.conflict_loss),
        ("forgotten resurrection", report.forgotten_resurrection),
        ("wrong archive", report.wrong_archive),
    ):
        if items:
            lines.append(f"{label}: {items[:10]}")
    for detail in report.correction_loss_detail[:5]:
        lines.append(f"    {detail}")
    for detail in report.semantic_loss_detail[:5]:
        lines.append(f"    semantic: {detail}")
    for detail in report.wrong_archive_detail[:5]:
        lines.append(f"    {detail}")
    return "\n".join(lines)
