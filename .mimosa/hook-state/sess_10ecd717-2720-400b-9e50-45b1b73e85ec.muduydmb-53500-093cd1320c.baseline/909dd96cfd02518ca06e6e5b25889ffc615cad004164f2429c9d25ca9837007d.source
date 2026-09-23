"""LongMemEval ingestion regression: no user may be lost to a key collision.

The prior benchmark run ingested 481 of 500 users before dying at user 482 on
``UNIQUE constraint failed: mm_beliefs.user_id, mm_beliefs.key`` while
persisting ``dim=fact key=works_as_digital_marketing_specialist``.  The
benchmark's response was to skip the failed user, so 19 users silently lost
their entire memory.

Two things are pinned here:

1. A **collision storm** — a mock extractor that deliberately reuses one key
   for many different values of the same predicate, which is the shape of the
   failure.  Every user must ingest cleanly.
2. The **real dataset**, when present, ingested end to end.

The real-dataset test skips when the oracle file is absent so the suite stays
runnable anywhere; the collision storm needs no data at all.
"""

import json
from pathlib import Path

import pytest

from mirror_memory.config.loader import load_config
from mirror_memory.core.models import Base, Belief
from mirror_memory.core.repository import set_memory_enabled
from mirror_memory.extraction.pipeline import ExtractionPipeline

ORACLE = Path("/tmp/LongMemEval/data/longmemeval_oracle.json")


class CollidingLLM:
    """An extractor that reuses one key for every value of a predicate.

    This is the failure shape: the model is told the predicate and object but
    keys its claim on the predicate alone, so every new job, city, or hobby
    arrives under the same key.  ``profession`` is used because the shipped
    policy declares it ``single``/``current_state`` -- that is what routes the
    claim into ``update_belief_by_id``, the path that used to crash.
    """

    def __init__(self):
        self.calls = 0

    def generate(self, *, system_prompt, payload_text, fallback=None):
        self.calls += 1
        text = payload_text.split("[User utterance]")[-1].strip()
        if not text:
            return '{"claims": []}'
        # A fresh object per call, always under the same key.
        import json as _json

        return _json.dumps({
            "claims": [{
                "dimension": "fact",
                "key": "works_as_role",
                "claim_text": text[:200],
                "confidence": 0.6,
                "relation": "supports",
                "predicate": "profession",
                "object": f"role_{self.calls}",
            }],
            "subject": "user",
            "context_tags": [],
        })


def _profession_rows(session):
    """The beliefs belonging to the colliding single-cardinality chain."""
    return [b for b in session.query(Belief).all() if b.predicate == "profession"]


class TestCollisionStorm:
    @pytest.fixture
    def engine(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        cfg = load_config("config/")
        cfg.llm_client = CollidingLLM()
        # Always extract, so every turn drives the UPDATE path.
        cfg.extraction.llm_min_keyword_hits = 0

        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        return cfg, Session()

    def test_repeated_updates_under_one_key_all_succeed(self, engine):
        cfg, session = engine
        set_memory_enabled(session, "u1", True)
        pipeline = ExtractionPipeline(config=cfg)

        for i in range(40):
            pipeline.observe(
                session, "u1", "s1", f"I started a new job number {i} today", i
            )
            session.commit()

        chain = _profession_rows(session)
        assert chain, "nothing was stored"
        # Exactly one current value; the rest are closed history.
        assert len([b for b in chain if b.status == "active"]) == 1
        # Every row has its own key -- no duplicates anywhere.
        keys = [b.key for b in session.query(Belief).all()]
        assert len(keys) == len(set(keys)), "duplicate keys were written"

    def test_many_users_survive_the_same_storm(self, engine):
        """The 19-lost-users scenario, at a smaller scale."""
        cfg, session = engine
        pipeline = ExtractionPipeline(config=cfg)

        for user in range(25):
            set_memory_enabled(session, f"u{user}", True)
            for turn in range(6):
                pipeline.observe(
                    session, f"u{user}", "s1",
                    f"user {user} changed jobs again on day {turn}", turn,
                )
                session.commit()

        # Every user stored something and none raised.
        for user in range(25):
            assert _profession_rows_for(session, f"u{user}"), f"user {user} stored nothing"

    def test_belief_history_stays_addressable(self, engine):
        """Superseded rows keep distinct keys, so history is queryable."""
        cfg, session = engine
        set_memory_enabled(session, "u1", True)
        pipeline = ExtractionPipeline(config=cfg)

        for i in range(10):
            pipeline.observe(session, "u1", "s1", f"new role iteration {i}", i)
            session.commit()

        chain = _profession_rows(session)
        assert len({r.key for r in chain}) == len(chain)
        assert any(r.status == "superseded" for r in chain)
        assert sum(1 for r in chain if r.status == "active") == 1


def _profession_rows_for(session, user_id):
    return [b for b in session.query(Belief).filter_by(user_id=user_id)
            if b.predicate == "profession"]


@pytest.mark.skipif(not ORACLE.exists(), reason="LongMemEval oracle not available")
class TestRealDatasetIngestion:
    def test_every_user_ingests_without_error(self):
        """Ingest every LongMemEval haystack; no user may be dropped."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        data = json.loads(ORACLE.read_text(encoding="utf-8"))

        cfg = load_config("config/")
        cfg.llm_client = CollidingLLM()
        cfg.extraction.llm_min_keyword_hits = 0

        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        pipeline = ExtractionPipeline(config=cfg)

        failures = []
        ingested = 0
        for idx, entry in enumerate(data):
            user_id = f"lme_{idx}"
            try:
                set_memory_enabled(session, user_id, True)
                for s_idx, sess in enumerate(entry.get("haystack_sessions") or []):
                    for t_idx, turn in enumerate(sess):
                        content = (turn.get("content") or "").strip()
                        if not content:
                            continue
                        pipeline.observe(
                            session, user_id, f"s{s_idx}", content, t_idx
                        )
                        session.commit()
                        ingested += 1
            except Exception as exc:  # noqa: BLE001 - the test is the counter
                failures.append((user_id, type(exc).__name__, str(exc)[:120]))
                session.rollback()

        assert not failures, f"{len(failures)} users failed: {failures[:3]}"
        assert ingested > 0
