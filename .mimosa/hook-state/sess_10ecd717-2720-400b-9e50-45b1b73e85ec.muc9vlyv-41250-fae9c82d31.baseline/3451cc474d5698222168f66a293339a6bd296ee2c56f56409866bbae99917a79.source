"""Tests for api.py (MemoryEngine) and server.py — integration tests."""

import os

import pytest

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core.models import Base
from mirror_memory.exceptions import ConfigError, ValidationError
from mirror_memory.models import BeliefInfo, DeleteResult, PanelData


class TestMemoryEngineValidation:
    def test_empty_user_id_raises(self):
        engine = MemoryEngine(config_path="config/", database_url="sqlite://")
        with pytest.raises(ValidationError, match="user_id"):
            engine.observe(user_id="", session_id="s1", text="hello")

    def test_empty_session_id_raises(self):
        engine = MemoryEngine(config_path="config/", database_url="sqlite://")
        with pytest.raises(ValidationError, match="session_id"):
            engine.observe(user_id="u1", session_id="", text="hello")

    def test_empty_text_raises(self):
        engine = MemoryEngine(config_path="config/", database_url="sqlite://")
        with pytest.raises(ValidationError, match="text"):
            engine.observe(user_id="u1", session_id="s1", text="")

    def test_text_too_long_raises(self):
        engine = MemoryEngine(config_path="config/", database_url="sqlite://")
        with pytest.raises(ValidationError, match="length"):
            engine.observe(user_id="u1", session_id="s1", text="x" * 50001)

    def test_limit_out_of_range_raises(self):
        engine = MemoryEngine(config_path="config/", database_url="sqlite://")
        with pytest.raises(ValidationError, match="limit"):
            engine.get_beliefs(user_id="u1", limit=0)

    def test_invalid_language_raises(self):
        engine = MemoryEngine(config_path="config/", database_url="sqlite://")
        with pytest.raises(ValidationError, match="language"):
            engine.panel(user_id="u1", language="fr")

    def test_missing_config_raises(self):
        with pytest.raises((ConfigError, ValueError)):
            MemoryEngine(config_path="/nonexistent")


class TestMemoryEngineObserve:
    @pytest.fixture
    def engine(self):
        return MemoryEngine(config_path="config/", database_url="sqlite://")

    def test_observe_returns_count(self, engine):
        count = engine.observe(user_id="u1", session_id="s1", text="I love painting")
        assert isinstance(count, int)
        assert count >= 0

    def test_observe_multiple_turns(self, engine):
        engine.observe(user_id="u1", session_id="s1", text="hello", turn_count=1)
        engine.observe(user_id="u1", session_id="s1", text="I love painting", turn_count=2)
        beliefs = engine.get_beliefs(user_id="u1")
        assert isinstance(beliefs, list)


class TestMemoryEngineRecall:
    @pytest.fixture
    def engine(self):
        return MemoryEngine(config_path="config/", database_url="sqlite://")

    def test_recall_returns_string_or_none(self, engine):
        result = engine.recall(user_id="u1", query="test")
        assert result is None or isinstance(result, str)

    def test_recall_after_observe(self, engine):
        engine.observe(user_id="u1", session_id="s1", text="I love painting landscapes")
        result = engine.recall(user_id="u1", query="What does the user like?")
        assert result is None or isinstance(result, str)


class TestMemoryEngineBeliefs:
    @pytest.fixture
    def engine(self):
        return MemoryEngine(config_path="config/", database_url="sqlite://")

    def test_get_beliefs_returns_list(self, engine):
        engine.observe(user_id="u1", session_id="s1", text="hello")
        beliefs = engine.get_beliefs(user_id="u1")
        assert isinstance(beliefs, list)
        for b in beliefs:
            assert isinstance(b, BeliefInfo)

    def test_get_beliefs_with_limit(self, engine):
        engine.observe(user_id="u1", session_id="s1", text="hello")
        beliefs = engine.get_beliefs(user_id="u1", limit=1)
        assert len(beliefs) <= 1


class TestMemoryEnginePanel:
    @pytest.fixture
    def engine(self):
        return MemoryEngine(config_path="config/", database_url="sqlite://")

    def test_panel_returns_panel_data(self, engine):
        result = engine.panel(user_id="u1")
        assert isinstance(result, PanelData)

    def test_panel_memory_enabled(self, engine):
        result = engine.panel(user_id="u1")
        assert result.memory["enabled"] is True


class TestMemoryEngineDelete:
    @pytest.fixture
    def engine(self):
        return MemoryEngine(config_path="config/", database_url="sqlite://")

    def test_delete_returns_result(self, engine):
        engine.observe(user_id="u1", session_id="s1", text="hello")
        result = engine.delete_memories(user_id="u1")
        assert isinstance(result, DeleteResult)

    def test_delete_empty_user(self, engine):
        result = engine.delete_memories(user_id="nonexistent")
        assert isinstance(result, DeleteResult)
        assert result.beliefs == 0


class TestMemoryEngineSetEnabled:
    @pytest.fixture
    def engine(self):
        return MemoryEngine(config_path="config/", database_url="sqlite://")

    def test_disable_and_reenable(self, engine):
        engine.set_enabled(user_id="u1", enabled=False)
        engine.set_enabled(user_id="u1", enabled=True)


class TestDeleteResult:
    def test_as_dict(self):
        r = DeleteResult(beliefs=5, belief_events=10)
        d = r.as_dict()
        assert d["beliefs"] == 5
        assert d["belief_events"] == 10


class TestServerEndpoints:
    @pytest.fixture
    def client(self, tmp_path):
        from mirror_memory.server import create_app
        from fastapi.testclient import TestClient
        db_path = tmp_path / "test_server.db"
        app = create_app(config_path="config/", database_url=f"sqlite:///{db_path}")
        with TestClient(app) as c:
            yield c

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_observe_valid(self, client):
        r = client.post("/observe", json={
            "user_id": "u1", "session_id": "s1", "text": "hello"
        })
        assert r.status_code == 200
        assert "claims_extracted" in r.json()

    def test_observe_empty_text_422(self, client):
        r = client.post("/observe", json={
            "user_id": "u1", "session_id": "s1", "text": ""
        })
        assert r.status_code == 422

    def test_recall_valid(self, client):
        client.post("/observe", json={"user_id": "u1", "session_id": "s1", "text": "hello"})
        r = client.post("/recall", json={"user_id": "u1", "query": "test"})
        assert r.status_code == 200

    def test_panel_valid(self, client):
        r = client.get("/panel?user_id=u1")
        assert r.status_code == 200

    def test_beliefs_valid(self, client):
        r = client.get("/beliefs?user_id=u1")
        assert r.status_code == 200

    def test_delete_valid(self, client):
        r = client.post("/delete", json={"user_id": "u1"})
        assert r.status_code == 200

    def test_auth_required(self):
        """With MM_API_KEY set, requests without auth should fail."""
        os.environ["MM_API_KEY"] = "test-secret"
        try:
            from mirror_memory.server import create_app
            from fastapi.testclient import TestClient
            app = create_app(config_path="config/", database_url="sqlite://")
            with TestClient(app) as client:
                r = client.post("/observe", json={
                    "user_id": "u1", "session_id": "s1", "text": "hello"
                })
                assert r.status_code == 401
        finally:
            del os.environ["MM_API_KEY"]

    def test_auth_with_valid_key(self):
        """With correct API key, requests should succeed."""
        os.environ["MM_API_KEY"] = "test-secret"
        try:
            from mirror_memory.server import create_app
            from fastapi.testclient import TestClient
            app = create_app(config_path="config/", database_url="sqlite://")
            with TestClient(app) as client:
                r = client.post("/observe", json={
                    "user_id": "u1", "session_id": "s1", "text": "hello"
                }, headers={"Authorization": "Bearer test-secret"})
                assert r.status_code == 200
        finally:
            del os.environ["MM_API_KEY"]