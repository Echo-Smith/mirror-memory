"""Tests for core/behavioral.py — detect_behavioral_signals."""

from datetime import UTC, datetime, timedelta

from mirror_memory.core.behavioral import detect_behavioral_signals


class TestBehavioralSignals:
    def test_no_signals_returns_empty(self):
        signals = detect_behavioral_signals(user_message="hello")
        assert signals == []

    def test_brief_speech(self):
        """Short speech with text content triggers brief_speech."""
        signals = detect_behavioral_signals(
            user_message="I'm fine thank you",
            vad_metadata={"speechDurationMs": 1000},
        )
        assert any(s["type"] == "brief_speech" for s in signals)

    def test_no_brief_speech_for_short_text(self):
        """Very short text (<=5 chars) should not trigger brief_speech."""
        signals = detect_behavioral_signals(
            user_message="ok",
            vad_metadata={"speechDurationMs": 1000},
        )
        assert not any(s["type"] == "brief_speech" for s in signals)

    def test_speech_hesitation(self):
        signals = detect_behavioral_signals(
            user_message="I was thinking about it for a while",
            vad_metadata={"speechDurationMs": 6000, "pauseCount": 4},
        )
        assert any(s["type"] == "speech_hesitation" for s in signals)

    def test_no_hesitation_with_few_pauses(self):
        signals = detect_behavioral_signals(
            user_message="I was thinking about it",
            vad_metadata={"speechDurationMs": 6000, "pauseCount": 1},
        )
        assert not any(s["type"] == "speech_hesitation" for s in signals)

    def test_long_pause(self):
        now = datetime.now(UTC)
        signals = detect_behavioral_signals(
            user_message="I was thinking about it",
            bot_message_timestamp=now - timedelta(minutes=10),
            user_message_timestamp=now,
        )
        assert any(s["type"] == "long_pause" for s in signals)

    def test_no_long_pause_for_short_delay(self):
        now = datetime.now(UTC)
        signals = detect_behavioral_signals(
            user_message="hello there",
            bot_message_timestamp=now - timedelta(seconds=30),
            user_message_timestamp=now,
        )
        assert not any(s["type"] == "long_pause" for s in signals)

    def test_short_message_drop(self):
        """Message much shorter than recent average triggers short_message."""
        signals = detect_behavioral_signals(
            user_message="ok fine",
            recent_message_lengths=[100, 120, 110, 90],
        )
        assert any(s["type"] == "short_message" for s in signals)

    def test_no_short_message_for_normal_length(self):
        signals = detect_behavioral_signals(
            user_message="I had a normal day today and everything was fine",
            recent_message_lengths=[30, 25, 35],
        )
        assert not any(s["type"] == "short_message" for s in signals)

    def test_late_night(self):
        signals = detect_behavioral_signals(
            user_message="can't sleep at all tonight",
            session_hour=2,
        )
        assert any(s["type"] == "late_night" for s in signals)

    def test_no_late_night_during_day(self):
        signals = detect_behavioral_signals(
            user_message="good morning everyone",
            session_hour=10,
        )
        assert not any(s["type"] == "late_night" for s in signals)

    def test_late_night_wrap_midnight(self):
        """Sleep window (23, 5) should trigger at hour=23 and hour=4."""
        s23 = detect_behavioral_signals(user_message="hi there", session_hour=23)
        s4 = detect_behavioral_signals(user_message="hi there", session_hour=4)
        assert any(s["type"] == "late_night" for s in s23)
        assert any(s["type"] == "late_night" for s in s4)

    def test_at_most_one_signal(self):
        """Even if multiple conditions match, only one signal returned."""
        signals = detect_behavioral_signals(
            user_message="ok fine",
            session_hour=2,
            vad_metadata={"speechDurationMs": 1000},
            recent_message_lengths=[100, 120, 110],
        )
        assert len(signals) <= 1

    def test_custom_question_templates(self):
        templates = {"brief_speech": {"en": "Custom template"}}
        signals = detect_behavioral_signals(
            user_message="short message here",
            vad_metadata={"speechDurationMs": 1000},
            question_templates=templates,
        )
        if signals:
            assert signals[0].get("question") == "Custom template"

    def test_language_zh(self):
        signals = detect_behavioral_signals(
            user_message="这么晚了还睡不着",
            session_hour=2,
            language="zh",
        )
        if signals:
            # Chinese question template should be used
            assert any(ord(c) > 0x4e00 for c in signals[0].get("question", ""))

    def test_empty_user_message(self):
        signals = detect_behavioral_signals(user_message="")
        assert signals == []
