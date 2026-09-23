"""Behavioural signal detection -- generic templates, zero domain coupling.

Signals are *observations*, not inferences.  An anomaly is a conversation
opener, not a data point.  The user's own interpretation is the real data.

Design principles:
- Signals are observations, not diagnostics.
- Anomalies are conversation entries, not data points.
- The user's explanation is the actual data.

Question templates are loaded from configuration (``question_templates``
key).  The defaults here are English-only placeholders; override them with
your domain-specific wording.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import Any

from mirror_memory.core.constants import (
    DEFAULT_SLEEP_WINDOW,
    MESSAGE_LENGTH_DROP_RATIO,
    RESPONSE_DELAY_MAX_SECONDS,
    RESPONSE_DELAY_MIN_SECONDS,
    VAD_BRIEF_SPEECH_MS,
    VAD_HESITATION_MIN_SPEECH_MS,
    VAD_HESITATION_PAUSE_COUNT,
)

# Default question templates (override via config).
_DEFAULT_TEMPLATES: dict[str, dict[str, str]] = {
    "speech_hesitation": {
        "en": "You paused several times while speaking -- is something hard to put into words?",
        "zh": "你刚才说的时候好像停了好几次，是有些地方不太好表达吗？",
    },
    "brief_speech": {
        "en": "You spoke very quickly -- are you not in the mood to talk about this?",
        "zh": "你刚才说得很快，是不太想聊这个吗？",
    },
    "long_pause": {
        "en": "You seem to have paused for a moment -- did something happen?",
        "zh": "你刚才好像停了一会儿，是发生了什么吗？",
    },
    "short_message": {
        "en": "Your message is shorter than usual -- are you tired or not in the mood to talk about this?",
        "zh": "你今天的回复比之前简短一些，是累了还是不太想聊这个？",
    },
    "late_night": {
        "en": "It's quite late -- are you having trouble sleeping, or is something on your mind?",
        "zh": "这么晚了还没休息，是睡不着还是有什么事在心里？",
    },
}


def _question(signal_type: str, language: str, templates: dict[str, dict[str, str]] | None = None) -> str:
    """Look up a question template for *signal_type* in *language*.

    Falls back to English if the requested language is missing.
    """
    tpl = (templates or _DEFAULT_TEMPLATES).get(signal_type, {})
    lang = language.strip().lower()
    return tpl.get(lang) or tpl.get("en") or ""


def detect_behavioral_signals(
    *,
    user_message: str,
    bot_message_timestamp: datetime | None = None,
    user_message_timestamp: datetime | None = None,
    recent_message_lengths: list[int] | None = None,
    session_hour: int = -1,
    vad_metadata: dict | None = None,
    sleep_window: tuple[int, int] | None = None,
    language: str = "en",
    question_templates: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Detect behavioural anomaly signals for the current turn.

    Returns a list of signal dicts, each with keys:
    ``type``, ``detail``, ``question``.

    Parameters
    ----------
    user_message:
        The current user message text.
    bot_message_timestamp:
        Timestamp of the preceding bot message.
    user_message_timestamp:
        Server-side timestamp of the current user message.
    recent_message_lengths:
        Character lengths of the last N user messages (for drop detection).
    session_hour:
        Hour of day (0-23) for the current session; -1 to skip.
    vad_metadata:
        Client-side voice metadata: ``{startAt, endAt, speechDurationMs,
        pauseCount}``.
    sleep_window:
        ``(start_hour, end_hour)``; defaults to ``(23, 5)``.
    language:
        ``"en"`` or ``"zh"`` (or any language key present in templates).
    question_templates:
        Override the default question templates.  Structure:
        ``{"signal_type": {"en": "...", "zh": "..."}, ...}``.
    """
    signals: list[dict[str, Any]] = []
    templates = question_templates or _DEFAULT_TEMPLATES

    # 1. VAD signals
    if vad_metadata:
        pause_count = vad_metadata.get("pauseCount", 0)
        speech_ms = vad_metadata.get("speechDurationMs", 0)
        if pause_count >= VAD_HESITATION_PAUSE_COUNT and speech_ms > VAD_HESITATION_MIN_SPEECH_MS:
            signals.append({
                "type": "speech_hesitation",
                "detail": f"pauses={pause_count}, duration={speech_ms}ms",
                "question": _question("speech_hesitation", language, templates),
            })
        elif speech_ms > 0 and speech_ms < VAD_BRIEF_SPEECH_MS and len(user_message or "") > 5:
            signals.append({
                "type": "brief_speech",
                "detail": f"duration={speech_ms}ms, len={len(user_message)}",
                "question": _question("brief_speech", language, templates),
            })

    # 2. Response delay anomaly
    effective_user_ts = user_message_timestamp
    if vad_metadata and vad_metadata.get("startAt"):
        with contextlib.suppress(TypeError, ValueError, OSError):
            effective_user_ts = datetime.fromtimestamp(vad_metadata["startAt"] / 1000, tz=UTC)
    if bot_message_timestamp and effective_user_ts:
        if bot_message_timestamp.tzinfo is None:
            bot_message_timestamp = bot_message_timestamp.replace(tzinfo=UTC)
        if effective_user_ts.tzinfo is None:
            effective_user_ts = effective_user_ts.replace(tzinfo=UTC)
        latency = (effective_user_ts - bot_message_timestamp).total_seconds()
        if RESPONSE_DELAY_MIN_SECONDS < latency < RESPONSE_DELAY_MAX_SECONDS:
            signals.append({
                "type": "long_pause",
                "detail": f"{int(latency // 60)}min",
                "question": _question("long_pause", language, templates),
            })

    # 3. Message length drop
    if recent_message_lengths and len(recent_message_lengths) >= 3:
        avg_len = sum(recent_message_lengths) / len(recent_message_lengths)
        current_len = len(user_message or "")
        if avg_len > 20 and current_len < avg_len * MESSAGE_LENGTH_DROP_RATIO and current_len > 0:
            signals.append({
                "type": "short_message",
                "detail": f"{current_len} vs avg {int(avg_len)}",
                "question": _question("short_message", language, templates),
            })

    # 4. Late-night session
    if session_hour >= 0:
        win = sleep_window or DEFAULT_SLEEP_WINDOW
        sleep_start, sleep_end = win
        if sleep_start > sleep_end:
            in_sleep = session_hour >= sleep_start or session_hour < sleep_end
        else:
            in_sleep = sleep_start <= session_hour < sleep_end
        if in_sleep:
            signals.append({
                "type": "late_night",
                "detail": f"hour={session_hour}",
                "question": _question("late_night", language, templates),
            })

    # At most one signal per turn (don't bombard the user).
    return signals[:1] if signals else []
