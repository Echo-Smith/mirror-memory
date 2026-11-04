"""Built-in LLM adapter for OpenAI-compatible APIs.

Provides a drop-in ``generate()`` interface that the extraction pipeline
expects, wrapping the official OpenAI Python SDK.

Usage::

    from mirror_memory.llm import OpenAILLM

    llm = OpenAILLM(api_key="sk-xxx", model="deepseek-chat",
                     base_url="https://api.deepseek.com")
    engine = MemoryEngine(config_path="config/", llm_client=llm)

Or let ``MemoryEngine`` create it automatically::

    engine = MemoryEngine(
        config_path="config/",
        llm_api_key="sk-xxx",
        llm_base_url="https://api.deepseek.com",
        llm_model="deepseek-chat",
    )
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class OpenAILLM:
    """LLM adapter for any OpenAI-compatible API (OpenAI, DeepSeek, etc.).

    Exposes the ``generate(system_prompt, payload_text, fallback)`` interface
    that ``SemanticExtractor`` and ``EvolutionWorker`` expect.

    Parameters
    ----------
    api_key:
        API key for the LLM provider.
    model:
        Model identifier (e.g. ``"gpt-4o-mini"``, ``"deepseek-chat"``).
    base_url:
        API base URL.  Defaults to OpenAI's official endpoint.
        Use ``"https://api.deepseek.com"`` for DeepSeek, etc.
    temperature:
        Sampling temperature.  Default ``0.0`` for deterministic extraction.
    max_tokens:
        Maximum response tokens.  Default ``800``.
    max_retries:
        Number of retries on transient errors.  Default ``2``.
    timeout:
        Request timeout in seconds.  Default ``60``.
    extra_body:
        Provider-specific request fields passed through verbatim to
        ``chat.completions.create``.  Use this for extensions that are not
        part of the OpenAI schema -- for example
        ``{"thinking": {"type": "disabled"}}`` to switch off a reasoning
        model's thinking mode.  Reasoning models otherwise spend their token
        budget on thinking and can return an empty completion, which silently
        yields zero extracted claims.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
        max_retries: int = 2,
        timeout: float = 60.0,
        extra_body: dict[str, Any] | None = None,
        on_call: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAILLM. "
                "Install it with: pip install openai"
            )

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra_body = extra_body
        # M5 task 2: a per-call hook receiving
        # {"ok": bool, "input_tokens": int|None, "output_tokens": int|None,
        # "latency_ms": int}.  The engine sets this to feed metrics (and the
        # entry label); storing tokens only at DEBUG made per-entry cost
        # invisible, which is exactly what the observability task needs.
        self._on_call: Callable[[dict[str, Any]], None] | None = on_call
        # Call health.  A failed call returns the caller's fallback instead of
        # raising, so the failure is counted here -- where it happens -- or a
        # rate-limited run looks like a model that simply found nothing.
        self.calls = 0
        self.call_failures = 0

    def set_on_call(self, hook: Callable[[dict[str, Any]], None] | None) -> None:
        """Install (or clear) the per-call usage hook (see ``__init__``)."""
        self._on_call = hook

    def _emit_call(
        self,
        *,
        ok: bool,
        input_tokens: int | None,
        output_tokens: int | None,
        latency_ms: int,
    ) -> None:
        hook = self._on_call
        if hook is None:
            return
        try:
            hook({
                "ok": ok,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
            })
        except Exception:  # observability must never break a call
            logger.warning("LLM usage hook raised; ignoring", exc_info=True)

    def generate(
        self,
        *,
        system_prompt: str,
        payload_text: str,
        fallback: Callable[[], str] | None = None,
    ) -> str:
        """Generate a response from the LLM.

        Parameters
        ----------
        system_prompt:
            The system message content.
        payload_text:
            The user message content.
        fallback:
            Optional callable returning a fallback string on error.
            If ``None``, errors are raised.

        Returns
        -------
        str
            The LLM response text.

        Raises
        ------
        openai.APIError
            If the API call fails and no fallback is provided.
        """
        try:
            started = time.monotonic()
            self.calls += 1
            request: dict[str, Any] = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": payload_text},
                ],
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            }
            if self._extra_body:
                request["extra_body"] = self._extra_body
            response = self._client.chat.completions.create(**request)
            elapsed_ms = int((time.monotonic() - started) * 1000)
            content = (response.choices[0].message.content or "").strip()

            # Log usage if available.
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
            if usage:
                logger.debug(
                    "LLM call: model=%s tokens=%s/%s latency=%dms",
                    self._model,
                    prompt_tokens,
                    completion_tokens,
                    elapsed_ms,
                )
            # M5 task 2: surface usage to the caller's hook (metrics).  When
            # the provider returns no usage the tokens stay ``None`` and the
            # hook records the call as usage-unknown -- never a guess.
            self._emit_call(
                ok=True,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                latency_ms=elapsed_ms,
            )

            return content

        except Exception:
            self.call_failures += 1
            logger.warning("LLM call failed (model=%s)", self._model, exc_info=True)
            # A failed call still counts (a call was attempted) but carries
            # no usage: report ok=False so the hook can tell the difference.
            self._emit_call(
                ok=False, input_tokens=None, output_tokens=None, latency_ms=0,
            )
            if fallback is not None:
                return fallback()
            raise
