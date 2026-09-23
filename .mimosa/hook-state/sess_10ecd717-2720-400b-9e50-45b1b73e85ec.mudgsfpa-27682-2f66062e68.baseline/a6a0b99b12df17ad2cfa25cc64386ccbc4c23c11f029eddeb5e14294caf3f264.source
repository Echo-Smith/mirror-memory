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
            if usage:
                logger.debug(
                    "LLM call: model=%s tokens=%d/%d latency=%dms",
                    self._model,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    elapsed_ms,
                )

            return content

        except Exception:
            logger.warning("LLM call failed (model=%s)", self._model, exc_info=True)
            if fallback is not None:
                return fallback()
            raise
