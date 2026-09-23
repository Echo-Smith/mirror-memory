"""Mirror Memory exception hierarchy.

All custom exceptions inherit from ``MirrorMemoryError`` so callers can
catch the base class for any library failure, or catch specific subclasses
for precise error handling.

Hierarchy::

    MirrorMemoryError
    ├── ConfigError          — invalid or missing configuration
    ├── ExtractionError      — K1/K2/K3 extraction failure
    ├── LLMError             — LLM client or call failure
    ├── StorageError         — database or persistence failure
    └── ValidationError      — input validation failure
"""

from __future__ import annotations


class MirrorMemoryError(Exception):
    """Base exception for all mirror-memory errors."""

    pass


class ConfigError(MirrorMemoryError):
    """Invalid, missing, or inconsistent configuration.

    Raised when:
    - Config directory does not exist
    - Required YAML files are missing
    - YAML parsing fails
    - Prompt templates are empty
    - Cross-field validation fails (e.g. floor > cap)
    """

    pass


class ExtractionError(MirrorMemoryError):
    """K1/K2/K3 extraction pipeline failure.

    Raised when extraction fails in a way the caller should handle.
    Note: the extraction pipeline is fail-open by default, so most
    extraction errors are logged and swallowed.  This exception is
    only raised when ``fail_open=False`` or for critical errors.
    """

    pass


class LLMError(MirrorMemoryError):
    """LLM client or API call failure.

    Wraps underlying provider errors (OpenAI, DeepSeek, etc.)
    so callers don't need to import provider-specific exceptions.
    """

    def __init__(self, message: str, *, provider_error: Exception | None = None) -> None:
        super().__init__(message)
        self.provider_error = provider_error


class StorageError(MirrorMemoryError):
    """Database or persistence failure.

    Raised when a database operation fails in a way the caller should handle.
    """

    pass


class ValidationError(MirrorMemoryError):
    """Input validation failure.

    Raised when public API arguments fail validation (empty user_id,
    text too long, invalid language, etc.).
    """

    pass
