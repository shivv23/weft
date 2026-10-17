"""Typed error hierarchy for Weft."""

from __future__ import annotations


class WeftError(Exception):
    """Base class for all Weft errors."""

    code = "weft_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(WeftError):
    code = "configuration_error"


class NotFoundError(WeftError):
    code = "not_found"


class ConflictError(WeftError):
    """Raised on optimistic-concurrency failures (e.g. blackboard version mismatch)."""

    code = "conflict"


class ClaimError(WeftError):
    """Raised when an agent loses a race to claim a handoff."""

    code = "claim_failed"


class ValidationError(WeftError):
    code = "validation_error"


class ProviderError(WeftError):
    """LLM / embedding provider failure."""

    code = "provider_error"


class StorageError(WeftError):
    code = "storage_error"


__all__ = [
    "ClaimError",
    "ConfigurationError",
    "ConflictError",
    "NotFoundError",
    "ProviderError",
    "StorageError",
    "ValidationError",
    "WeftError",
]
