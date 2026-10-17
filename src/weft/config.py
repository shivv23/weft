"""Environment-driven configuration. No secret is ever hardcoded in the codebase."""

from __future__ import annotations

import json
import threading

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="WEFT_",
        extra="ignore",
        frozen=True,
    )

    # --- FalkorDB ---
    falkor_host: str = "localhost"
    falkor_port: int = 6379
    falkor_password: str | None = None
    falkor_url: str | None = None
    falkor_max_connections: int = 64

    # --- Embeddings ---
    embeddings_provider: str = "hash"  # hash | openai | ollama
    embeddings_model: str = "text-embedding-3-small"
    embeddings_dim: int = 256
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_url: str = "http://localhost:11434"

    # --- LLM ---
    llm_provider: str = "none"  # none | openai | anthropic | ollama
    llm_model: str = "gpt-4o-mini"
    openai_chat_api_key: str | None = None  # falls back to openai_api_key
    anthropic_api_key: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048

    # --- Consolidation worker ---
    consolidation_enabled: bool = True
    consolidation_poll_seconds: float = 1.0
    consolidation_batch: int = 8

    # --- Worker pool (multi-tenant background processing) ---
    workers_in_api: bool = False
    worker_tenant_refresh_seconds: float = 30.0
    maintenance_interval_seconds: float = 300.0

    # --- Memory lifecycle ---
    episode_half_life_days: float = 14.0
    episode_archive_after_days: float = 90.0
    episode_archive_salience: float = 0.05
    fact_promote_evidence: int = 3
    procedure_min_support: int = 2

    # --- Retrieval weights (must sum to ~1; normalised at use) ---
    weight_vector: float = 0.35
    weight_fulltext: float = 0.15
    weight_graph: float = 0.20
    weight_recency: float = 0.15
    weight_importance: float = 0.10
    weight_frequency: float = 0.05

    # --- API ---
    api_keys: str = ""  # "key:tenant[:role]" CSV; empty = auth disabled
    cors_origins: list[str] = Field(default=["*"])
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    root_path: str = ""  # set when served behind a path prefix (e.g. /weft)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_origins(cls, v: object) -> object:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @property
    def chat_api_key(self) -> str | None:
        return self.openai_chat_api_key or self.openai_api_key

    @property
    def falkor_dsn(self) -> str | None:
        return self.falkor_url

    @property
    def api_key_map(self) -> dict[str, tuple[str, str]]:
        """Parse WEFT_API_KEYS into {key: (tenant, role)}."""
        out: dict[str, tuple[str, str]] = {}
        for part in self.api_keys.split(","):
            part = part.strip()
            if not part:
                continue
            bits = part.split(":")
            if len(bits) == 2:
                out[bits[0]] = (bits[1], "member")
            elif len(bits) >= 3:
                out[bits[0]] = (bits[1], bits[2] or "member")
        return out

    @property
    def retrieval_weights(self) -> dict[str, float]:
        raw = {
            "vector": self.weight_vector,
            "fulltext": self.weight_fulltext,
            "graph": self.weight_graph,
            "recency": self.weight_recency,
            "importance": self.weight_importance,
            "frequency": self.weight_frequency,
        }
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}


_lock = threading.Lock()
_cached: Settings | None = None


def get_settings() -> Settings:
    global _cached
    if _cached is None:
        with _lock:
            if _cached is None:
                _cached = Settings()
    return _cached


def reset_settings_cache() -> None:
    """Testing hook: re-read environment on next get_settings()."""
    global _cached
    with _lock:
        _cached = None


__all__ = ["Settings", "get_settings", "reset_settings_cache"]
