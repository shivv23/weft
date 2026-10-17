"""Pluggable embedding providers.

The default ``hash`` provider is deterministic, offline and dependency-free:
tokens are hashed into a fixed-dim signed bag-of-features vector. It is not a
neural embedding, but it gives stable lexical similarity for demos and tests.
Production deployments set ``WEFT_EMBEDDINGS_PROVIDER=openai`` (any
OpenAI-compatible endpoint) or ``ollama``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from itertools import pairwise
from typing import Protocol, runtime_checkable

import httpx

from weft.config import Settings, get_settings
from weft.errors import ProviderError

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9']+")


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    dim: int

    async def embed_many(self, texts: list[str]) -> list[list[float]]: ...

    async def embed(self, text: str) -> list[float]: ...


class HashEmbedding:
    """Deterministic signed feature hashing. Offline, zero-config."""

    name = "hash"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    async def embed(self, text: str) -> list[float]:
        return self._one(text)

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return vec
        feats: list[str] = tokens + [f"{a}_{b}" for a, b in pairwise(tokens)]
        for tok in feats:
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class OpenAIEmbedding:
    """Any OpenAI-compatible embeddings endpoint (OpenAI, vLLM, LiteLLM, ...)."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dim: int,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            resp = await self._client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            data.sort(key=lambda d: d["index"])
            return [row["embedding"] for row in data]
        except httpx.HTTPError as exc:
            raise ProviderError(f"embedding request failed: {exc}") from exc

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_many([text]))[0]

    async def aclose(self) -> None:
        await self._client.aclose()


class OllamaEmbedding:
    """Local Ollama embeddings (/api/embed)."""

    name = "ollama"

    def __init__(self, *, url: str, model: str, dim: int) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.dim = dim
        self._client = httpx.AsyncClient(timeout=60.0)

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            resp = await self._client.post(
                f"{self.url}/api/embed", json={"model": self.model, "input": texts}
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]
        except httpx.HTTPError as exc:
            raise ProviderError(f"ollama embed failed: {exc}") from exc

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_many([text]))[0]

    async def aclose(self) -> None:
        await self._client.aclose()


def build_embedder(settings: Settings | None = None) -> EmbeddingProvider:
    s = settings or get_settings()
    provider = s.embeddings_provider.lower()
    if provider == "hash":
        return HashEmbedding(dim=s.embeddings_dim)
    if provider == "openai":
        if not s.openai_api_key:
            raise ProviderError("WEFT_OPENAI_API_KEY is required for embeddings_provider=openai")
        return OpenAIEmbedding(
            api_key=s.openai_api_key,
            model=s.embeddings_model,
            dim=s.embeddings_dim,
            base_url=s.openai_base_url,
        )
    if provider == "ollama":
        return OllamaEmbedding(url=s.ollama_url, model=s.embeddings_model, dim=s.embeddings_dim)
    raise ProviderError(f"unknown embeddings provider: {provider!r}")


__all__ = [
    "EmbeddingProvider",
    "HashEmbedding",
    "OllamaEmbedding",
    "OpenAIEmbedding",
    "build_embedder",
]
