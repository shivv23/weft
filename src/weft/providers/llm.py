"""Pluggable chat-LLM providers used by the consolidation worker.

No API key is embedded in the codebase: every credential comes from the
environment. With ``WEFT_LLM_PROVIDER=none`` Weft uses the deterministic
heuristic extractor so the whole system runs offline.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol, runtime_checkable

import httpx

from weft.config import Settings, get_settings
from weft.errors import ProviderError

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    async def complete(self, system: str, user: str) -> str: ...


class OpenAIChat:
    """OpenAI-compatible chat completions (OpenAI, Groq, Together, vLLM, ...)."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(timeout=timeout)

    async def complete(self, system: str, user: str) -> str:
        try:
            resp = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as exc:
            raise ProviderError(f"chat completion failed: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


class AnthropicChat:
    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_tokens: int = 2048,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(timeout=timeout)

    async def complete(self, system: str, user: str) -> str:
        try:
            resp = await self._client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            )
            resp.raise_for_status()
            blocks = resp.json()["content"]
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        except httpx.HTTPError as exc:
            raise ProviderError(f"anthropic completion failed: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


class OllamaChat:
    name = "ollama"

    def __init__(
        self, *, url: str, model: str, temperature: float = 0.0, timeout: float = 120.0
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self._client = httpx.AsyncClient(timeout=timeout)

    async def complete(self, system: str, user: str) -> str:
        try:
            resp = await self._client.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except httpx.HTTPError as exc:
            raise ProviderError(f"ollama completion failed: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


def build_llm(settings: Settings | None = None) -> LLMProvider | None:
    """Return the configured chat provider, or None for heuristic-only mode."""
    s = settings or get_settings()
    provider = s.llm_provider.lower()
    if provider in ("none", "", "off"):
        return None
    if provider == "openai":
        key = s.chat_api_key
        if not key:
            raise ProviderError("WEFT_OPENAI_API_KEY is required for llm_provider=openai")
        return OpenAIChat(
            api_key=key,
            model=s.llm_model,
            base_url=s.openai_base_url,
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
        )
    if provider == "anthropic":
        if not s.anthropic_api_key:
            raise ProviderError("WEFT_ANTHROPIC_API_KEY is required for llm_provider=anthropic")
        return AnthropicChat(
            api_key=s.anthropic_api_key, model=s.llm_model, max_tokens=s.llm_max_tokens
        )
    if provider == "ollama":
        return OllamaChat(url=s.ollama_url, model=s.llm_model, temperature=s.llm_temperature)
    raise ProviderError(f"unknown llm provider: {provider!r}")


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_response(raw: str) -> dict:
    """Best-effort JSON extraction from an LLM response (fences, prose, etc.)."""
    text = raw.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise ProviderError(f"LLM response was not valid JSON: {raw[:400]}")


__all__ = [
    "AnthropicChat",
    "LLMProvider",
    "OllamaChat",
    "OpenAIChat",
    "build_llm",
    "parse_json_response",
]
