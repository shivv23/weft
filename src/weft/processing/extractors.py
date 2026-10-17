"""Memory extraction: turn conversation transcripts into graph memory.

Two implementations behind one interface:

* :class:`LLMExtractor` — prompts a chat model for structured JSON
  (episodes, entities, relations, facts) and validates it. Used in
  production (``WEFT_LLM_PROVIDER=openai|anthropic|ollama``).
* :class:`HeuristicExtractor` — deterministic offline fallback using lexical
  patterns, so the platform demos and tests run with zero API keys.

Both return the same :class:`Extraction` shape consumed by the consolidator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from weft.providers.llm import LLMProvider, parse_json_response

SYSTEM_PROMPT = """\
You are a memory consolidation engine for an AI agent platform.
Given a conversation transcript, extract durable memory as STRICT JSON with this shape:

{
  "episodes": [
    {
      "title": "short title",
      "summary": "3-5 sentence summary of what happened",
      "kind": "interaction|milestone|error|decision",
      "importance": 0.0-1.0,
      "entities": ["names mentioned"],
      "action_signature": "step1>step2>step3 or null",
      "outcome": "success|failure|null"
    }
  ],
  "entities": [
    {"name": "...", "type": "Person|Organization|System|Artifact|Concept",
     "description": ""}
  ],
  "relations": [
    {"subject": "...", "predicate": "snake_case", "object": "...",
     "confidence": 0.0-1.0}
  ],
  "facts": [
    {"subject": "...", "predicate": "snake_case", "object": "...",
     "confidence": 0.0-1.0}
  ]
}

Rules:
- Extract only what is stated or clearly implied. Never invent names or ids.
- Facts are stable attributes/claims (plan tier, ownership, preferences, ids).
- Relations connect two entities (works_on, owns, member_of, uses).
- One episode per coherent exchange; importance reflects consequence, not length.
- action_signature: if the assistant performed a sequence of tool/action steps,
  join their names with '>'; include outcome if known.
- If nothing is memorable, return empty arrays.
- Respond with JSON only.
"""


@dataclass
class Extraction:
    episodes: list[dict[str, Any]] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class Extractor(Protocol):
    name: str

    async def extract(
        self, transcript: list[dict[str, Any]], *, user_id: str | None = None
    ) -> Extraction: ...


class LLMExtractor:
    name = "llm"

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def extract(
        self, transcript: list[dict[str, Any]], *, user_id: str | None = None
    ) -> Extraction:
        lines = []
        for m in transcript:
            role = m.get("role", "user")
            content = str(m.get("content", ""))[:4000]
            lines.append(f"{role}: {content}")
        user_prompt = f"User id: {user_id or 'unknown'}\n\nTranscript:\n" + "\n".join(lines)
        raw = await self.llm.complete(SYSTEM_PROMPT, user_prompt)
        data = parse_json_response(raw)
        return _validated(data)


class HeuristicExtractor:
    """Pattern-based extraction used when no LLM is configured.

    It is intentionally conservative: it recognises entity shapes (names,
    organisations, codes, emails, urls), a handful of statement patterns and
    action signatures supplied by agents in message metadata.
    """

    name = "heuristic"

    _CAPTION = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
    _CODE = re.compile(r"\b([A-Z]{2,}-\d{2,})\b")
    _EMAIL = re.compile(r"\b([\w.+-]+@[\w-]+\.[\w.]+)\b")
    _URL = re.compile(r"\b((?:https?://|www\.)[^\s]+)\b")
    _STOP: ClassVar[set[str]] = {
        "The",
        "And",
        "But",
        "When",
        "Then",
        "What",
        "How",
        "Why",
        "Yes",
        "No",
        "Hi",
        "Hello",
        "Thanks",
        "Please",
        "Sure",
        "Okay",
        "Also",
        "Can",
        "You",
        "We",
        "I",
        "Our",
        "My",
        "Your",
        "This",
        "That",
        "These",
        "Those",
        "It",
    }
    _FACT_PATTERNS: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (
            re.compile(
                r"\b(?:my|the)\s+([\w\s]{2,30}?)\s+is\s+(?:a\s+|an\s+)?([\w\s]{1,40})\b",
                re.I,
            ),
            "is",
        ),
        (re.compile(r"\b([\w\s]{2,30}?)\s+uses\s+([\w\s]{2,40})\b", re.I), "uses"),
        (re.compile(r"\b([\w\s]{2,30}?)\s+works\s+on\s+([\w\s]{2,40})\b", re.I), "works_on"),
        (re.compile(r"\b([\w\s]{2,30}?)\s+owns\s+([\w\s]{2,40})\b", re.I), "owns"),
        (re.compile(r"\b([\w\s]{2,30}?)\s+prefers\s+([\w\s]{2,40})\b", re.I), "prefers"),
        (re.compile(r"\b([\w\s]{2,30}?)\s+is\s+a\s+([\w\s]{2,40})\b", re.I), "is_a"),
        (re.compile(r"\b([\w\s]{2,30}?)\s+is\s+an\s+([\w\s]{2,40})\b", re.I), "is_a"),
    ]
    _HIGH_IMPACT = re.compile(
        r"\b(refund|charged|invoice|outage|cancel|urgent|broken|failed|security|breach|"
        r"payment|deadline|escalat|lawsuit|compliance)\b",
        re.I,
    )
    _RESOLVED = re.compile(r"\b(resolved|fixed|completed|issued|processed|restored)\b", re.I)

    async def extract(
        self, transcript: list[dict[str, Any]], *, user_id: str | None = None
    ) -> Extraction:
        out = Extraction()
        text_all = "\n".join(str(m.get("content", "")) for m in transcript)
        entities: dict[str, dict[str, Any]] = {}

        for m in transcript:
            content = str(m.get("content", ""))
            for match in self._EMAIL.finditer(content):
                entities[match.group(1)] = {
                    "name": match.group(1),
                    "type": "Email",
                    "description": "",
                }
            for match in self._URL.finditer(content):
                entities[match.group(1)] = {
                    "name": match.group(1),
                    "type": "URL",
                    "description": "",
                }
            for match in self._CODE.finditer(content):
                entities[match.group(1)] = {
                    "name": match.group(1),
                    "type": "Artifact",
                    "description": "identifier code",
                }
            for match in self._CAPTION.finditer(content):
                name = match.group(1)
                if name.split()[0] in self._STOP or len(name) < 3:
                    continue
                if name.endswith((".", ",", ":", ";")):
                    continue
                typ = (
                    "Organization"
                    if any(w in name for w in ("Inc", "Ltd", "Corp", "LLC", "Labs", "Corp"))
                    else "Person"
                )
                entities.setdefault(name, {"name": name, "type": typ, "description": ""})

        meta_entities = [e for m in transcript for e in (m.get("meta") or {}).get("entities", [])]
        for e in meta_entities:
            if isinstance(e, dict) and e.get("name"):
                entities[str(e["name"])] = {
                    "name": str(e["name"]),
                    "type": str(e.get("type", "Concept")),
                    "description": str(e.get("description", "")),
                }

        out.entities = list(entities.values())
        name_set = {e["name"].lower(): e["name"] for e in out.entities}

        def canon(phrase: str) -> str | None:
            p = phrase.strip().strip(".!,;:")
            if len(p) < 2 or len(p) > 60:
                return None
            low = p.lower()
            if low in name_set:
                return name_set[low]
            for ent_name in out.entities:
                if ent_name.lower() in low or low in ent_name.lower():
                    return ent_name["name"]
            return p if low not in {"the", "a", "an", "it", "this", "that", "is"} else None

        for pattern, predicate in self._FACT_PATTERNS:
            for match in pattern.finditer(text_all):
                subj = canon(match.group(1))
                obj = canon(match.group(2))
                if not subj or not obj or subj.lower() == obj.lower():
                    continue
                if predicate == "is" and subj.lower() in {"my", "the", "this"}:
                    continue
                out.facts.append(
                    {
                        "subject": subj,
                        "predicate": predicate,
                        "object": obj,
                        "confidence": 0.6,
                    }
                )

        exchanges = _group_exchanges(transcript)
        for user_text, assistant_text, metas in exchanges:
            summary_src = (assistant_text or user_text)[:1200]
            title_src = (user_text or assistant_text)[:160]
            title = " ".join(title_src.split()[:10])
            importance = 0.4
            if self._HIGH_IMPACT.search(user_text + " " + assistant_text):
                importance = 0.75
            if self._RESOLVED.search(assistant_text):
                importance = min(1.0, importance + 0.1)
            action_sig = None
            outcome = None
            for meta in metas:
                acts = meta.get("actions")
                if isinstance(acts, list) and acts:
                    action_sig = ">".join(str(a) for a in acts)
                if meta.get("outcome"):
                    outcome = str(meta["outcome"])
            episode_entities = [
                e["name"]
                for e in out.entities
                if e["name"].lower() in (user_text + " " + assistant_text).lower()
            ]
            out.episodes.append(
                {
                    "title": title or "conversation",
                    "summary": summary_src,
                    "kind": "interaction",
                    "importance": importance,
                    "entities": episode_entities[:12],
                    "action_signature": action_sig,
                    "outcome": outcome,
                }
            )
        return out


def _group_exchanges(
    transcript: list[dict[str, Any]],
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """Group a transcript into (user_text, assistant_text, metas) exchanges."""
    exchanges: list[tuple[str, list[str], list[dict[str, Any]]]] = []
    for m in transcript:
        role = m.get("role", "user")
        content = str(m.get("content", ""))
        meta = m.get("meta") or {}
        if role == "user" or not exchanges:
            exchanges.append((content if role == "user" else "", [], [meta]))
        elif role == "assistant":
            exchanges[-1][1].append(content)
            exchanges[-1][2].append(meta)
        else:
            exchanges[-1][2].append(meta)
    return [(u, "\n".join(a for a in assistant if a), metas) for u, assistant, metas in exchanges]


def _validated(data: dict[str, Any]) -> Extraction:
    def clamp(v: Any, default: float = 0.5) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return default

    episodes = []
    for e in data.get("episodes") or []:
        if not isinstance(e, dict):
            continue
        episodes.append(
            {
                "title": str(e.get("title") or "episode")[:300],
                "summary": str(e.get("summary") or "")[:2000],
                "kind": str(e.get("kind") or "interaction"),
                "importance": clamp(e.get("importance"), 0.5),
                "entities": [str(x) for x in (e.get("entities") or [])][:20],
                "action_signature": (
                    str(e["action_signature"]) if e.get("action_signature") else None
                ),
                "outcome": str(e["outcome"]) if e.get("outcome") else None,
            }
        )
    entities = [
        {
            "name": str(e.get("name"))[:200],
            "type": str(e.get("type") or "Concept")[:60],
            "description": str(e.get("description") or "")[:500],
        }
        for e in (data.get("entities") or [])
        if isinstance(e, dict) and e.get("name")
    ]
    relations = [
        {
            "subject": str(r.get("subject"))[:200],
            "predicate": str(r.get("predicate") or "related_to")[:60],
            "object": str(r.get("object"))[:200],
            "confidence": clamp(r.get("confidence"), 0.6),
        }
        for r in (data.get("relations") or [])
        if isinstance(r, dict) and r.get("subject") and r.get("object")
    ]
    facts = [
        {
            "subject": str(f.get("subject"))[:200],
            "predicate": str(f.get("predicate") or "is")[:60],
            "object": str(f.get("object"))[:200],
            "confidence": clamp(f.get("confidence"), 0.6),
        }
        for f in (data.get("facts") or [])
        if isinstance(f, dict) and f.get("subject") and f.get("object")
    ]
    return Extraction(episodes=episodes, entities=entities, relations=relations, facts=facts)


def build_extractor(llm: LLMProvider | None) -> Extractor:
    if llm is not None:
        return LLMExtractor(llm)
    return HeuristicExtractor()


__all__ = [
    "Extraction",
    "Extractor",
    "HeuristicExtractor",
    "LLMExtractor",
    "build_extractor",
]
