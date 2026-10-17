"""Pydantic domain models shared by the library, REST API and MCP server."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MemoryKind = Literal["episode", "fact", "entity", "procedure"]
Role = Literal["system", "user", "assistant", "tool"]


class MessageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    role: Role
    content: str
    meta: dict[str, Any] = Field(default_factory=dict)
    ts: str | None = None


class MessageOut(BaseModel):
    id: str
    seq: int
    role: str
    content: str
    ts: str
    meta: dict[str, Any] = Field(default_factory=dict)


class SessionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    agent_id: str | None = None
    title: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class SessionOut(BaseModel):
    id: str
    user_id: str
    agent_id: str | None = None
    title: str | None = None
    status: str = "open"
    started_at: str
    last_activity_at: str
    message_count: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)


class AgentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    kind: str = "worker"
    description: str = ""
    capabilities: list[str] = Field(default_factory=list)


class AgentOut(AgentIn):
    created_at: str
    handoffs_sent: int = 0
    handoffs_received: int = 0


class EntityOut(BaseModel):
    id: str
    name: str
    type: str
    description: str | None = None
    mention_count: int = 0


class FactOut(BaseModel):
    id: str
    subject: str
    predicate: str
    object: str
    text: str
    confidence: float
    evidence_count: int
    first_seen: str
    last_seen: str
    valid_to: str | None = None
    source_episode_ids: list[str] = Field(default_factory=list)


class FactIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    object: str
    subject_type: str = "Concept"
    object_type: str = "Concept"
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    source_episode_id: str | None = None
    source: str = "api"


class ProcedureIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    steps: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class ProcedureOut(BaseModel):
    id: str
    name: str
    description: str
    steps: list[str]
    tags: list[str]
    success_count: int = 0
    failure_count: int = 0
    avg_duration_ms: float | None = None
    version: int = 1
    last_used_at: str | None = None


class ProcedureFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    duration_ms: float | None = None
    episode_id: str | None = None


class EpisodeOut(BaseModel):
    id: str
    kind: str
    title: str
    summary: str
    importance: float
    ts: str
    session_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    salience: float = 1.0
    access_count: int = 0
    entities: list[str] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    vector: float = 0.0
    fulltext: float = 0.0
    graph: float = 0.0
    recency: float = 0.0
    importance: float = 0.0
    frequency: float = 0.0

    def total(self, weights: dict[str, float]) -> float:
        return sum(getattr(self, k) * weights.get(k, 0.0) for k in self.model_dump())


class MemoryItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    kind: MemoryKind
    text: str
    score: float = 0.0
    scores: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    ts: str | None = None
    matched_via: list[str] = Field(default_factory=list)
    explanation: dict[str, Any] = Field(default_factory=dict)


class RecallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    k: int = Field(default=8, ge=1, le=100)
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    kinds: list[MemoryKind] | None = None
    since: str | None = None
    include_explanation: bool = True
    touch: bool = True


class RecallResponse(BaseModel):
    query: str
    items: list[MemoryItem]
    took_ms: float


class BlackboardSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Any
    expected_version: int | None = None


class BlackboardEntry(BaseModel):
    scope: str
    key: str
    value: Any
    version: int
    updated_by: str | None = None
    updated_at: str


class HandoffCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_agent: str
    to_agent: str
    title: str = ""
    reason: str = ""
    context: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5, ge=0, le=9)
    session_id: str | None = None
    user_id: str | None = None
    ref_ids: list[str] = Field(default_factory=list)


class HandoffOut(BaseModel):
    id: str
    from_agent: str
    to_agent: str
    title: str
    reason: str
    context: dict[str, Any]
    priority: int
    status: str
    session_id: str | None = None
    user_id: str | None = None
    created_at: str
    claimed_at: str | None = None
    completed_at: str | None = None
    result: Any = None


class ContextPackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    user_id: str | None = None
    session_id: str | None = None
    query: str = ""
    k: int = Field(default=8, ge=1, le=50)


class ContextPack(BaseModel):
    agent_id: str
    query: str
    memories: list[MemoryItem] = Field(default_factory=list)
    pending_handoffs: list[HandoffOut] = Field(default_factory=list)
    blackboard: list[BlackboardEntry] = Field(default_factory=list)
    session: SessionOut | None = None
    took_ms: float = 0.0


class StatsOut(BaseModel):
    tenant: str
    counts: dict[str, int]
    job_queue: dict[str, int]
    graphs: int = 1


__all__ = [name for name in dir() if not name.startswith("_")]
