"""Time helpers: ISO-8601 UTC timestamps (FalkorDB has no native temporal type,
so we store RFC-3339 strings, which sort lexicographically) and decay maths."""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def now_ms() -> int:
    return int(time.time() * 1000)


def iso_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat(timespec="milliseconds")


def parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def age_days(iso_ts: str, *, now: datetime | None = None) -> float:
    ref = now or datetime.now(UTC)
    return max(0.0, (ref - parse_iso(iso_ts)).total_seconds() / 86400.0)


def exponential_decay(age_in_days: float, half_life_days: float) -> float:
    """Multiplier in (0, 1]: 1.0 at age 0, 0.5 after one half-life."""
    if half_life_days <= 0:
        return 1.0
    return math.pow(0.5, age_in_days / half_life_days)


__all__ = [
    "age_days",
    "exponential_decay",
    "iso_from_ms",
    "now_iso",
    "now_ms",
    "parse_iso",
]
