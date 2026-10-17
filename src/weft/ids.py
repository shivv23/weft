"""Time-sortable identifiers (ULID-style) and safe graph-name derivation."""

from __future__ import annotations

import os
import re
import time

# Crockford base32 alphabet (no I, L, O, U).
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_TS_LEN = 10
_RAND_LEN = 16


def ulid() -> str:
    """Return a 26-char ULID: 48-bit ms timestamp + 80 bits randomness."""
    ts = int(time.time() * 1000)
    chars = [_ALPHABET[0]] * (_TS_LEN + _RAND_LEN)
    for i in range(_TS_LEN - 1, -1, -1):
        chars[i] = _ALPHABET[ts & 0x1F]
        ts >>= 5
    rand = int.from_bytes(os.urandom(10), "big")
    for i in range(_RAND_LEN - 1, -1, -1):
        chars[_TS_LEN + i] = _ALPHABET[rand & 0x1F]
        rand >>= 5
    return "".join(chars)


def ulid_time_ms(value: str) -> int:
    """Extract the millisecond timestamp encoded in a ULID."""
    ts = 0
    for ch in value[:_TS_LEN]:
        ts = (ts << 5) | _ALPHABET.index(ch.upper())
    return ts


_SAFE_TENANT = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


def tenant_graph_name(tenant: str) -> str:
    """Derive the FalkorDB graph name for a tenant.

    FalkorDB multigraph topology gives every tenant a fully isolated graph in a
    single instance: no shared indexes, no accidental cross-tenant matches, and
    a graph can be moved/backed up/sharded independently.
    """
    t = (tenant or "").strip().lower()
    if _SAFE_TENANT.match(t):
        return f"weft_{t}"
    import hashlib

    digest = hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:24]
    return f"weft_x{digest}"


__all__ = ["tenant_graph_name", "ulid", "ulid_time_ms"]
