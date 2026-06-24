"""Conversation stores for the unified Ask-AI agent (COG-130, COG-131).

The agent classifies one message per turn. For multi-turn dialogues — most
importantly a ``clarify`` round followed by the user's answer — the classifier
and the capabilities must see the WHOLE conversation, not just the latest
message in isolation. Without that, an under-specified answer like "I wanna do
both" looks ambiguous on its own and the agent re-asks the same question
forever (the COG-130 infinite-clarify loop).

The frontend already threads a stable ``session_id`` into every request, so the
backend owns conversation state keyed by that id — keeping the clients thin (the
webapp/CLI/MCP all converge for free, no per-client transcript plumbing; see
the interface-convergence note in CLAUDE.md / COG-128).

COG-131 (thread history) builds on the same store: a signed-in user can list
their past threads and re-open one. Each conversation therefore also records its
``owner`` (the auth subject — the Clerk user id, surfaced generically as
``TenantContext.subject``), a ``title`` (derived from the first user message),
and ``created_at`` so the store can list a user's threads newest-first. Threads
with no owner (the public demo's shared key) simply never appear in anyone's
list — history is an authenticated, per-user feature.

This mirrors :mod:`cograph_client.agent.plan_store`:

- ``ConversationStore`` — an async Protocol so the backend is swappable.
- ``InMemoryConversationStore`` — the zero-config default; non-durable.
- ``PostgresConversationStore`` — durable + shared across ECS tasks over a
  generic Postgres DSN (``settings.database_url``). Vendor-neutral: a plain DSN,
  no cloud-provider identifiers.
- ``make_conversation_store()`` — Postgres when ``settings.database_url`` is set,
  else in-memory.

The stored transcript is bounded to a rolling tail (``_MAX_TURNS``) so a row
can't grow without limit; the planner additionally trims to a smaller window
before grounding the classifier prompt.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from cograph_client.config import settings

# Keep the rolling tail of a dialogue: enough turns to show a meaningful thread
# in the history UI without a row growing without bound. The planner trims to a
# smaller window again before building the classifier prompt.
_MAX_TURNS = 200

# How long a derived thread title may be before it is truncated.
_TITLE_MAX = 80


@dataclass
class Turn:
    """One conversational turn — a user message or an assistant response.

    ``kind``/``intent`` are recorded for assistant turns so the convergence
    guard can count prior ``clarify`` rounds and the classifier prompt can avoid
    re-asking an already-answered dimension.
    """

    role: str  # "user" | "assistant"
    text: str
    kind: Optional[str] = None  # assistant: answer | clarify | plan | result
    intent: Optional[str] = None  # assistant: the chosen intent(s), joined

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(
            role=d.get("role", ""),
            text=d.get("text", ""),
            kind=d.get("kind"),
            intent=d.get("intent"),
        )


@dataclass
class Conversation:
    """A persisted, tenant-scoped rolling transcript keyed by ``session_id``.

    ``owner`` is the auth subject (user id) that started the thread, used to list
    a user's own threads; it is ``None`` for ownerless (demo) sessions.
    """

    session_id: str
    tenant_id: str
    owner: Optional[str] = None
    title: str = ""
    turns: list[Turn] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_json(self) -> str:
        return json.dumps(
            {
                "session_id": self.session_id,
                "tenant_id": self.tenant_id,
                "owner": self.owner,
                "title": self.title,
                "turns": [t.to_dict() for t in self.turns],
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            }
        )

    @classmethod
    def from_payload(cls, payload: Any) -> "Conversation":
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode()
        data = json.loads(payload) if isinstance(payload, str) else payload
        return cls(
            session_id=data["session_id"],
            tenant_id=data.get("tenant_id", ""),
            owner=data.get("owner"),
            title=data.get("title", "") or "",
            turns=[Turn.from_dict(t) for t in data.get("turns", [])],
            created_at=_parse_dt(data.get("created_at")),
            updated_at=_parse_dt(data.get("updated_at")),
        )

    def summary(self) -> "ConversationSummary":
        """A lightweight listing row (no full transcript)."""
        last_user = next(
            (t.text for t in reversed(self.turns) if t.role == "user"), ""
        )
        return ConversationSummary(
            session_id=self.session_id,
            title=self.title or _derive_title(self.turns),
            created_at=self.created_at,
            updated_at=self.updated_at,
            turn_count=len(self.turns),
            preview=last_user[:120],
        )


@dataclass
class ConversationSummary:
    """Metadata for a thread-list entry — no transcript, cheap to send."""

    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    turn_count: int
    preview: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "turn_count": self.turn_count,
            "preview": self.preview,
        }


class ConversationStore(Protocol):
    async def load(self, session_id: str, tenant_id: str) -> list[Turn]: ...
    async def append(
        self,
        session_id: str,
        tenant_id: str,
        turns: list[Turn],
        owner: Optional[str] = None,
    ) -> None: ...
    async def list_for_owner(
        self, tenant_id: str, owner: str
    ) -> list[ConversationSummary]: ...
    async def get(
        self, session_id: str, tenant_id: str, owner: Optional[str] = None
    ) -> Optional[Conversation]: ...


def _parse_dt(raw: Any) -> datetime:
    if not raw:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


def _trim(turns: list[Turn]) -> list[Turn]:
    """Keep only the most recent ``_MAX_TURNS`` (oldest-first ordering)."""
    return turns[-_MAX_TURNS:] if len(turns) > _MAX_TURNS else turns


def _derive_title(turns: list[Turn]) -> str:
    """A human-friendly thread title from the first user message."""
    first = next((t.text for t in turns if t.role == "user" and t.text), "")
    first = " ".join(first.split())  # collapse whitespace/newlines
    if len(first) > _TITLE_MAX:
        first = first[: _TITLE_MAX - 1].rstrip() + "…"
    return first or "New conversation"


def _newest_first(convos: list[Conversation]) -> list[Conversation]:
    _OLDEST = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(convos, key=lambda c: c.updated_at or _OLDEST, reverse=True)


class InMemoryConversationStore:
    """Tenant-scoped in-memory transcript store — the zero-config default.

    Mirrors :class:`~cograph_client.agent.plan_store.InMemoryPlanStore`: an
    ``asyncio.Lock`` guards the dict and reads return copies so a caller can't
    mutate stored state by reference. Transcripts do not survive a process
    restart; use :class:`PostgresConversationStore` for durability.
    """

    def __init__(self) -> None:
        self._convos: dict[tuple[str, str], Conversation] = {}
        self._lock = asyncio.Lock()

    async def load(self, session_id: str, tenant_id: str) -> list[Turn]:
        if not session_id:
            return []
        async with self._lock:
            convo = self._convos.get((tenant_id, session_id))
            if convo is None:
                return []
            return [Turn.from_dict(t.to_dict()) for t in convo.turns]

    async def append(
        self,
        session_id: str,
        tenant_id: str,
        turns: list[Turn],
        owner: Optional[str] = None,
    ) -> None:
        if not session_id or not turns:
            return
        now = datetime.now(timezone.utc)
        async with self._lock:
            convo = self._convos.get((tenant_id, session_id))
            if convo is None:
                convo = Conversation(
                    session_id=session_id,
                    tenant_id=tenant_id,
                    owner=owner,
                    created_at=now,
                )
            merged = _trim([*convo.turns, *turns])
            convo.turns = merged
            convo.updated_at = now
            if owner and not convo.owner:
                convo.owner = owner
            if not convo.title:
                convo.title = _derive_title(merged)
            self._convos[(tenant_id, session_id)] = convo

    async def list_for_owner(
        self, tenant_id: str, owner: str
    ) -> list[ConversationSummary]:
        if not owner:
            return []
        async with self._lock:
            convos = [
                Conversation.from_payload(c.to_json())
                for c in self._convos.values()
                if c.tenant_id == tenant_id and c.owner == owner
            ]
        return [c.summary() for c in _newest_first(convos)]

    async def get(
        self, session_id: str, tenant_id: str, owner: Optional[str] = None
    ) -> Optional[Conversation]:
        if not session_id:
            return None
        async with self._lock:
            convo = self._convos.get((tenant_id, session_id))
            if convo is None:
                return None
            if owner is not None and convo.owner != owner:
                return None  # not this user's thread
            return Conversation.from_payload(convo.to_json())


class PostgresConversationStore:
    """Durable ``ConversationStore`` over a generic Postgres DSN via asyncpg.

    The full rolling transcript is serialized to a ``payload`` jsonb column; the
    columns the agent lists/scopes on (tenant, owner, title, timestamps) are
    mirrored alongside it. The pool + table are created lazily on first use
    (idempotent ``CREATE TABLE``/``ADD COLUMN IF NOT EXISTS``) so importing /
    constructing never touches the network. Vendor-neutral by construction — the
    only configuration is a plain DSN; no cloud-provider ARNs, account ids, or
    hostnames live here.
    """

    _TABLE = "cograph_conversations"

    def __init__(self, dsn: Optional[str] = None) -> None:
        self._dsn = dsn if dsn is not None else settings.database_url
        self._pool: Any = None
        self._lock = asyncio.Lock()

    async def _ensure_pool(self) -> Any:
        if self._pool is not None:
            return self._pool
        async with self._lock:
            if self._pool is not None:
                return self._pool
            import asyncpg  # imported lazily so the dependency is optional

            pool = await asyncpg.create_pool(dsn=self._dsn)
            async with pool.acquire() as conn:
                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._TABLE} (
                        session_id text NOT NULL,
                        tenant_id text NOT NULL,
                        updated_at timestamptz,
                        payload jsonb NOT NULL,
                        PRIMARY KEY (tenant_id, session_id)
                    )
                    """
                )
                # Additive columns for thread history (COG-131) — idempotent so a
                # table created by the COG-130 deploy is upgraded in place.
                await conn.execute(
                    f"ALTER TABLE {self._TABLE} ADD COLUMN IF NOT EXISTS owner text"
                )
                await conn.execute(
                    f"ALTER TABLE {self._TABLE} ADD COLUMN IF NOT EXISTS title text"
                )
                await conn.execute(
                    f"ALTER TABLE {self._TABLE} "
                    f"ADD COLUMN IF NOT EXISTS created_at timestamptz"
                )
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {self._TABLE}_owner_idx "
                    f"ON {self._TABLE} (tenant_id, owner, updated_at DESC)"
                )
            self._pool = pool
            return self._pool

    async def load(self, session_id: str, tenant_id: str) -> list[Turn]:
        if not session_id:
            return []
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT payload FROM {self._TABLE} "
                f"WHERE tenant_id = $1 AND session_id = $2",
                tenant_id,
                session_id,
            )
        if row is None:
            return []
        return Conversation.from_payload(row["payload"]).turns

    async def append(
        self,
        session_id: str,
        tenant_id: str,
        turns: list[Turn],
        owner: Optional[str] = None,
    ) -> None:
        if not session_id or not turns:
            return
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Read-modify-write under a row lock so concurrent turns on the
                # same session don't clobber each other's appends.
                row = await conn.fetchrow(
                    f"SELECT payload FROM {self._TABLE} "
                    f"WHERE tenant_id = $1 AND session_id = $2 FOR UPDATE",
                    tenant_id,
                    session_id,
                )
                if row is not None:
                    convo = Conversation.from_payload(row["payload"])
                else:
                    convo = Conversation(
                        session_id=session_id, tenant_id=tenant_id, owner=owner
                    )
                convo.turns = _trim([*convo.turns, *turns])
                convo.updated_at = datetime.now(timezone.utc)
                if owner and not convo.owner:
                    convo.owner = owner
                if not convo.title:
                    convo.title = _derive_title(convo.turns)
                await conn.execute(
                    f"""
                    INSERT INTO {self._TABLE}
                        (session_id, tenant_id, owner, title,
                         created_at, updated_at, payload)
                    VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                    ON CONFLICT (tenant_id, session_id) DO UPDATE SET
                        owner = COALESCE({self._TABLE}.owner, EXCLUDED.owner),
                        title = COALESCE(NULLIF({self._TABLE}.title, ''),
                                         EXCLUDED.title),
                        updated_at = EXCLUDED.updated_at,
                        payload = EXCLUDED.payload
                    """,
                    session_id,
                    tenant_id,
                    convo.owner,
                    convo.title,
                    convo.created_at,
                    convo.updated_at,
                    convo.to_json(),
                )

    async def list_for_owner(
        self, tenant_id: str, owner: str
    ) -> list[ConversationSummary]:
        if not owner:
            return []
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT payload FROM {self._TABLE} "
                f"WHERE tenant_id = $1 AND owner = $2 "
                f"ORDER BY updated_at DESC",
                tenant_id,
                owner,
            )
        return [Conversation.from_payload(r["payload"]).summary() for r in rows]

    async def get(
        self, session_id: str, tenant_id: str, owner: Optional[str] = None
    ) -> Optional[Conversation]:
        if not session_id:
            return None
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT payload FROM {self._TABLE} "
                f"WHERE tenant_id = $1 AND session_id = $2",
                tenant_id,
                session_id,
            )
        if row is None:
            return None
        convo = Conversation.from_payload(row["payload"])
        if owner is not None and convo.owner != owner:
            return None
        return convo


_store: Optional[InMemoryConversationStore] = None
_durable_store: Optional[PostgresConversationStore] = None


def make_conversation_store() -> ConversationStore:
    """Select the conversation-store backend from configuration.

    Returns a :class:`PostgresConversationStore` when ``settings.database_url``
    is set (durable, shared across ECS tasks), else an
    :class:`InMemoryConversationStore`. Both are process-level singletons so the
    durable backend owns one asyncpg pool per process (created lazily — calling
    this never touches the network). Mirrors
    :func:`cograph_client.agent.plan_store.make_plan_store`.
    """
    global _store, _durable_store
    if settings.database_url:
        if _durable_store is None:
            _durable_store = PostgresConversationStore()
        return _durable_store
    if _store is None:
        _store = InMemoryConversationStore()
    return _store


def reset_conversation_store() -> None:
    """Test helper — clear both singletons."""
    global _store, _durable_store
    _store = None
    _durable_store = None


__all__ = [
    "Conversation",
    "ConversationStore",
    "ConversationSummary",
    "InMemoryConversationStore",
    "PostgresConversationStore",
    "Turn",
    "make_conversation_store",
    "reset_conversation_store",
]
