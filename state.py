import json
import uuid
from typing import Any, Dict

from cat.db.database import get_async_db

KEY_PREFIX = "autochunk"


def _key(agent_id: str, name: str) -> str:
    return f"{KEY_PREFIX}:{agent_id}:{name}"


class RunLock:
    """Per-agent distributed lock (Redis), so that a single run per agent is active across the workers sharing the
    White Rabbit job store. The lock is owned through a token: it is never released by another run."""

    def __init__(self, agent_id: str, ttl_seconds: int):
        self.key = _key(agent_id, "lock")
        self.ttl_seconds = ttl_seconds
        self.token = uuid.uuid4().hex

    async def acquire(self) -> bool:
        return bool(await get_async_db().set(self.key, self.token, nx=True, ex=self.ttl_seconds))

    async def release(self) -> None:
        db = get_async_db()
        if await db.get(self.key) == self.token:
            await db.delete(self.key)

    @staticmethod
    async def is_locked(agent_id: str) -> bool:
        return bool(await get_async_db().exists(_key(agent_id, "lock")))


async def save_report(agent_id: str, report: Dict[str, Any]) -> None:
    await get_async_db().set(_key(agent_id, "last_report"), json.dumps(report, default=str))


async def load_report(agent_id: str) -> Dict[str, Any] | None:
    raw = await get_async_db().get(_key(agent_id, "last_report"))
    return json.loads(raw) if raw else None


async def save_pending(agent_id: str, pending: Dict[str, Any] | None) -> None:
    """The files whose re-ingestion with the active chunker failed: they are retried by the next run."""
    db = get_async_db()
    if not pending or not pending.get("files"):
        await db.delete(_key(agent_id, "pending"))
        return
    await db.set(_key(agent_id, "pending"), json.dumps(pending, default=str))


async def load_pending(agent_id: str) -> Dict[str, Any] | None:
    raw = await get_async_db().get(_key(agent_id, "pending"))
    return json.loads(raw) if raw else None


async def clear_agent_state(agent_id: str) -> None:
    db = get_async_db()
    await db.delete(_key(agent_id, "last_report"), _key(agent_id, "pending"), _key(agent_id, "lock"))
