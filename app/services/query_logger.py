"""Structured JSONL logging of backend chat/retrieval queries.

One `backend_query` event is appended per chat request, for offline
evaluation and paper analysis. Writing must never break a chat response:
all failures are caught here and downgraded to a warning log line.
"""

import asyncio
import hashlib
import json
import math
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings

logger = logging.getLogger("app.query_log")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class QueryLogger:
    """Appends one structured JSON object per line to a JSONL file."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.settings, "QUERY_LOG_ENABLED", True))

    def _write_line_sync(self, path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")

    async def _log_jsonl(self, *, path: Path, event: dict[str, Any], label: str) -> None:
        if not self.enabled:
            return
        try:
            payload = self._compact_vectors(event) if self.settings.QUERY_LOG_COMPACT_VECTORS else event
            line = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            logger.warning(
                "query_logger: failed to serialize %s event",
                label,
                exc_info=True,
            )
            return
        try:
            async with self._lock:
                await asyncio.to_thread(self._write_line_sync, path, line)
        except Exception:
            logger.warning(
                "query_logger: failed to write %s event",
                label,
                exc_info=True,
            )

    @classmethod
    def _compact_vectors(cls, value: Any) -> Any:
        """Replace large numeric arrays with reproducible metadata.

        Historical files remain untouched. Short numeric lists such as years,
        scores, and IDs keep their original wire shape.
        """
        if isinstance(value, list):
            if len(value) >= 128 and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
                encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
                return {
                    "omitted": True,
                    "dimension": len(value),
                    "l2_norm": round(math.sqrt(sum(float(item) ** 2 for item in value)), 6),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                }
            return [cls._compact_vectors(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._compact_vectors(item) for key, item in value.items()}
        return value

    async def log_event(self, event: dict[str, Any]) -> None:
        await self._log_jsonl(
            path=self.settings.query_log_path_resolved,
            event=event,
            label="backend_query",
        )

    async def log_frontend_event(self, event: dict[str, Any]) -> None:
        await self._log_jsonl(
            path=self.settings.frontend_event_log_path_resolved,
            event=event,
            label="frontend_interaction",
        )

    async def log_speech_session(self, event: dict[str, Any]) -> None:
        await self._log_jsonl(
            path=self.settings.speech_log_path_resolved,
            event=event,
            label="speech_session",
        )


@lru_cache(maxsize=1)
def get_query_logger() -> QueryLogger:
    return QueryLogger(get_settings())
