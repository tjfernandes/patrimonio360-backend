"""Cache dos percursos planeados, por route_id, com TTL (ROUTE_TTL_SECONDS).

Guarda o percurso tal como foi devolvido (para GET /tour/routes/{id} e para a
narração das paragens) e o que é preciso para o ajustar («mais curto», «só
esta sala», «troca esta paragem»): o tema interpretado, os candidatos e o
ponto de partida. Em memória, como a sessão do chat.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from app.core.config import get_settings


@dataclass
class StoredRoute:
    route_id: str
    museum_slug: str
    museum_id: str | None
    museum_name: str | None
    language: str
    conversation_id: str | None
    theme_text: str
    facets: dict[str, Any]
    plan: dict[str, Any]
    candidates: list[dict[str, Any]]
    start_panorama: str | None
    created_at: float = field(default_factory=time.time)
    parent_route_id: str | None = None


class RouteStore:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = max(int(ttl_seconds), 60)
        self._routes: dict[str, StoredRoute] = {}
        self._lock = threading.Lock()

    def _evict(self) -> None:
        now = time.time()
        for key in [k for k, r in self._routes.items() if now - r.created_at > self.ttl_seconds]:
            self._routes.pop(key, None)

    def put(self, route: StoredRoute) -> None:
        with self._lock:
            self._evict()
            self._routes[route.route_id] = route

    def get(self, route_id: str | None) -> StoredRoute | None:
        if not route_id:
            return None
        with self._lock:
            self._evict()
            return self._routes.get(route_id)

    def __len__(self) -> int:
        with self._lock:
            self._evict()
            return len(self._routes)


@lru_cache(maxsize=1)
def get_route_store() -> RouteStore:
    return RouteStore(int(getattr(get_settings(), "ROUTE_TTL_SECONDS", 7200)))
