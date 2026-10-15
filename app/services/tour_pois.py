"""Os mapas dos POIs das visitas, poi_tours/panorama-overlays-inventory-<museu>.json.

São a fonte única do hotspot de cada peça (overlayId, panoramaKey, os n.ºs do
catálogo em inventoryIds) e de se a peça é Tesouro Nacional (`nationalTreasure`:
o selo na página do POI, escrito pelos scripts de tools/poi_mapping). Relidos
quando um ficheiro muda, sem reiniciar.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_PREFIX = "panorama-overlays-inventory-"
_SLUG_RE = re.compile(r"^[a-z0-9_-]{1,40}$")


def normalize_slug(value: str | None) -> str | None:
    slug = (value or "").strip().lower()
    return slug if _SLUG_RE.match(slug) else None


def public_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """O que o frontend recebe de cada hotspot (GET /api/v1/chat/tour/pois)."""
    inventory_ids = entry.get("inventoryIds")
    if isinstance(inventory_ids, str):
        inventory_ids = [inventory_ids]
    return {
        "overlayId": str(entry.get("overlayId") or ""),
        "panoramaKey": str(entry.get("panoramaKey") or ""),
        "inventoryIds": [str(value).strip() for value in (inventory_ids or []) if str(value).strip()],
        "location": entry.get("location"),
        "title": entry.get("title"),
        "nationalTreasure": entry.get("nationalTreasure") is True,
    }


class TourPoiStore:
    """Os ficheiros de uma pasta poi_tours, relidos quando algum muda (vista,
    no máximo, de `check_interval` em `check_interval` segundos)."""

    def __init__(self, directory: Path | str, *, check_interval: float = 2.0) -> None:
        self.directory = Path(directory)
        self.check_interval = check_interval
        self._lock = threading.Lock()
        self._checked_at = float("-inf")
        self._signature: tuple | None = None
        self._by_museum: dict[str, list[dict[str, Any]]] = {}

    def _current_signature(self) -> tuple:
        files = []
        try:
            for path in self.directory.glob(f"{_PREFIX}*.json"):
                stat = path.stat()
                files.append((path.name, stat.st_mtime_ns, stat.st_size))
        except OSError:
            pass
        return tuple(sorted(files))

    def _reload_if_changed(self) -> None:
        now = time.monotonic()
        if now - self._checked_at < self.check_interval:
            return
        with self._lock:
            if now - self._checked_at < self.check_interval:
                return
            signature = self._current_signature()
            self._checked_at = now
            if signature == self._signature:
                return
            by_museum: dict[str, list[dict[str, Any]]] = {}
            for name, _, _ in signature:
                slug = normalize_slug(name[len(_PREFIX):-len(".json")])
                if not slug:
                    continue
                try:
                    payload = json.loads((self.directory / name).read_text(encoding="utf-8"))
                except Exception as exc:  # um ficheiro estragado não pode derrubar o chat
                    logger.warning("tour_pois: %s ilegível (%s); museu sem POIs", name, exc)
                    continue
                if isinstance(payload, list):
                    by_museum[slug] = [item for item in payload if isinstance(item, dict)]
            self._by_museum = by_museum
            self._signature = signature

    @property
    def version(self) -> tuple | None:
        self._reload_if_changed()
        return self._signature

    def museums(self) -> list[str]:
        self._reload_if_changed()
        return sorted(self._by_museum)

    def entries(self, museum: str | None) -> list[dict[str, Any]] | None:
        """As entradas do ficheiro do museu, ou None se o museu não tem visita."""
        self._reload_if_changed()
        slug = normalize_slug(museum)
        if slug is None or slug not in self._by_museum:
            return None
        return self._by_museum[slug]


@lru_cache(maxsize=1)
def get_tour_poi_store() -> TourPoiStore:
    return TourPoiStore(get_settings().poi_tours_dir_resolved)
