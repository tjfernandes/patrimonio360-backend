"""Os objetos identificados em cada panorama, pré-indexados offline.

`tools/apontar/indexar_objetos_panorama.py` percorre as vistas de cada
panorama, recorta os objetos com o SAM 3 por conceito, procura cada recorte
no índice de imagens e guarda só os que se identificam com confiança, com a
posição angular (yaw, pitch) e o contorno em graus. Fica em
poi_tours/panorama-objects-<museu>.json, relido quando muda, e sai por
GET /api/v1/chat/tour/objects. O visitante pede «que peças há aqui?» e o
frontend assinala-as no panorama: nunca toca em nada que não se saiba
identificar (experiência, 2026-10-26).
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

_PREFIX = "panorama-objects-"
_SLUG_RE = re.compile(r"^[a-z0-9_-]{1,40}$")


def normalize_slug(value: str | None) -> str | None:
    slug = (value or "").strip().lower()
    return slug if _SLUG_RE.match(slug) else None


def normalize_inventory(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def public_object(entry: dict[str, Any], *, hotspot_overlay_id: str | None = None) -> dict[str, Any]:
    """O que o frontend recebe de cada objeto: a peça, a confiança, o centro
    (yaw, pitch em graus; yaw cresce para a direita, pitch para cima), a
    caixa angular e o contorno em graus, pela ordem do contorno."""
    polygon = entry.get("polygon") or []
    bbox = entry.get("bbox") or {}
    return {
        "objectId": str(entry.get("object_id") or ""),
        "artifactId": str(entry.get("artifact_id") or ""),
        "inventoryNumber": entry.get("inventory_number"),
        "title": entry.get("title"),
        "score": round(float(entry.get("score") or 0.0), 3),
        "imageName": entry.get("image_name"),
        "yaw": round(float(entry.get("yaw") or 0.0), 2),
        "pitch": round(float(entry.get("pitch") or 0.0), 2),
        "bbox": {key: round(float(bbox.get(key) or 0.0), 2) for key in ("yaw_min", "yaw_max", "pitch_min", "pitch_max")},
        "polygon": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in polygon if isinstance(p, (list, tuple)) and len(p) == 2],
        "hotspotOverlayId": hotspot_overlay_id,
    }


class PanoramaObjectStore:
    """Os ficheiros panorama-objects-<museu>.json de uma pasta, relidos quando
    algum muda (vista, no máximo, de `check_interval` em `check_interval` s)."""

    def __init__(self, directory: Path | str, *, check_interval: float = 2.0) -> None:
        self.directory = Path(directory)
        self.check_interval = check_interval
        self._lock = threading.Lock()
        self._checked_at = float("-inf")
        self._signature: tuple | None = None
        self._by_museum: dict[str, dict[str, Any]] = {}

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
            by_museum: dict[str, dict[str, Any]] = {}
            for name, _, _ in signature:
                slug = normalize_slug(name[len(_PREFIX):-len(".json")])
                if not slug:
                    continue
                try:
                    payload = json.loads((self.directory / name).read_text(encoding="utf-8"))
                except Exception as exc:  # um ficheiro estragado não pode derrubar o chat
                    logger.warning("panorama_objects: %s ilegível (%s); museu sem objetos", name, exc)
                    continue
                if isinstance(payload, dict) and isinstance(payload.get("panoramas"), dict):
                    by_museum[slug] = payload
            self._by_museum = by_museum
            self._signature = signature

    def museums(self) -> list[str]:
        self._reload_if_changed()
        return sorted(self._by_museum)

    def museum(self, museum: str | None) -> dict[str, Any] | None:
        self._reload_if_changed()
        slug = normalize_slug(museum)
        return self._by_museum.get(slug) if slug else None

    def panorama(self, museum: str | None, panorama_key: str | None) -> dict[str, Any] | None:
        """A entrada do panorama ({"objects": [...], "segmented": n, …}), ou
        None se o museu não tem ficheiro ou o panorama não foi indexado."""
        payload = self.museum(museum)
        if payload is None:
            return None
        entry = (payload.get("panoramas") or {}).get((panorama_key or "").strip())
        return entry if isinstance(entry, dict) else None


@lru_cache(maxsize=1)
def get_panorama_object_store() -> PanoramaObjectStore:
    return PanoramaObjectStore(get_settings().poi_tours_dir_resolved)


def hotspot_overlays_by_inventory(poi_entries: list[dict[str, Any]] | None, panorama_key: str) -> dict[str, str]:
    """inventário normalizado → overlayId dos hotspots deste panorama (para o
    frontend ligar um objeto identificado ao hotspot que já existe)."""
    found: dict[str, str] = {}
    for entry in poi_entries or []:
        if str(entry.get("panoramaKey") or "").strip() != panorama_key:
            continue
        overlay_id = str(entry.get("overlayId") or "").strip()
        inventory_ids = entry.get("inventoryIds")
        if isinstance(inventory_ids, str):
            inventory_ids = [inventory_ids]
        for inventory in inventory_ids or []:
            key = normalize_inventory(inventory)
            if key and overlay_id and key not in found:
                found[key] = overlay_id
    return found
