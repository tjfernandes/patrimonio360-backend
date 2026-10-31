"""Os objetos identificados em cada panorama, pré-indexados offline.

`tools/apontar/indexar_objetos_panorama.py` percorre as vistas de cada
panorama, recorta os objetos com o SAM 3 por conceito, procura cada recorte
no índice de imagens e guarda só os que se identificam com confiança, com a
posição angular (yaw, pitch) e o contorno em graus. Fica em
poi_tours/panorama-objects-<museu>.json, relido quando muda, e sai por
GET /api/v1/chat/tour/objects. O visitante pede «que peças há aqui?» e o
frontend assinala-as no panorama: nunca toca em nada que não se saiba
identificar (experiência, 2026-10-26).

Desde 2026-10-31 o mesmo ficheiro dá «Ver na tour» às peças sem hotspot:
`locate()` diz em que panorama e para onde (yaw, pitch) uma peça foi
localizada com segurança, e `object_navigation_target()` faz dela um alvo de
navegação (TOUR_OBJECT_TARGETS_ENABLED, chat_service._resolve_target_dicts).
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
        # museu -> inventário normalizado -> objetos dessa peça (todos os panoramas)
        self._by_inventory: dict[str, dict[str, list[dict[str, Any]]]] = {}

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
            self._by_inventory = {}
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

    _CONFIRMED_BY = frozenset({"vlm", "hotspot"})

    @classmethod
    def is_definitive(cls, obj: dict[str, Any], *, min_score: float) -> bool:
        """Localização em que se pode confiar: confirmada pelo VLM ou por
        hotspot, ou aceite só pelo cosseno com score ≥ `min_score`."""
        if str(obj.get("accepted_by") or "") in cls._CONFIRMED_BY:
            return True
        try:
            return float(obj.get("score") or 0.0) >= min_score
        except (TypeError, ValueError):
            return False

    def _inventory_index(self, slug: str) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            index = self._by_inventory.get(slug)
            if index is not None:
                return index
            index = {}
            payload = self._by_museum.get(slug) or {}
            for panorama_key, entry in (payload.get("panoramas") or {}).items():
                if not isinstance(entry, dict):
                    continue
                for obj in entry.get("objects") or []:
                    if not isinstance(obj, dict):
                        continue
                    key = normalize_inventory(obj.get("inventory_number"))
                    if not key:
                        continue
                    located = dict(obj)
                    located["panorama_key"] = str(panorama_key)
                    index.setdefault(key, []).append(located)
            self._by_inventory[slug] = index
            return index

    def locate(self, museum: str | None, inventory: Any, *, min_score: float = 0.80) -> dict[str, Any] | None:
        """Onde a peça `inventory` está na visita: o melhor objeto definitivo
        entre os panoramas em que foi localizada (confirmados primeiro, depois
        pelo cosseno; empate pela área). None se não foi localizada com segurança."""
        payload = self.museum(museum)
        slug = normalize_slug(museum)
        if payload is None or slug is None:
            return None
        key = normalize_inventory(inventory)
        if not key:
            return None
        candidates = [o for o in self._inventory_index(slug).get(key, []) if self.is_definitive(o, min_score=min_score)]
        if not candidates:
            return None

        def rank(obj: dict[str, Any]) -> tuple:
            confirmed = str(obj.get("accepted_by") or "") in self._CONFIRMED_BY
            return (confirmed, float(obj.get("score") or 0.0), float(obj.get("area_fraction") or 0.0))

        return max(candidates, key=rank)


@lru_cache(maxsize=1)
def get_panorama_object_store() -> PanoramaObjectStore:
    return PanoramaObjectStore(get_settings().poi_tours_dir_resolved)


OBJECT_OVERLAY_PREFIX = "object:"


def object_navigation_target(obj: dict[str, Any], *, inventory_id: str, location: str | None = None) -> dict[str, Any]:
    """Alvo de «Ver na tour» a partir de um objeto localizado (`locate`), na
    forma dos alvos dos hotspots (tour_navigation.resolve_targets) mais a
    origem e a direção. O overlay_id é sintético: o frontend exige um, e a
    visita, não o encontrando, vai ao panorama e vira a câmara para yaw/pitch."""
    return {
        "overlay_id": f"{OBJECT_OVERLAY_PREFIX}{obj.get('object_id') or ''}",
        "panorama_key": str(obj.get("panorama_key") or ""),
        "inventory_id": inventory_id,
        "location": location,
        "title": (str(obj.get("title") or "").strip() or None),
        "source": "object",
        "yaw": round(float(obj.get("yaw") or 0.0), 2),
        "pitch": round(float(obj.get("pitch") or 0.0), 2),
        "score": round(float(obj.get("score") or 0.0), 3),
    }


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
