from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import re
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings



@dataclass(frozen=True)
class _TourNavigationEntry:
    overlay_id: str
    panorama_key: str
    inventory_id: str
    location: str | None = None
    title: str | None = None


class TourNavigationService:
    """Resolves tour navigation targets from inventory ids using poi_tours JSON files."""

    _INVENTORY_SPLIT_RE = re.compile(r"[;,\n|]+")
    _NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._inventory_index: dict[str, dict[str, list[_TourNavigationEntry]]] = {}
        self._loaded_museums: set[str] = set()

    def _normalize_inventory(self, raw: str) -> str:
        cleaned = (raw or "").upper().strip()
        if not cleaned:
            return ""
        return self._NON_ALNUM_RE.sub("", cleaned)

    def _extract_candidates(self, inventory_text: str) -> list[str]:
        raw = (inventory_text or "").strip()
        if not raw:
            return []

        candidates: list[str] = [raw]
        without_parenthesis = re.sub(r"\([^)]*\)", " ", raw).strip()
        if without_parenthesis and without_parenthesis != raw:
            candidates.append(without_parenthesis)
        for token in self._INVENTORY_SPLIT_RE.split(raw):
            value = token.strip()
            if value:
                candidates.append(value)

        seen: set[str] = set()
        unique: list[str] = []
        for value in candidates:
            normalized = self._normalize_inventory(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(value)
        return unique

    def _mapping_file_candidates(self, museum_slug: str, museum_id: str | None) -> list[Path]:
        base_dir = self.settings.poi_tours_dir_resolved
        values: list[str] = []

        for value in (museum_slug, museum_id):
            cleaned = (value or "").strip().lower()
            if cleaned and cleaned not in values:
                values.append(cleaned)

        return [base_dir / f"panorama-overlays-inventory-{value}.json" for value in values]

    def _ensure_loaded(self, museum_slug: str, museum_id: str | None) -> None:
        museum_key = (museum_slug or "").strip().lower()
        if not museum_key:
            return
        if museum_key in self._loaded_museums:
            return

        index: dict[str, list[_TourNavigationEntry]] = {}
        loaded_any = False

        for file_path in self._mapping_file_candidates(museum_slug=museum_slug, museum_id=museum_id):
            if not file_path.exists() or not file_path.is_file():
                continue
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, list):
                continue

            loaded_any = True
            for item in payload:
                if not isinstance(item, dict):
                    continue
                overlay_id = str(item.get("overlayId") or "").strip()
                panorama_key = str(item.get("panoramaKey") or "").strip()
                if not overlay_id or not panorama_key:
                    continue

                inventory_ids = item.get("inventoryIds")
                if isinstance(inventory_ids, str):
                    inventory_list = [inventory_ids]
                elif isinstance(inventory_ids, list):
                    inventory_list = [str(value).strip() for value in inventory_ids if str(value).strip()]
                else:
                    inventory_list = []
                if not inventory_list:
                    continue

                location = str(item.get("location") or "").strip() or None
                title = str(item.get("title") or "").strip() or None

                for inventory_id in inventory_list:
                    for candidate in self._extract_candidates(inventory_id):
                        normalized = self._normalize_inventory(candidate)
                        if not normalized:
                            continue
                        index.setdefault(normalized, []).append(
                            _TourNavigationEntry(
                                overlay_id=overlay_id,
                                panorama_key=panorama_key,
                                inventory_id=inventory_id,
                                location=location,
                                title=title,
                            )
                        )

        if loaded_any:
            self._inventory_index[museum_key] = index
        else:
            self._inventory_index[museum_key] = {}
        self._loaded_museums.add(museum_key)

    def resolve_targets(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        inventories: list[str],
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        self._ensure_loaded(museum_slug=museum_slug, museum_id=museum_id)
        museum_key = (museum_slug or "").strip().lower()
        index = self._inventory_index.get(museum_key, {})
        if not index:
            return []

        results: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        max_results = max(limit, 1)

        for inventory_text in inventories:
            for candidate in self._extract_candidates(inventory_text):
                normalized = self._normalize_inventory(candidate)
                if not normalized:
                    continue
                entries = index.get(normalized, [])
                for entry in entries:
                    key = (entry.overlay_id, entry.panorama_key)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    results.append(
                        {
                            "overlay_id": entry.overlay_id,
                            "panorama_key": entry.panorama_key,
                            "inventory_id": entry.inventory_id,
                            "location": entry.location,
                            "title": entry.title,
                        }
                    )
                    if len(results) >= max_results:
                        return results
        return results


@lru_cache(maxsize=1)
def get_tour_navigation_service() -> TourNavigationService:
    return TourNavigationService(get_settings())
